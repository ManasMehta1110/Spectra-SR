"""Stage 3 -- the deterministic core. Plan Section 5, phase 3.

Architecture (locked in conversation, docs/plan.md's "Architecture note"): shallow feature
extraction -> N x Residual Hybrid Attention Group (windowed self-attention + ChannelAttention
reused verbatim from optical_guided_sr.model, since its ablation already showed a real,
significant PSNR contribution -- plus one Overlapping Cross-Attention Block per group for the
long-range repeated-structure modeling that's the actual reason to pick this over
DualEDSRPlus's pure-CNN receptive field) -> global residual over a bicubic-upsampled copy of
the input (same EDSR-family philosophy DualEDSRPlus already uses) -> two chained 2x
LearnedUpsampler stages (ICNR-initialized PixelShuffle, ported from
optical_guided_sr.model._icnr_init/LearnedUpsampler) -> output conv.

HONESTY NOTE: this is a HAT-*inspired* design built from the architectural description agreed
on in conversation, not a byte-exact reproduction of the published HAT paper's reference
implementation (reconstructed from description, not transcribed from source) -- the Hybrid
Attention Block here runs windowed self-attention and a channel-attention branch in parallel and
sums them, which captures the "hybrid" idea the name refers to, but specific implementation
details (exact CAB internals, exact OCAB overlap handling) may differ from the paper. Don't cite
this as matching HAT's published numbers without re-verifying against the actual paper/code.

Deliberately deterministic throughout -- no adversarial or diffusion component in this module,
for the same hallucination-risk reason DualEDSRPlus's own GAN alternative was rejected on the
prior project.

Window attention operates at LR (input) resolution throughout the RHAG stack -- upsampling
happens only at the end -- so `train_patch_size` must be exactly divisible by `window_size`;
enforced in HATCoreConfig.__post_init__ rather than failing confusingly deep inside a forward
pass.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reused verbatim from optical_guided_sr.model (plan Section 1: "kept as the baseline ablation
# variant... The ICNR PixelShuffle init is reused verbatim in any new upsampling head").
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.avgpool(x))


def _icnr_init(weight: torch.Tensor, scale: int, initializer=None) -> None:
    """ICNR initialization for a sub-pixel (PixelShuffle) convolution's weight (Aitken et al.,
    2017). A naive random init gives every one of the `scale**2` output channels that
    PixelShuffle rearranges into one spatial block an independent random kernel, producing a
    checkerboard bias from the very first forward pass. This instead tiles one random kernel
    across all `scale**2` channels in each block, so the initial upsampled output is spatially
    uniform.
    """
    if initializer is None:
        initializer = lambda w: nn.init.kaiming_normal_(w, a=0, mode="fan_in", nonlinearity="relu")
    out_ch, in_ch, kh, kw = weight.shape
    base_ch = out_ch // (scale ** 2)
    base = torch.empty(base_ch, in_ch, kh, kw, device=weight.device, dtype=weight.dtype)
    initializer(base)
    weight.data.copy_(base.repeat_interleave(scale ** 2, dim=0))


class LearnedUpsampler(nn.Module):
    """Sub-pixel convolution (PixelShuffle) upsampler."""

    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.proj = nn.Conv2d(in_ch, out_ch * scale * scale, 3, padding=1)
        # NO terminal activation. This previously ended in nn.ReLU, which was a real, measured
        # bug: this network predicts a RESIDUAL over bicubic, so the feature path must be able to
        # carry negative values (a residual has to darken pixels as well as brighten them). A
        # terminal ReLU also creates a one-way trapdoor -- if the input is driven negative the
        # output is exactly 0, the gradient is exactly 0, and the branch is dead permanently.
        # That is precisely what happened: after 20 epochs the second upsampler emitted
        # |a| mean = 0.000000, std = 0.000000, collapsing the residual to 0.3% of image contrast
        # and making the whole model an expensive bicubic passthrough (it scored 28.25 dB against
        # bicubic's 28.25 dB -- identical to 2 dp). EDSR/SwinIR/HAT upsamplers are conv +
        # PixelShuffle with no activation here, for this reason.
        self.post = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        _icnr_init(self.proj.weight, scale)

    def forward(self, x):
        return self.post(F.pixel_shuffle(self.proj(x), self.scale))


# ---------------------------------------------------------------------------
# New: windowed attention machinery.
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B * n_windows, window_size*window_size, C). H and W must be divisible by
    window_size -- callers (HATCoreConfig) enforce this at config time, not silently pad here."""
    b, c, h, w = x.shape
    x = x.view(b, c, h // window_size, window_size, w // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (b, h//ws, w//ws, ws, ws, c)
    return windows.view(-1, window_size * window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int, b: int) -> torch.Tensor:
    """Inverse of window_partition: (B*n_windows, ws*ws, C) -> (B, C, H, W)."""
    c = windows.shape[-1]
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, c, h, w)


class RelativePositionBias(nn.Module):
    """Learnable bias added to attention logits, indexed by each pair of tokens' relative
    position within a window -- standard Swin-family technique for injecting spatial awareness
    into windowed attention without full positional embeddings."""

    def __init__(self, window_size: int, n_heads: int):
        super().__init__()
        self.window_size = window_size
        n_positions = (2 * window_size - 1) ** 2
        self.table = nn.Parameter(torch.zeros(n_positions, n_heads))
        nn.init.trunc_normal_(self.table, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flat = coords.flatten(1)  # (2, ws*ws)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, ws*ws, ws*ws)
        relative = relative.permute(1, 2, 0).contiguous()  # (ws*ws, ws*ws, 2)
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        index = relative.sum(-1)  # (ws*ws, ws*ws), each entry indexes into `table`
        self.register_buffer("index", index, persistent=False)

    def forward(self) -> torch.Tensor:
        n = self.window_size ** 2
        bias = self.table[self.index.view(-1)].view(n, n, -1)  # (n, n, n_heads)
        return bias.permute(2, 0, 1).contiguous()  # (n_heads, n, n)


class WindowAttention(nn.Module):
    """Standard multi-head self-attention restricted to non-overlapping windows, with a
    relative position bias -- the "long-range repeated structure" modeling piece (field grids,
    urban blocks) that's the actual justification for this architecture over a pure-CNN one."""

    def __init__(self, dim: int, window_size: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos_bias = RelativePositionBias(window_size, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n_windows*B, N, C) where N = window_size**2."""
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, n_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, N, N)
        attn = attn + self.rel_pos_bias().unsqueeze(0)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class OverlappingCrossAttention(nn.Module):
    """Queries from non-overlapping windows, keys/values from a larger overlapping window
    (extended by `overlap_ratio` on each side) -- lets tokens see beyond their own window's
    boundary without the cost of full global attention. This is the specific mechanism the
    architecture discussion identified as HAT's actual edge over plain Swin-based SR for
    field-grid/urban-block-scale structure.
    """

    def __init__(self, dim: int, window_size: int, n_heads: int, overlap_ratio: float = 0.5):
        super().__init__()
        self.window_size = window_size
        self.overlap_size = int(window_size * overlap_ratio)
        self.kv_window_size = window_size + 2 * self.overlap_size
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W). Unlike WindowAttention, operates on the full feature map directly
        (needs spatial context to build the overlapping key/value windows), not pre-partitioned
        windows."""
        b, c, h, w = x.shape
        ws, ov, kv_ws = self.window_size, self.overlap_size, self.kv_window_size

        q_windows = window_partition(x, ws)  # (B*n_windows, ws*ws, C)
        q = self.q_proj(q_windows)

        padded = F.pad(x, [ov, ov, ov, ov], mode="reflect")
        kv_list = []
        for i in range(0, h, ws):
            for j in range(0, w, ws):
                patch = padded[:, :, i:i + kv_ws, j:j + kv_ws]  # (B, C, kv_ws, kv_ws)
                kv_list.append(patch.flatten(2).transpose(1, 2))  # (B, kv_ws*kv_ws, C)
        # stack-then-reshape, NOT cat(dim=0) -- window_partition returns queries in batch-major
        # order (index = b*n_windows + window), while cat(dim=0) over per-window (B, ...) tensors
        # produces window-major order (index = window*B + b). Those coincide only at B=1, so a
        # cat here silently made every sample in a batch>=2 attend to keys/values belonging to a
        # DIFFERENT sample -- no crash, no shape error, just quietly wrong attention. Caught by
        # test_overlapping_cross_attention_is_batch_consistent.
        kv_windows = torch.stack(kv_list, dim=1).reshape(b * len(kv_list), kv_ws * kv_ws, c)
        kv = self.kv_proj(kv_windows)
        k, v = kv.chunk(2, dim=-1)

        bw = q.shape[0]
        q = q.reshape(bw, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bw, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bw, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bw, ws * ws, c)
        out = self.proj(out)
        return window_reverse(out, ws, h, w, b)


class HybridAttentionBlock(nn.Module):
    """Window attention (long-range structure) and a channel-attention branch (the reused,
    already-validated ChannelAttention -- per its own ablation, +4.41 dB when present) running
    in parallel and summed -- the "hybrid" in the name. LayerNorm'd transformer-style residual
    structure around each branch, plus a standard MLP."""

    def __init__(self, dim: int, window_size: int, n_heads: int, mlp_ratio: float = 2.0,
                 cab_scale: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, n_heads)
        self.channel_attn = ChannelAttention(dim)
        self.cab_scale = cab_scale  # EDSR-style small-scale residual, same reasoning as RCAB's
                                     # res_scale=0.1 in optical_guided_sr.model
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)."""
        b, c, h, w = x.shape
        ws = self.window_size

        shortcut = x
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_norm = self.norm1(x_flat).transpose(1, 2).reshape(b, c, h, w)

        windows = window_partition(x_norm, ws)
        attn_out = self.attn(windows)
        attn_out = window_reverse(attn_out, ws, h, w, b)

        cab_out = self.channel_attn(x_norm) * self.cab_scale

        x = shortcut + attn_out + cab_out

        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.transpose(1, 2).reshape(b, c, h, w)


class ResidualHybridAttentionGroup(nn.Module):
    """N HybridAttentionBlocks + one OverlappingCrossAttention block + a closing conv (injects
    local/convolutional inductive bias back in after the attention layers, standard in the HAT
    lineage), wrapped in a group-level residual."""

    def __init__(self, dim: int, n_blocks: int, window_size: int, n_heads: int,
                 res_scale: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            HybridAttentionBlock(dim, window_size, n_heads) for _ in range(n_blocks)
        ])
        self.ocab = OverlappingCrossAttention(dim, window_size, n_heads)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        # EDSR-style residual scaling (Lim et al. 2017), which that paper introduced specifically
        # because deep residual SR networks are unstable without it. Absent here previously, and
        # the cost was measured directly: activation magnitude compounded ~2x per group through a
        # trained network (0.23 -> 2.3 -> 5.1 -> 9.9 -> 16.9 -> 26.1 -> 40.3), which then drove
        # the upsampler's ReLU permanently negative and killed the residual branch. Note the
        # inner CAB branch already used a 0.1 scale for exactly this reason -- it just was never
        # applied at group level, where the compounding actually happens.
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        for block in self.blocks:
            x = block(x)
        x = x + self.ocab(x)
        x = self.conv(x)
        return x * self.res_scale + shortcut


# ---------------------------------------------------------------------------
# Full network + configs.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HATCoreConfig:
    """Two starting configs (locked in conversation). Start with `COLAB_REALISTIC` and only
    scale to `FULL` once it clears the Phase 3 gate (beats DualEDSRPlus on held-out synthetic
    pairs) -- Colab Pro is a single, session-limited instance, not unlimited compute, despite
    the "no compute constraints" framing used during architecture design (plan Section 2). The
    actual dev laptop this was built on has 4.3GB VRAM (RTX 3050) -- confirms the plan's "laptops
    are dev-only" assumption directly, and by more than expected: measured COLAB_REALISTIC at
    batch=1 -> 1.86GB, batch=2 -> 3.70GB (right at the edge), batch=4 -> 7.36GB (only "succeeded"
    via Windows' shared-GPU-memory/system-RAM fallback, not real dedicated VRAM -- effectively
    unusably slow for real training). FULL genuinely OOM'd even at batch=4. Use SMOKE_TEST or
    COLAB_REALISTIC-at-batch<=1 for anything run on a dev laptop; real training needs Colab Pro.
    """

    embed_dim: int
    n_groups: int
    n_blocks_per_group: int
    window_size: int
    n_heads: int
    train_patch_size: int
    n_bands: int = 4  # B02/B03/B04/B08, per config.Config.bands
    scale: int = 4    # 10m -> 2.5m
    # EDSR-style residual scaling applied at each ResidualHybridAttentionGroup. Exposed on the
    # config (rather than left as a constructor default) because it is a real architecture knob
    # worth sweeping: without it, activations compounded ~2x per group and killed the residual
    # branch outright -- see ResidualHybridAttentionGroup and LearnedUpsampler.
    res_scale: float = 0.1

    def __post_init__(self):
        if self.train_patch_size % self.window_size != 0:
            raise ValueError(
                f"train_patch_size ({self.train_patch_size}) must be divisible by "
                f"window_size ({self.window_size}) -- window attention operates at LR "
                f"resolution and cannot partition an indivisible feature map."
            )
        if self.scale % 2 != 0:
            raise ValueError(
                f"scale ({self.scale}) must be even -- the upsampler chains 2x stages "
                f"(LearnedUpsampler, reused from optical_guided_sr.model)."
            )


FULL = HATCoreConfig(embed_dim=180, n_groups=6, n_blocks_per_group=6, window_size=16, n_heads=6,
                      train_patch_size=128)
COLAB_REALISTIC = HATCoreConfig(embed_dim=112, n_groups=4, n_blocks_per_group=4, window_size=8,
                                 n_heads=4, train_patch_size=96)
# Not in the original locked config table -- added for local smoke-testing on the 4.3GB dev
# laptop, where even COLAB_REALISTIC's window attention over a 96x96 feature map is more than
# needed just to verify the architecture runs and learns at all.
SMOKE_TEST = HATCoreConfig(embed_dim=32, n_groups=2, n_blocks_per_group=2, window_size=8,
                            n_heads=2, train_patch_size=32)


class SpectraHATCore(nn.Module):
    """Takes a single fused latent as input -- agnostic to whether it came from one frame or
    Stage 2's (stretch-tier) multi-temporal fusion, so Stage 2 can slot in front later without
    changing this class. Predicts a residual over a bicubic-upsampled copy of the input (same
    EDSR-family philosophy DualEDSRPlus already uses), not the full HR image outright.
    """

    def __init__(self, config: HATCoreConfig = COLAB_REALISTIC):
        super().__init__()
        self.config = config
        c, dim = config.n_bands, config.embed_dim

        self.shallow = nn.Conv2d(c, dim, 3, padding=1)
        self.groups = nn.ModuleList([
            ResidualHybridAttentionGroup(dim, config.n_blocks_per_group, config.window_size,
                                          config.n_heads, res_scale=config.res_scale)
            for _ in range(config.n_groups)
        ])
        self.deep_conv = nn.Conv2d(dim, dim, 3, padding=1)

        n_upsample_stages = config.scale.bit_length() - 1  # scale=4 -> 2 stages of 2x
        self.upsamplers = nn.ModuleList([
            LearnedUpsampler(dim, dim, scale=2) for _ in range(n_upsample_stages)
        ])
        self.output_conv = nn.Conv2d(dim, c, 3, padding=1)
        # Zero-init the residual's final projection so the network STARTS as exactly bicubic and
        # grows a residual only insofar as it reduces loss. Standard practice for residual
        # branches (FixUp; "Bag of Tricks" zero-init-gamma; ControlNet's zero convolutions;
        # diffusion output layers). Motivated here by a measured failure: at random init the
        # residual was 57% of image contrast -- huge garbage that dominates early loss, so the
        # fastest descent direction is to suppress the branch rather than teach it, and once
        # suppressed it cannot recover. Starting at zero removes that incentive entirely: every
        # unit of residual the model grows has to pay for itself.
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        """lr: (B, n_bands, H, W) -> (B, n_bands, H*scale, W*scale)."""
        shallow_feat = self.shallow(lr)

        x = shallow_feat
        for group in self.groups:
            x = group(x)
        x = self.deep_conv(x) + shallow_feat  # long skip: global residual over deep features

        for upsampler in self.upsamplers:
            x = upsampler(x)
        residual = self.output_conv(x)

        bicubic = F.interpolate(lr, scale_factor=self.config.scale, mode="bicubic",
                                 align_corners=False)
        return bicubic + residual

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
