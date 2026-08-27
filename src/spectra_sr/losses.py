"""Loss stack for Stage 3 training. Extends optical_guided_sr.losses (SSIM ported verbatim --
that project's own ablation already found its contribution real and significant, no need to
re-derive it) with terms specific to multispectral, physically-constrained SR. Weights match the
original SPECTRA-SR spec's Stage 3 loss table: Charbonnier 1.0, SSIM 0.2, SAM 0.3, index-
preservation 0.2, re-degradation cycle 0.5.

`gradient` (edge-aware, Sobel-domain) is NOT part of that original table -- added after a real
data-volume ablation (19 -> 47 NAIP tiles, pretrain_run1 vs. pretrain_run2) showed more data gives
a consistent +0.6-0.7dB PSNR gain but does NOT move the downstream NDVI-classification-agreement
check versus naive bicubic (the PS-relevant signal). None of the existing terms explicitly reward
matching sharp edges -- Charbonnier/SSIM are dominated by large flat regions, SAM/index are
per-pixel spectral ratios blind to spatial sharpness. w_gradient=0.3 is a starting weight (same
order as w_sam), not yet tuned -- this run is itself the next isolated-variable experiment.

SAM and index-preservation are NOT reimplemented here -- they reuse
spectra_sr.guardrails.spectral_angle_mapper/ndvi/ndwi directly. Those functions are already
real, differentiable, and tested (test_guardrails.py); duplicating them as separate "loss"
versions would be exactly the kind of copy that drifts out of sync with the tested original.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .degradation import DegradationOperator
from .guardrails import ndvi, ndwi, spectral_angle_mapper


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """sqrt((pred - target)^2 + eps^2), mean-reduced -- smooth L1, the primary fidelity anchor."""
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    window_2d = g @ g.t()
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    """Differentiable 1 - SSIM via a Gaussian sliding window -- ported verbatim from
    optical_guided_sr.losses.SSIMLoss. Already generalizes to any channel count via
    `groups=c`, so no changes needed for 4-band Sentinel-2 vs. the original's 1-band thermal."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
        super().__init__()
        self.window_size, self.sigma, self.data_range = window_size, sigma, data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c = pred.shape[1]
        window = _gaussian_window(self.window_size, self.sigma, c, pred.device, pred.dtype)
        pad = self.window_size // 2
        mu_p = F.conv2d(pred, window, padding=pad, groups=c)
        mu_t = F.conv2d(target, window, padding=pad, groups=c)
        mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
        sig_p2 = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu_p2
        sig_t2 = F.conv2d(target * target, window, padding=pad, groups=c) - mu_t2
        sig_pt = F.conv2d(pred * target, window, padding=pad, groups=c) - mu_pt
        c1, c2 = (0.01 * self.data_range) ** 2, (0.03 * self.data_range) ** 2
        ssim_map = ((2 * mu_pt + c1) * (2 * sig_pt + c2)) / ((mu_p2 + mu_t2 + c1) * (sig_p2 + sig_t2 + c2))
        return 1.0 - ssim_map.mean()


def index_preservation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """|NDVI(pred) - NDVI(target)| + |NDWI(pred) - NDWI(target)|, mean-reduced. Reuses
    guardrails.ndvi/ndwi -- see module docstring."""
    return (ndvi(pred) - ndvi(target)).abs().mean() + (ndwi(pred) - ndwi(target)).abs().mean()


def redegradation_cycle_loss(pred: torch.Tensor, lr_input: torch.Tensor,
                              degradation_operator: DegradationOperator) -> torch.Tensor:
    """||A(pred) - lr_input||^2 -- ties Stage 3 training directly to Stage 0's acceptance-gated
    operator rather than training in isolation from it."""
    return F.mse_loss(degradation_operator.forward(pred), lr_input)


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = _SOBEL_X.t()


def _image_gradients(x: torch.Tensor) -> torch.Tensor:
    """Per-channel Sobel gradient magnitude via depthwise conv2d (groups=c) -- same
    per-channel-independent pattern as SSIMLoss's Gaussian window, so bands aren't
    cross-contaminated. +1e-6 inside the sqrt avoids the same zero-gradient singularity that hit
    spectral_angle_mapper's arccos (see guardrails.py) -- sqrt(x) has unbounded gradient at x=0,
    and flat regions (gx=gy=0) are common in real imagery, not an edge case to ignore.

    Replicate (not zero) padding: zero-padding a real image tile fabricates a sharp fake edge at
    every tile border (the true pixel value vs. an assumed-0 neighbor), which would inflate this
    loss for every real patch regardless of actual sharpness, and would also break the
    brightness-shift invariance this term is built on (a zero-padded border does NOT shift along
    with a uniform intensity offset the way the interior does)."""
    c = x.shape[1]
    kx = _SOBEL_X.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3).expand(c, 1, 3, 3)
    ky = _SOBEL_Y.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3).expand(c, 1, 3, 3)
    x_padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(x_padded, kx, groups=c)
    gy = F.conv2d(x_padded, ky, groups=c)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Charbonnier distance between Sobel gradient magnitudes of pred vs. target -- the
    edge-aware term. A uniform brightness/offset shift has zero spatial gradient everywhere
    (Sobel kernels sum to zero), so this term is blind to exactly the kind of difference the
    existing pixelwise terms overreact to, and sensitive to exactly the kind (blur, lost edges)
    they underweight -- see module docstring."""
    return charbonnier_loss(_image_gradients(pred), _image_gradients(target))


class SpectraCombinedLoss(nn.Module):
    """The full Stage 3 loss stack, weighted per the original spec's Stage 3 table."""

    def __init__(self, degradation_operator: DegradationOperator,
                 w_charbonnier: float = 1.0, w_ssim: float = 0.2, w_sam: float = 0.3,
                 w_index: float = 0.2, w_cycle: float = 0.5, w_gradient: float = 0.3,
                 w_perceptual: float = 0.0):
        super().__init__()
        self.degradation_operator = degradation_operator
        self.ssim = SSIMLoss()
        self.weights = dict(charbonnier=w_charbonnier, ssim=w_ssim, sam=w_sam,
                             index=w_index, cycle=w_cycle, gradient=w_gradient)
        # Off by default (w_perceptual=0.0): constructing it downloads/loads a 528MB VGG16 and
        # costs real GPU memory, so callers that don't want it shouldn't pay for it. Instantiated
        # lazily only when actually weighted.
        self.perceptual = None
        if w_perceptual > 0:
            from .perceptual import VGGPerceptualLoss
            self.perceptual = VGGPerceptualLoss()
            self.weights["perceptual"] = w_perceptual

    def forward(self, pred: torch.Tensor, target: torch.Tensor, lr_input: torch.Tensor) -> dict:
        """Returns a dict of {term_name: value} including "total" -- callers that want per-term
        logging (worth having for a hackathon deck's "here's what each loss term buys you"
        story) get it without a second forward pass."""
        terms = {
            "charbonnier": charbonnier_loss(pred, target),
            "ssim": self.ssim(pred, target),
            "sam": spectral_angle_mapper(pred, target),
            "index": index_preservation_loss(pred, target),
            "cycle": redegradation_cycle_loss(pred, lr_input, self.degradation_operator),
            "gradient": gradient_loss(pred, target),
        }
        if self.perceptual is not None:
            terms["perceptual"] = self.perceptual(pred, target)
        total = sum(self.weights[name] * value for name, value in terms.items())
        terms["total"] = total
        return terms
