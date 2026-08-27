import numpy as np
import pytest
import rasterio
import torch

from spectra_sr.degradation import DegradationOperator
from spectra_sr.model import (
    SMOKE_TEST, OverlappingCrossAttention, SpectraHATCore, window_partition, window_reverse,
)


def test_window_partition_reverse_is_identity():
    """The single most common source of silent bugs in windowed attention -- get the
    partition/reverse indexing wrong and the model still runs, it just scrambles spatial
    information. Verify round-tripping is exact before trusting anything built on top of it."""
    x = torch.randn(2, 8, 16, 16)
    windows = window_partition(x, window_size=4)
    assert windows.shape == (2 * 16, 16, 8)  # (B * n_windows, ws*ws, C)
    reconstructed = window_reverse(windows, window_size=4, h=16, w=16, b=2)
    assert torch.allclose(x, reconstructed)


def test_window_partition_preserves_spatial_identity_not_just_shape():
    """Shape matching alone doesn't catch a transposed H/W or scrambled window order -- use
    distinct per-pixel values (not random noise) so any indexing bug would produce a real,
    detectable mismatch, not just a shape that happens to line up."""
    h = w = 8
    x = torch.arange(h * w, dtype=torch.float32).reshape(1, 1, h, w)
    windows = window_partition(x, window_size=4)
    reconstructed = window_reverse(windows, window_size=4, h=h, w=w, b=1)
    assert torch.equal(x, reconstructed)


def test_overlapping_cross_attention_is_batch_consistent():
    """Real bug this caught: OverlappingCrossAttention built its key/value windows with
    torch.cat(kv_list, dim=0), giving WINDOW-major ordering (index = window*B + b), while
    window_partition returns queries in BATCH-major ordering (index = b*n_windows + window).
    Those coincide only at B=1 -- which is the only batch size this project had ever trained at,
    so it never surfaced. At B>=2 each sample silently attended to keys/values belonging to a
    DIFFERENT sample in the batch: no crash, no shape mismatch, just wrong attention. Any
    per-sample operation must give identical results whether samples are batched or run one at a
    time, so assert exactly that.
    """
    torch.manual_seed(0)
    module = OverlappingCrossAttention(dim=16, window_size=4, n_heads=2).eval()
    x = torch.randn(3, 16, 8, 8)
    with torch.no_grad():
        batched = module(x)
        one_at_a_time = torch.cat([module(x[i:i + 1]) for i in range(x.shape[0])])
    assert torch.allclose(batched, one_at_a_time, atol=1e-6), (
        f"batched vs per-sample differ by {(batched - one_at_a_time).abs().max().item():.6f} -- "
        f"samples are leaking into each other's attention"
    )


def test_full_model_is_batch_consistent():
    """Same invariant one level up: the assembled model must not let batch members influence
    each other. Guards the whole stack, not just the block fixed above -- this matters directly
    for the planned move to larger batch sizes on cloud GPUs, where a batch-mixing bug would look
    like 'the model just doesn't scale' rather than a correctness failure."""
    torch.manual_seed(0)
    model = SpectraHATCore(SMOKE_TEST).eval()
    lr = torch.randn(3, SMOKE_TEST.n_bands, SMOKE_TEST.train_patch_size,
                     SMOKE_TEST.train_patch_size)
    with torch.no_grad():
        batched = model(lr)
        one_at_a_time = torch.cat([model(lr[i:i + 1]) for i in range(lr.shape[0])])
    assert torch.allclose(batched, one_at_a_time, atol=1e-5)


def test_forward_output_shape():
    model = SpectraHATCore(SMOKE_TEST)
    x = torch.randn(1, SMOKE_TEST.n_bands, SMOKE_TEST.train_patch_size, SMOKE_TEST.train_patch_size)
    out = model(x)
    expected = SMOKE_TEST.train_patch_size * SMOKE_TEST.scale
    assert out.shape == (1, SMOKE_TEST.n_bands, expected, expected)


def test_gradients_flow_to_every_parameter():
    """Catches dead branches -- a module that's instantiated but never actually used in
    forward() (e.g. wired up wrong) would pass the shape test above but silently receive zero
    gradient, and nobody would notice until training plateaus for no clear reason."""
    model = SpectraHATCore(SMOKE_TEST)
    x = torch.randn(2, SMOKE_TEST.n_bands, SMOKE_TEST.train_patch_size, SMOKE_TEST.train_patch_size)
    out = model(x)
    loss = out.mean()
    loss.backward()

    dead_params = [name for name, p in model.named_parameters() if p.grad is None]
    assert not dead_params, f"parameters with no gradient at all: {dead_params}"


def test_config_rejects_indivisible_patch_size():
    from spectra_sr.model import HATCoreConfig
    with pytest.raises(ValueError, match="divisible"):
        HATCoreConfig(embed_dim=32, n_groups=1, n_blocks_per_group=1, window_size=8,
                      n_heads=2, train_patch_size=30)  # 30 % 8 != 0


def test_overfits_a_single_real_synthetic_pair(naip_primary_file):
    """The real sanity check: a fresh, untrained architecture should be able to memorize one
    small real training example almost perfectly, given enough steps. If it can't, something
    in the architecture or gradient path is broken, regardless of what the shape/gradient-flow
    tests above say. Uses real NAIP imagery as HR ground truth, degraded through the *already
    real-data-tested* Stage 0 operator to synthesize the LR input -- the same pretrain-tier
    pairing the actual pipeline will use, not a synthetic checkerboard. Pinned to a specific
    file via the naip_primary_file fixture (tests/conftest.py), not glob(...)[0] -- see that
    fixture's docstring for the real flakiness this fixes.
    """
    torch.manual_seed(0)
    patch = SMOKE_TEST.train_patch_size * SMOKE_TEST.scale  # HR crop size

    with rasterio.open(naip_primary_file) as src:
        hr_raw = src.read(window=rasterio.windows.Window(0, 0, patch, patch)).astype(np.float32)
    hr = torch.from_numpy(hr_raw).unsqueeze(0) / 255.0  # NAIP is uint8 -- normalize to [0,1]

    degradation = DegradationOperator(n_bands=SMOKE_TEST.n_bands, scale=SMOKE_TEST.scale)
    with torch.no_grad():
        degradation.log_sigma.fill_(torch.log(torch.tensor(1.0)))
        lr = degradation.forward(hr)

    model = SpectraHATCore(SMOKE_TEST)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 400 steps, not the 150 originally guessed here -- measured the real curve first (see
    # conversation): loss plateaus around step 50-200 (LayerNorm-heavy transformer blocks
    # commonly need a warmup before gradients really reshape the weights), then accelerates,
    # reaching ~67% reduction by step 400 and ~78% by 600. Threshold set from that observed
    # curve with real margin, not picked before knowing how this architecture actually behaves.
    initial_loss = None
    final_loss = None
    for step in range(400):
        optimizer.zero_grad()
        pred = model(lr)
        loss = torch.nn.functional.l1_loss(pred, hr)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.4, (
        f"expected substantial overfitting on one example; initial={initial_loss:.4f} "
        f"final={final_loss:.4f}"
    )
