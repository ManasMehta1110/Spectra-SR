import numpy as np
import rasterio
import torch

from spectra_sr.degradation import DegradationOperator
from spectra_sr.losses import (
    SpectraCombinedLoss, SSIMLoss, charbonnier_loss, gradient_loss, index_preservation_loss,
    redegradation_cycle_loss,
)


def test_charbonnier_near_zero_for_identical_inputs():
    x = torch.rand(1, 4, 8, 8)
    loss = charbonnier_loss(x, x)
    assert loss.item() < 1e-2  # eps=1e-3 default means it's never exactly 0, by design


def test_ssim_loss_near_zero_for_identical_inputs():
    x = torch.rand(1, 4, 16, 16)
    loss = SSIMLoss()(x, x)
    assert loss.item() < 1e-4


def test_index_preservation_zero_for_identical_inputs():
    x = torch.rand(1, 4, 8, 8) + 0.1
    loss = index_preservation_loss(x, x)
    assert loss.item() < 1e-5


def test_redegradation_cycle_zero_for_self_consistent_pair(naip_primary_file):
    with rasterio.open(naip_primary_file) as src:
        hr_raw = src.read(window=rasterio.windows.Window(0, 0, 64, 64)).astype(np.float32)
    hr = torch.from_numpy(hr_raw).unsqueeze(0) / 255.0

    op = DegradationOperator(n_bands=hr.shape[1], scale=4)
    with torch.no_grad():
        op.log_sigma.fill_(torch.log(torch.tensor(1.0)))
        lr = op.forward(hr)

    loss = redegradation_cycle_loss(hr, lr, op)
    assert loss.item() < 1e-8  # hr *is* the thing that produced lr -- should be exact


def test_gradient_loss_near_zero_for_identical_inputs():
    x = torch.rand(1, 4, 16, 16)
    loss = gradient_loss(x, x)
    assert loss.item() < 1e-2  # charbonnier-style floor, same as test_charbonnier's


def test_gradient_loss_ignores_uniform_brightness_shift():
    """The real point of this term: a uniform intensity offset changes every pixel value but no
    edge (Sobel kernels sum to zero, so a constant shift has zero spatial gradient everywhere) --
    verify the loss stays near its identical-input floor even though a plain pixelwise loss
    reacts strongly, which is what makes this term complementary rather than redundant."""
    x = torch.rand(1, 4, 16, 16) * 0.5 + 0.25
    shifted = x + 0.3
    grad = gradient_loss(x, shifted)
    pixel = charbonnier_loss(x, shifted)
    assert grad.item() < pixel.item() * 0.1


def test_gradient_loss_detects_real_blur(naip_primary_file):
    """Converse check: a genuinely blurred version of the same real image (lost edges) should
    score clearly worse than the near-zero identical-input floor."""
    with rasterio.open(naip_primary_file) as src:
        hr_raw = src.read(window=rasterio.windows.Window(0, 0, 64, 64)).astype(np.float32)
    hr = torch.from_numpy(hr_raw).unsqueeze(0) / 255.0
    blurred = torch.nn.functional.avg_pool2d(hr, kernel_size=5, stride=1, padding=2)

    loss = gradient_loss(blurred, hr)
    assert loss.item() > 1e-3


def test_combined_loss_includes_weighted_gradient_term(naip_primary_file):
    with rasterio.open(naip_primary_file) as src:
        hr_raw = src.read(window=rasterio.windows.Window(0, 0, 64, 64)).astype(np.float32)
    hr = torch.from_numpy(hr_raw).unsqueeze(0) / 255.0

    op = DegradationOperator(n_bands=hr.shape[1], scale=4)
    with torch.no_grad():
        op.log_sigma.fill_(torch.log(torch.tensor(1.0)))
        lr = op.forward(hr)

    criterion = SpectraCombinedLoss(op, w_gradient=0.3)
    terms = criterion(hr, hr, lr)
    assert "gradient" in terms
    assert terms["gradient"].item() < 1e-2


def test_combined_loss_decreases_during_real_training(naip_primary_file):
    """The real integration check: does the full weighted loss stack, all five terms together,
    actually train a real model on real data -- not just that each term is individually
    well-behaved in isolation."""
    torch.manual_seed(0)
    from spectra_sr.model import SMOKE_TEST, SpectraHATCore

    patch = SMOKE_TEST.train_patch_size * SMOKE_TEST.scale
    with rasterio.open(naip_primary_file) as src:
        hr_raw = src.read(window=rasterio.windows.Window(0, 0, patch, patch)).astype(np.float32)
    hr = torch.from_numpy(hr_raw).unsqueeze(0) / 255.0

    op = DegradationOperator(n_bands=SMOKE_TEST.n_bands, scale=SMOKE_TEST.scale)
    with torch.no_grad():
        op.log_sigma.fill_(torch.log(torch.tensor(1.0)))
        lr = op.forward(hr)

    model = SpectraHATCore(SMOKE_TEST)
    criterion = SpectraCombinedLoss(op)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(400):
        optimizer.zero_grad()
        pred = model(lr)
        terms = criterion(pred, hr, lr)
        terms["total"].backward()
        optimizer.step()
        losses.append(terms["total"].item())

    assert losses[-1] < losses[0] * 0.5, f"initial={losses[0]:.4f} final={losses[-1]:.4f}"
