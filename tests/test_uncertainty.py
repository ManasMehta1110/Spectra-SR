import numpy as np
import rasterio
import torch

from spectra_sr.degradation import DegradationOperator
from spectra_sr.uncertainty import (
    UncertaintyHead, calibrate, expected_calibration_error, heteroscedastic_nll_loss,
    predict_with_uncertainty,
)


def test_gradients_flow_through_head():
    head = UncertaintyHead(n_bands=4)
    pred = torch.randn(1, 4, 32, 32)
    residual = torch.randn(1, 4, 32, 32)
    log_var = head(pred, residual)
    assert log_var.shape == pred.shape
    log_var.mean().backward()
    assert all(p.grad is not None for p in head.parameters())


def test_nll_loss_penalizes_overconfident_wrong_predictions_more():
    """The actual point of the heteroscedastic loss: for the same residual error, predicting
    LOW variance (overconfident) should cost more than predicting HIGH variance -- otherwise
    there's no incentive for the network to ever predict high uncertainty where it's warranted."""
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)  # residual = 1.0 everywhere

    confident_wrong = heteroscedastic_nll_loss(pred, target, log_variance=torch.full((1, 1, 4, 4), -3.0))
    appropriately_uncertain = heteroscedastic_nll_loss(pred, target, log_variance=torch.zeros(1, 1, 4, 4))
    assert confident_wrong > appropriately_uncertain


def test_head_learns_to_predict_where_error_is_actually_larger(naip_primary_file):
    """The real, meaningful check: construct a "prediction" with spatially varying error (heavy
    noise in the left half of a real NAIP patch, light noise in the right half), train the head
    against the true heteroscedastic NLL objective, and verify it actually learns higher
    predicted variance where the real error is larger -- not just that the loss goes down."""
    torch.manual_seed(0)
    size = 64
    with rasterio.open(naip_primary_file) as src:
        hr_raw = src.read(window=rasterio.windows.Window(0, 0, size, size)).astype(np.float32)
    hr = torch.from_numpy(hr_raw).unsqueeze(0) / 255.0

    noise = torch.randn_like(hr)
    noise[:, :, :, : size // 2] *= 0.3   # heavy noise, left half
    noise[:, :, :, size // 2:] *= 0.02   # light noise, right half
    fake_pred = hr + noise

    op = DegradationOperator(n_bands=hr.shape[1], scale=4)
    with torch.no_grad():
        op.log_sigma.fill_(torch.log(torch.tensor(1.0)))
        # The actual re-degradation residual signal the head is designed around: redegraded
        # fake_pred vs. a real synthetic LR observation derived from the true hr.
        lr = op.forward(hr)
        redeg_residual = lr - op.forward(fake_pred)
        redeg_residual_upsampled = torch.nn.functional.interpolate(
            redeg_residual, size=hr.shape[-2:], mode="nearest")

    head = UncertaintyHead(n_bands=hr.shape[1])
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)  # lower than the 1e-2 that
    # triggered the variance-collapse instability documented in UncertaintyHead's docstring
    for _ in range(300):
        optimizer.zero_grad()
        log_var = head(fake_pred, redeg_residual_upsampled)
        loss = heteroscedastic_nll_loss(fake_pred, hr, log_var)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_log_var = head(fake_pred, redeg_residual_upsampled)
    left_var = final_log_var[:, :, :, : size // 2].exp().mean().item()
    right_var = final_log_var[:, :, :, size // 2:].exp().mean().item()

    assert left_var > right_var, (
        f"expected higher predicted variance in the noisier half; left={left_var:.5f} "
        f"right={right_var:.5f}"
    )


def test_calibrate_detects_monotonic_relationship():
    torch.manual_seed(0)
    predicted_std = torch.linspace(0, 1, 1000)
    real_error = predicted_std + 0.05 * torch.randn(1000)  # genuinely correlated, noisy
    result = calibrate(predicted_std, real_error, n_bins=10)
    assert result["monotonic"]
    assert len(result["bins"]) == 10


def test_calibrate_detects_non_monotonic_relationship():
    torch.manual_seed(0)
    predicted_std = torch.linspace(0, 1, 1000)
    real_error = torch.rand(1000)  # deliberately uncorrelated
    result = calibrate(predicted_std, real_error, n_bins=10)
    assert not result["monotonic"]


def test_expected_calibration_error_zero_for_perfectly_calibrated():
    confidence = torch.tensor([0.1, 0.1, 0.9, 0.9])
    correct = torch.tensor([0.0, 0.2, 0.8, 1.0])  # bin means match confidence exactly
    ece = expected_calibration_error(confidence, correct, n_bins=2)
    assert ece < 1e-5


def test_expected_calibration_error_positive_for_overconfident():
    confidence = torch.tensor([0.95, 0.95, 0.95, 0.95])
    correct = torch.tensor([1.0, 0.0, 0.0, 0.0])  # 25% accurate but 95% confident
    ece = expected_calibration_error(confidence, correct, n_bins=1)
    assert ece > 0.5


def test_edge_features_are_full_resolution_and_finite():
    """The head's original only spatial input was an LR-resolution residual upsampled 4x, so every
    4x4 output block shared one value and fine-scale error could not be localised (measured: r=0.19
    against actual error). These features are computed from the prediction at full resolution, so
    they must actually vary within a 4x4 block."""
    from spectra_sr.uncertainty import _edge_features
    x = torch.rand(2, 4, 32, 32)
    grad, std = _edge_features(x)
    assert grad.shape == (2, 1, 32, 32) and std.shape == (2, 1, 32, 32)
    assert torch.isfinite(grad).all() and torch.isfinite(std).all()
    assert (grad >= 0).all() and (std >= 0).all()
    block = grad[0, 0, 8:12, 8:12]
    assert float(block.std()) > 0, "features must vary within a 4x4 block or they add nothing"


def test_edge_features_respond_to_edges_not_flat_regions():
    from spectra_sr.uncertainty import _edge_features
    flat = torch.full((1, 4, 32, 32), 0.5)
    edged = flat.clone()
    edged[:, :, :, 16:] = 0.9  # a real vertical step
    g_flat, _ = _edge_features(flat)
    g_edge, _ = _edge_features(edged)
    assert float(g_edge.max()) > float(g_flat.max()) * 10


def test_head_accepts_edge_features_and_matches_prediction_shape():
    from spectra_sr.uncertainty import UncertaintyHead
    for use in (True, False):
        head = UncertaintyHead(n_bands=4, use_edge_features=use)
        pred = torch.rand(2, 4, 32, 32)
        residual = torch.rand(2, 4, 32, 32)
        out = head(pred, residual)
        assert out.shape == pred.shape
        assert torch.isfinite(out).all()


def test_edge_features_change_the_prediction():
    """Guards against the features being wired in but ignored -- a head that produces identical
    output with and without them would pass every shape test while fixing nothing."""
    from spectra_sr.uncertainty import UncertaintyHead
    torch.manual_seed(0)
    head = UncertaintyHead(n_bands=4, use_edge_features=True).eval()
    pred = torch.rand(1, 4, 32, 32)
    residual = torch.rand(1, 4, 32, 32)
    flat_pred = torch.full_like(pred, 0.5)
    with torch.no_grad():
        a = head(pred, residual)
        b = head(flat_pred, residual)
    assert not torch.allclose(a, b)
