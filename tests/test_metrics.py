import numpy as np
import rasterio
import torch

from spectra_sr.degradation import DegradationOperator
from spectra_sr.metrics import compute_metrics, downstream_classification_agreement


def test_compute_metrics_perfect_for_identical_inputs():
    x = torch.rand(1, 4, 32, 32) * 0.5 + 0.25
    m = compute_metrics(x, x)
    assert m.psnr == 100.0
    assert m.ssim > 0.999
    assert m.rmse < 1e-6
    # Same arccos-boundary floor as spectral_angle_mapper elsewhere (~0.0014 rad ~= 0.08 deg,
    # from clamping to [-1+1e-6, 1-1e-6] to prevent arccos's unbounded gradient at the exact
    # boundary) -- this test's threshold was first written in the wrong units (1e-2 deg, tighter
    # than the real floor), not a new bug.
    assert m.sam_degrees < 0.5


def test_compute_metrics_reports_real_degradation(naip_primary_file):
    with rasterio.open(naip_primary_file) as src:
        hr_raw = src.read(window=rasterio.windows.Window(0, 0, 64, 64)).astype(np.float32)
    hr = torch.from_numpy(hr_raw).unsqueeze(0) / 255.0

    op = DegradationOperator(n_bands=4, scale=4)
    with torch.no_grad():
        op.log_sigma.fill_(torch.log(torch.tensor(2.0)))
        blurred_at_hr_res = torch.nn.functional.interpolate(
            op.forward(hr), size=hr.shape[-2:], mode="bicubic", align_corners=False)

    m = compute_metrics(blurred_at_hr_res, hr)
    # A genuinely blurred version of the same real image should score worse than perfect, but
    # not nonsensically -- sanity-bound rather than pin to one exact number.
    assert 5.0 < m.psnr < 60.0
    assert 0.0 < m.ssim < 0.999
    assert m.rmse > 1e-4


def test_downstream_agreement_is_perfect_when_prediction_equals_target(naip_primary_file):
    with rasterio.open(naip_primary_file) as src:
        hr_raw = src.read(window=rasterio.windows.Window(0, 0, 64, 64)).astype(np.float32)
    hr = torch.from_numpy(hr_raw).unsqueeze(0) / 255.0
    lr = torch.nn.functional.avg_pool2d(hr, kernel_size=4)

    result = downstream_classification_agreement(sr_pred=hr, lr_input=lr, hr_target=hr)
    assert result.sr_agreement == 1.0


def test_downstream_agreement_survives_cross_sensor_brightness_mismatch(naip_primary_file):
    """Real bug this caught: pretrain_run5 (SEN2NAIPCrossSensorDataset, real Sentinel-2 LR vs.
    real NAIP HR) showed baseline_agreement=0.256 -- worse than a coin flip -- purely because the
    two real sensors' pixel values aren't calibrated to the same brightness/contrast scale
    (LR mean ~0.16 vs. HR mean ~0.56 in the real data), not because bicubic lost spatial detail.
    Simulate that here with a synthetic brightness/contrast shift on an otherwise-identical LR
    (so a perfect bicubic reconstruction would be trivially correct if radiometry were matched)
    and verify the baseline no longer collapses to near-random just from the shift.
    """
    with rasterio.open(naip_primary_file) as src:
        hr_raw = src.read(window=rasterio.windows.Window(0, 0, 64, 64)).astype(np.float32)
    hr = torch.from_numpy(hr_raw).unsqueeze(0) / 255.0
    lr = torch.nn.functional.avg_pool2d(hr, kernel_size=4)

    # Simulate a different sensor's radiometry: dim and low-contrast relative to hr, same
    # structure/content otherwise -- exactly the real ~0.16-vs-~0.56 mean mismatch found in
    # pretrain_run5.
    dim_lr = lr * 0.3 + 0.05

    result = downstream_classification_agreement(sr_pred=hr, lr_input=dim_lr, hr_target=hr)
    assert result.baseline_agreement > 0.7, (
        f"radiometric mismatch alone should no longer tank baseline_agreement to near-random "
        f"(got {result.baseline_agreement:.3f})"
    )


def test_downstream_agreement_degrades_for_a_genuinely_wrong_prediction(naip_primary_file):
    with rasterio.open(naip_primary_file) as src:
        hr_raw = src.read(window=rasterio.windows.Window(0, 0, 64, 64)).astype(np.float32)
    hr = torch.from_numpy(hr_raw).unsqueeze(0) / 255.0
    lr = torch.nn.functional.avg_pool2d(hr, kernel_size=4)
    wrong_pred = hr[:, [3, 2, 1, 0]]  # real spectral corruption -- NDVI computed from wrong bands

    result = downstream_classification_agreement(sr_pred=wrong_pred, lr_input=lr, hr_target=hr)
    assert result.sr_agreement < 1.0
    assert np.isfinite(result.improvement)
