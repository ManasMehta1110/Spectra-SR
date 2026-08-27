import numpy as np
import rasterio
import torch

from spectra_sr.degradation import DegradationOperator
from spectra_sr.guardrails import ndvi, ndwi, run_guardrails, spectral_angle_mapper


def _real_hr_lr(naip_file, size=64, scale=4):
    with rasterio.open(naip_file) as src:
        hr_raw = src.read(window=rasterio.windows.Window(0, 0, size, size)).astype(np.float32)
    hr = torch.from_numpy(hr_raw).unsqueeze(0) / 255.0
    op = DegradationOperator(n_bands=hr.shape[1], scale=scale)
    with torch.no_grad():
        op.log_sigma.fill_(torch.log(torch.tensor(1.0)))
        y = op.forward(hr)
    return op, hr, y


def test_spectral_angle_mapper_zero_for_identical_spectra():
    a = torch.rand(1, 4, 8, 8) + 0.1
    sam = spectral_angle_mapper(a, a)
    # Two stacked effects set this floor, not just float32 noise: (1) arccos is numerically
    # sensitive right at cos~=1 anyway, and (2) spectral_angle_mapper deliberately clamps to
    # [-1+1e-6, 1-1e-6] rather than [-1, 1] to prevent arccos's unbounded gradient there from
    # exploding during training (see that function's docstring -- this was a real, confirmed
    # training-NaN cause, not a defensive nicety). arccos(1-1e-6) ~= sqrt(2e-6) ~= 0.0014 rad is
    # the resulting guaranteed floor -- still functionally zero (~0.08 degrees), just not exact.
    assert sam.item() < 5e-3


def test_spectral_angle_mapper_positive_for_band_permutation():
    """Swapping bands should genuinely change the spectral *shape* at most pixels, not just
    scale it -- SAM should register a real, nonzero angle, not stay near zero."""
    torch.manual_seed(0)
    a = torch.rand(1, 4, 8, 8) + 0.1
    b = a[:, [3, 2, 1, 0]]  # reversed band order -- a real spectral shape distortion
    sam = spectral_angle_mapper(a, b)
    assert sam.item() > 0.05


def test_ndvi_ndwi_in_sensible_range_on_real_data(naip_primary_file):
    with rasterio.open(naip_primary_file) as src:
        raw = src.read().astype(np.float32)
    x = torch.from_numpy(raw).unsqueeze(0) / 255.0
    v = ndvi(x)
    w = ndwi(x)
    assert v.min() >= -1.01 and v.max() <= 1.01
    assert w.min() >= -1.01 and w.max() <= 1.01


def test_guardrails_all_pass_for_a_self_consistent_prediction(naip_primary_file):
    """The real-world "clean" case: x_hat is exactly the HR image that generated y through the
    operator -- every check should pass, since there's nothing actually wrong here."""
    op, hr, y = _real_hr_lr(naip_primary_file)
    result = run_guardrails(hr, y, op, radiometric_threshold=(0.05, 0.05, 0.05, 0.05))
    assert result.passed, result.checks


def test_guardrails_catch_a_spectrally_corrupted_prediction(naip_primary_file):
    """A genuinely bad prediction (bands scrambled) should fail the spectral/index checks --
    not just run without error."""
    op, hr, y = _real_hr_lr(naip_primary_file)
    corrupted = hr[:, [3, 2, 1, 0]]  # real spectral corruption: band order reversed
    result = run_guardrails(corrupted, y, op, radiometric_threshold=(0.05, 0.05, 0.05, 0.05))
    assert not result.passed
    assert not result.checks["spectral_sam_degrees"][0]


def test_guardrails_catch_geometric_misalignment(naip_primary_file):
    """A prediction that's spatially shifted relative to the real observation should fail the
    geometric check specifically."""
    op, hr, y = _real_hr_lr(naip_primary_file, size=96)
    shifted = torch.roll(hr, shifts=8, dims=-1)  # real spatial shift, not a synthetic flag
    result = run_guardrails(shifted, y, op, radiometric_threshold=(1.0, 1.0, 1.0, 1.0),
                             sam_threshold_deg=90, index_threshold=10)
    assert not result.checks["geometric_shift_px"][0]
