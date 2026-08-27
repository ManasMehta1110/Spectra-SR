import numpy as np
import rasterio
import torch

from spectra_sr.degradation import DegradationOperator
from spectra_sr.projection import project_to_data_consistency


def _real_hr_lr_pair(naip_file, scale=4, size=128):
    with rasterio.open(naip_file) as src:
        hr_raw = src.read(window=rasterio.windows.Window(0, 0, size, size)).astype(np.float32)
    hr = torch.from_numpy(hr_raw).unsqueeze(0) / 255.0

    op = DegradationOperator(n_bands=hr.shape[1], scale=scale)
    with torch.no_grad():
        op.log_sigma.fill_(torch.log(torch.tensor(1.0)))
        y = op.forward(hr)
    return op, hr, y


def test_output_shape_matches_prediction(naip_primary_file):
    op, hr, y = _real_hr_lr_pair(naip_primary_file)
    x_hat = hr + 0.1 * torch.randn_like(hr)  # a stand-in "imperfect model prediction"
    corrected = project_to_data_consistency(x_hat, y, op)
    assert corrected.shape == x_hat.shape


def test_projection_improves_data_consistency_for_an_imperfect_prediction(naip_primary_file):
    """The actual point of Stage 5: given a genuinely imperfect prediction (real HR plus real
    noise, standing in for what an undertrained/imperfect model would produce), the projected
    output should re-degrade closer to the real observation y than the unprojected prediction
    did. This is the property the null-space framing's "structurally incapable of contradicting
    the input" claim actually depends on -- not just that the function runs."""
    op, hr, y = _real_hr_lr_pair(naip_primary_file)
    torch.manual_seed(0)
    x_hat = hr + 0.15 * torch.randn_like(hr)  # deliberately corrupted prediction

    error_before = torch.nn.functional.mse_loss(op.forward(x_hat), y).item()
    corrected = project_to_data_consistency(x_hat, y, op)
    error_after = torch.nn.functional.mse_loss(op.forward(corrected), y).item()

    assert error_after < error_before, (
        f"projection should improve data-consistency; before={error_before:.5f} "
        f"after={error_after:.5f}"
    )


def test_projection_of_a_near_perfect_prediction_stays_near_it(naip_primary_file):
    """Sanity/stability check: if x_hat already closely reproduces y when degraded (as close to
    "the model got it right" as this test can construct), the projection's correction should be
    small, not a large, destabilizing jump -- a projection step that blows up a good prediction
    would be worse than not having one."""
    op, hr, y = _real_hr_lr_pair(naip_primary_file)
    x_hat = hr.clone()  # the true HR itself -- degrading it IS y, by construction

    corrected = project_to_data_consistency(x_hat, y, op)
    correction_magnitude = (corrected - x_hat).abs().mean().item()
    signal_magnitude = x_hat.abs().mean().item()

    assert correction_magnitude < 0.1 * signal_magnitude, (
        f"expected a small correction for an already-consistent prediction, got "
        f"correction={correction_magnitude:.5f} vs signal={signal_magnitude:.5f}"
    )
