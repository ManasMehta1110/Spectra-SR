import math

import pytest
import torch

from spectra_sr.calibration import (
    NOMINAL_SIGMA_COVERAGE, apply_recalibration, evaluate_calibration,
)


def _synthetic(n=200_000, true_sigma=0.1, predicted_sigma=0.1, seed=0):
    """Residuals drawn from a KNOWN Gaussian, with a predicted_std that may deliberately disagree.
    Because the ground-truth relationship is constructed rather than measured, the correct answer
    is known exactly and the estimator can be checked against it."""
    g = torch.Generator().manual_seed(seed)
    truth = torch.zeros(n)
    prediction = torch.randn(n, generator=g) * true_sigma
    return prediction, truth, torch.full((n,), predicted_sigma)


def test_perfectly_calibrated_input_scores_as_calibrated():
    pred, truth, std = _synthetic(true_sigma=0.1, predicted_sigma=0.1)
    r = evaluate_calibration(pred, truth, std)
    assert abs(r.z_std - 1.0) < 0.02
    assert r.ece < 0.01
    assert r.verdict == "well calibrated"


def test_detects_overconfidence():
    """Predicted std half the real error: the model claims more certainty than it has. This is the
    dangerous direction -- it understates risk on exactly the inferred detail the PS warns about --
    so it must be detected and named, not merely scored."""
    pred, truth, std = _synthetic(true_sigma=0.2, predicted_sigma=0.1)
    r = evaluate_calibration(pred, truth, std)
    assert r.z_std > 1.8, f"expected z_std ~2.0, got {r.z_std}"
    assert "OVER-confident" in r.verdict
    assert r.ece > 0.1


def test_detects_underconfidence():
    pred, truth, std = _synthetic(true_sigma=0.05, predicted_sigma=0.1)
    r = evaluate_calibration(pred, truth, std)
    assert r.z_std < 0.6
    assert "UNDER-confident" in r.verdict


def test_sigma_coverage_matches_normal_theory_when_calibrated():
    pred, truth, std = _synthetic(true_sigma=0.1, predicted_sigma=0.1)
    r = evaluate_calibration(pred, truth, std)
    for k, nominal, empirical in r.sigma_coverage:
        assert abs(empirical - nominal) < 0.01, f"{k} sigma: expected {nominal}, got {empirical}"


def test_recalibration_factor_actually_fixes_the_scale():
    """The factor is only useful if applying it works. Verify end to end rather than trusting the
    algebra: fit on one draw, apply, confirm the corrected z_std is ~1."""
    pred, truth, std = _synthetic(true_sigma=0.2, predicted_sigma=0.1, seed=1)
    r = evaluate_calibration(pred, truth, std)
    corrected = apply_recalibration(std, r.recalibration_factor)
    r2 = evaluate_calibration(pred, truth, corrected)
    assert abs(r2.z_std - 1.0) < 0.02, f"recalibration left z_std at {r2.z_std}"
    assert r2.ece < r.ece


def test_reliability_curve_is_monotonic_and_bounded():
    pred, truth, std = _synthetic()
    r = evaluate_calibration(pred, truth, std)
    empirical = [e for _, e in r.curve]
    assert all(0.0 <= e <= 1.0 for e in empirical)
    assert all(empirical[i] <= empirical[i + 1] + 1e-6 for i in range(len(empirical) - 1)), (
        "wider intervals must never contain fewer residuals"
    )


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_calibration(torch.zeros(10), torch.zeros(10), torch.ones(5))


def test_handles_multidimensional_image_tensors():
    g = torch.Generator().manual_seed(3)
    truth = torch.zeros(2, 4, 32, 32)
    pred = torch.randn(2, 4, 32, 32, generator=g) * 0.1
    std = torch.full((2, 4, 32, 32), 0.1)
    r = evaluate_calibration(pred, truth, std)
    assert abs(r.z_std - 1.0) < 0.06  # smaller sample, looser tolerance
