"""Stage 7 -- guardrails. Plan Section 5, phase 6. Post-hoc verification; any tile failing any
check falls back to the deterministic-only output, logged.

Four checks implemented for real: spectral fidelity (SAM), radiometric (per-band RMSE vs.
NEDeltaRho), index preservation (NDVI/NDWI), geometric (phase-correlation shift) -- all directly
testable against real data already in this repo (degradation.py, preprocessing.py).

OOD refusal is deliberately left unimplemented, not faked: it needs real training-manifold
feature statistics from an actually-trained encoder, which doesn't exist yet (Stage 3 hasn't
been trained on real data, only smoke-tested). Building a placeholder OOD check now would mean
either faking statistics or silently always passing/failing -- worse than an honest stub, per
the "authenticity over staging" principle. Revisit once Stage 3 has real trained weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from skimage.registration import phase_cross_correlation

from .degradation import PLACEHOLDER_NEDRHO, DegradationOperator

# Band order matches config.Config.bands = ("B02", "B03", "B04", "B08") = (blue, green, red, nir)
_BLUE, _GREEN, _RED, _NIR = 0, 1, 2, 3


def spectral_angle_mapper(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-pixel angle (radians) between two multi-band spectral vectors, mean-reduced over the
    batch/spatial dims. a, b: (B, C, H, W). 0 for identical spectral *shape* regardless of
    magnitude -- the standard remote-sensing fidelity metric for "did the spectral signature
    stay right," complementary to plain per-band RMSE.

    Clamped to [-1+eps, 1-eps], not [-1, 1] -- arccos's gradient is genuinely unbounded at the
    boundary (d/dx[arccos(x)] = -1/sqrt(1-x^2) -> infinity as x -> +-1), and clamping only the
    *input domain* to a valid range doesn't protect against that: a pixel landing exactly at the
    boundary still backpropagates an effectively-infinite local gradient. Confirmed as a real,
    isolated cause of training NaN within ~20 steps (this term alone, independent of any other
    loss term) before this fix -- not a defensive nicety.
    """
    a_flat = a.flatten(2)  # (B, C, H*W)
    b_flat = b.flatten(2)
    dot = (a_flat * b_flat).sum(dim=1)
    norm_a = a_flat.norm(dim=1)
    norm_b = b_flat.norm(dim=1)
    cos_angle = (dot / (norm_a * norm_b + 1e-8)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.arccos(cos_angle).mean()


def ndvi(x: torch.Tensor) -> torch.Tensor:
    """(NIR - Red) / (NIR + Red). x: (B, 4, H, W) in the config.Config.bands order.

    Output clamped to [-1, 1] -- NDVI is physically bounded there by definition, and clamping
    also fixes a real training instability: as a training loss, the division's gradient is
    unbounded near denominator~=0 (d/dx[a/x] = -a/x^2), which an untrained model's early,
    physically-unconstrained output can easily hit -- observed directly as gradient norms
    reaching ~1e10 and the model going NaN within ~50 training steps before this fix. Clamping
    the *output* range zeroes the gradient once a pixel is pinned at +-1, cutting off the
    explosion at its source rather than only patching symptoms with gradient clipping.
    """
    red, nir = x[:, _RED], x[:, _NIR]
    return ((nir - red) / (nir + red + 1e-4)).clamp(-1.0, 1.0)


def ndwi(x: torch.Tensor) -> torch.Tensor:
    """McFeeters NDWI: (Green - NIR) / (Green + NIR). Same clamping rationale as ndvi()."""
    green, nir = x[:, _GREEN], x[:, _NIR]
    return ((green - nir) / (green + nir + 1e-4)).clamp(-1.0, 1.0)


@dataclass
class GuardrailResult:
    passed: bool
    checks: dict  # {check_name: (passed: bool, value: float)}


def run_guardrails(x_hat: torch.Tensor, y: torch.Tensor, degradation_operator: DegradationOperator,
                    sam_threshold_deg: float = 2.0,
                    radiometric_threshold: Optional[Tuple[float, ...]] = None,
                    index_threshold: float = 0.02,
                    geometric_threshold_px: float = 0.1) -> GuardrailResult:
    """Run every check; a tile fails overall if any individual check fails. Thresholds default
    to the spec's stated values (Stage 7 table) except radiometric, which defaults to Stage 0's
    PLACEHOLDER_NEDRHO -- same "not yet sourced from the real Sentinel-2 handbook" caveat
    applies here as it does in degradation.acceptance_test.
    """
    if radiometric_threshold is None:
        radiometric_threshold = tuple([PLACEHOLDER_NEDRHO] * degradation_operator.n_bands)

    with torch.no_grad():
        redegraded = degradation_operator.forward(x_hat)  # A(x_hat), LR resolution -- compared
                                                            # against y directly (both LR)

        sam_value = float(spectral_angle_mapper(redegraded, y))
        sam_deg = sam_value * 180.0 / 3.141592653589793
        sam_passed = sam_deg < sam_threshold_deg

        per_band_rmse = tuple(
            float(torch.sqrt(torch.nn.functional.mse_loss(redegraded[:, b], y[:, b])))
            for b in range(degradation_operator.n_bands)
        )
        radiometric_passed = all(
            rmse <= thresh for rmse, thresh in zip(per_band_rmse, radiometric_threshold)
        )

        ndvi_delta = float((ndvi(redegraded) - ndvi(y)).abs().mean())
        ndwi_delta = float((ndwi(redegraded) - ndwi(y)).abs().mean())
        index_passed = ndvi_delta < index_threshold and ndwi_delta < index_threshold

        a = redegraded[0, 0].cpu().numpy()
        b = y[0, 0].cpu().numpy()
        shift, _error, _diffphase = phase_cross_correlation(b, a, upsample_factor=20)
        geometric_px = float(max(abs(shift[0]), abs(shift[1])))
        geometric_passed = geometric_px < geometric_threshold_px

    checks = {
        "spectral_sam_degrees": (sam_passed, sam_deg),
        "radiometric_rmse": (radiometric_passed, per_band_rmse),
        "ndvi_delta": (index_passed, (ndvi_delta, ndwi_delta)),
        "geometric_shift_px": (geometric_passed, geometric_px),
    }
    overall = all(passed for passed, _ in checks.values())
    return GuardrailResult(passed=overall, checks=checks)


def out_of_distribution_check(encoder_features, training_manifold_stats, threshold: float) -> bool:
    """NOT IMPLEMENTED -- needs real training-manifold feature statistics collected from Stage 3's
    encoder over its actual training set. Trained weights now exist (run 10 and later), so the
    remaining gap is the collection/fitting step itself, not a lack of anything to fit against.
    A placeholder here would mean faking statistics or a check that silently always passes/fails,
    which is worse than leaving this honestly unimplemented until it's built for real."""
    raise NotImplementedError(
        "Needs real training-manifold statistics from a trained Stage 3 model -- not available "
        "yet. See module docstring."
    )
