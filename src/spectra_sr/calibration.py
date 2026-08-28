"""Regression uncertainty calibration -- does the predicted variance actually track real error?

The problem statement asks twice for uncertainty to be "clearly accounted for". A model that
emits a variance map satisfies that only if the variance means something. An uncalibrated head is
worse than no head: a confidently-wrong pixel and a correctly-uncertain one look identical, so a
downstream user cannot tell which inferred details to distrust.

`uncertainty.calibrate()` already checks that higher-predicted-uncertainty bins have higher real
error (monotonic ranking). That was the honest bar when only a handful of patches existed. With
570 held-out ROIs the stricter question is answerable: is the predicted standard deviation right
in *magnitude*, not merely in rank order?

The test used here is the standard one for Gaussian regression uncertainty. If the head is
calibrated, then

    z = (prediction - truth) / predicted_std

is distributed N(0, 1). Three consequences are checked:

  * `z_std` should be 1.0. Below 1 means the model is under-confident (variance too large); above
    1 means over-confident (variance too small). Over-confidence is the dangerous direction --
    it understates risk on exactly the inferred detail the PS warns about.
  * Empirical coverage of each central interval should match its nominal level (68.3 / 95.4 /
    99.7 percent for 1 / 2 / 3 sigma).
  * Expected Calibration Error (ECE) aggregates the gap across a sweep of confidence levels into
    one number, in the same units as probability -- 0.05 means the intervals are wrong by about
    five percentage points on average.

Reported alongside is a scalar `recalibration_factor`: the single multiplier on predicted_std
that would make z_std exactly 1. Applying it is legitimate post-hoc calibration (fit it on a
calibration split, never on the split used to report results), and is the cheapest available fix
when the ranking is good but the scale is off.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import torch

# Central-interval coverage of a standard normal at k sigma, for k = 1, 2, 3.
NOMINAL_SIGMA_COVERAGE = ((1.0, 0.6827), (2.0, 0.9545), (3.0, 0.9973))


@dataclass
class CalibrationReport:
    z_std: float                       # std of (pred - truth)/predicted_std; 1.0 == calibrated
    recalibration_factor: float        # multiply predicted_std by this to reach z_std == 1
    ece: float                         # expected calibration error over the confidence sweep
    sigma_coverage: List[Tuple[float, float, float]] = field(default_factory=list)
    # each entry: (k, nominal_coverage, empirical_coverage)
    curve: List[Tuple[float, float]] = field(default_factory=list)
    # each entry: (nominal_confidence, empirical_coverage) for the reliability diagram

    @property
    def verdict(self) -> str:
        if self.z_std > 1.15:
            return f"OVER-confident by {self.z_std:.2f}x -- understates real error"
        if self.z_std < 0.85:
            return f"UNDER-confident by {1 / self.z_std:.2f}x -- overstates real error"
        return "well calibrated"


def _normal_icdf(p: float) -> float:
    """Inverse standard-normal CDF via the error function, so this module needs no scipy."""
    return math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * p - 1.0)).item()


def evaluate_calibration(prediction: torch.Tensor, truth: torch.Tensor,
                          predicted_std: torch.Tensor, n_levels: int = 19,
                          eps: float = 1e-8) -> CalibrationReport:
    """All three tensors same shape. Returns a CalibrationReport; see module docstring."""
    if not (prediction.shape == truth.shape == predicted_std.shape):
        raise ValueError(f"shape mismatch: pred {tuple(prediction.shape)}, "
                         f"truth {tuple(truth.shape)}, std {tuple(predicted_std.shape)}")

    z = ((prediction - truth) / predicted_std.clamp_min(eps)).flatten().detach().float()
    z_std = float(z.std())
    abs_z = z.abs()

    sigma_coverage = [
        (k, nominal, float((abs_z <= k).float().mean()))
        for k, nominal in NOMINAL_SIGMA_COVERAGE
    ]

    # Reliability curve: sweep nominal central-interval confidence, measure what fraction of
    # residuals actually fall inside. A perfectly calibrated model traces the diagonal.
    curve, gaps = [], []
    for i in range(1, n_levels + 1):
        nominal = i / (n_levels + 1)
        k = _normal_icdf(0.5 + nominal / 2.0)  # half-width of the central interval, in sigmas
        empirical = float((abs_z <= k).float().mean())
        curve.append((nominal, empirical))
        gaps.append(abs(empirical - nominal))

    return CalibrationReport(
        z_std=z_std,
        recalibration_factor=z_std,  # scaling std by z_std makes the new z_std exactly 1
        ece=float(sum(gaps) / len(gaps)),
        sigma_coverage=sigma_coverage,
        curve=curve,
    )


def apply_recalibration(predicted_std: torch.Tensor, factor: float) -> torch.Tensor:
    """Post-hoc scalar recalibration. Fit `factor` on a calibration split (via
    `evaluate_calibration(...).recalibration_factor`) and apply it to a disjoint test split --
    fitting and reporting on the same data would make any head look perfectly calibrated."""
    return predicted_std * factor
