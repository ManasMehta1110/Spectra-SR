"""Stage 5 -- data-consistency (null-space) projection. Plan Section 5, phase 4.

x <- x + A_dagger(y - A(x)), applied using Stage 0's fitted DegradationOperator. This is what
makes the "structurally incapable of contradicting what Sentinel-2 measured" claim literally
true rather than aspirational -- applied once at the end of the deterministic core's forward
pass for Core scope (the spec's "at every diffusion step" language applies once Stage 4
diffusion, stretch-tier, exists; Core has no sampling steps to project at).

Relies on DegradationOperator.pseudo_inverse, already verified (test_degradation.py) to satisfy
its own real contract -- A(A_dagger(y)) reproduces y better than doing nothing does. This
module's job is just correctly wiring that into the actual projection formula and correctly
handling the LR/HR shape mismatch (A(x_hat) is LR-resolution, x_hat is HR-resolution).
"""
import torch

from .degradation import DegradationOperator


def project_to_data_consistency(x_hat: torch.Tensor, y: torch.Tensor,
                                 degradation_operator: DegradationOperator) -> torch.Tensor:
    """x_hat: (B, C, H*scale, W*scale) model prediction. y: (B, C, H, W) real LR observation.
    Returns a corrected prediction, same shape as x_hat, nudged toward reproducing y when
    re-degraded -- without this, the model has no structural guarantee it isn't contradicting
    what Sentinel-2 actually measured.
    """
    residual = y - degradation_operator.forward(x_hat)  # (B, C, H, W), LR resolution
    correction = degradation_operator.pseudo_inverse(residual)  # -> HR resolution
    return x_hat + correction
