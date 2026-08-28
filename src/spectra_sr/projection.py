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
                                 degradation_operator: DegradationOperator,
                                 tikhonov_lambda: float = 3e-2,
                                 step: float = 1.0) -> torch.Tensor:
    """x_hat: (B, C, H*scale, W*scale) model prediction. y: (B, C, H, W) real LR observation.
    Returns a corrected prediction, same shape as x_hat, nudged toward reproducing y when
    re-degraded -- without this, the model has no structural guarantee it isn't contradicting
    what Sentinel-2 actually measured.

    `step` scales the correction (1.0 = full projection). Exposed because the guarantee this
    provides is only as good as the operator: Stage 0's A is still a placeholder (sigma=1.0, not
    fitted to real Sentinel-2), so a full-strength projection enforces consistency with a *guessed*
    forward model. A partial step trades some of that consistency back for accuracy. Choose both
    this and `tikhonov_lambda` from measured evidence -- see scripts/evaluate_projection.py --
    rather than assuming the defaults are right for a given operator.
    """
    residual = y - degradation_operator.forward(x_hat)  # (B, C, H, W), LR resolution
    correction = degradation_operator.pseudo_inverse(residual, tikhonov_lambda=tikhonov_lambda)
    return x_hat + step * correction
