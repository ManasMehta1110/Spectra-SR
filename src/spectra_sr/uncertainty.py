"""Stage 6, Core (slimmed) -- one calibrated uncertainty source. Plan Section 5, phase 5.

Per plan Section 3's scope gating, picked the heteroscedastic NLL head over a small deep
ensemble: an ensemble needs training multiple full SpectraHATCore instances (expensive even at
2-3 models), while a head is a lightweight addition trained once alongside -- or after -- the
already-validated Stage 3 core. The full Stage 6 (both sources fused, K-sample diffusion
sampling, 5-model ensemble) stays stretch-tier.

Deliberately a SEPARATE module from SpectraHATCore, not a second output branch bolted onto it --
keeps Stage 3's tested, stable interface untouched, and lets this head take a genuinely
physically-meaningful input beyond the prediction alone: the re-degradation residual
`A(pred) - lr` from Stage 0's operator. A large residual means the prediction is inconsistent
with what Sentinel-2 actually observed, which is real, directly-relevant evidence for
uncertainty -- not just correlated with it. Ties Stage 6 into the same degradation-operator
machinery as Stage 0/5, rather than treating uncertainty as an unrelated add-on.

Calibration is the actual deliverable, not the raw uncertainty number -- fit/measure the mapping
from predicted uncertainty to real error on held-out data (plan Section 8: reliability diagram +
Expected Calibration Error) before this is presented as a feature anywhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .degradation import DegradationOperator


class UncertaintyHead(nn.Module):
    """Predicts per-pixel log-variance from (prediction, re-degradation residual). A small conv
    stack, not a full second HAT core -- this is meant to be cheap relative to Stage 3.

    `log_var_range` clamps the output -- heteroscedastic NLL heads are known to be unstable to
    train without this: if predicted variance is ever pushed near zero, residual^2/variance
    spikes toward infinity, the resulting gradient explosion wrecks whatever spatial pattern was
    already learned, and training settles into a worse local optimum afterward. Hit exactly this
    failure mode directly (see test_uncertainty.py / conversation): an unclamped head learned a
    correct, strongly spatially-varying variance map through ~150 training steps (log_var std
    growing to ~1.0), then collapsed after a loss spike around step 150-200 and never recovered
    the correct left/right pattern -- clamping is not a defensive nicety, it's required.
    """

    def __init__(self, n_bands: int, hidden: int = 32, log_var_range: tuple = (-8.0, 8.0)):
        super().__init__()
        self.log_var_range = log_var_range
        self.net = nn.Sequential(
            nn.Conv2d(n_bands * 2, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_bands, 3, padding=1),
        )

    def forward(self, pred: torch.Tensor, residual_upsampled: torch.Tensor) -> torch.Tensor:
        """Returns log-variance, same shape as `pred`. Not variance directly -- predicting
        log-variance and exponentiating is the standard heteroscedastic-NLL trick that keeps
        the network from ever having to represent a negative variance."""
        x = torch.cat([pred, residual_upsampled], dim=1)
        raw = self.net(x)
        return torch.clamp(raw, *self.log_var_range)


def heteroscedastic_nll_loss(pred: torch.Tensor, target: torch.Tensor,
                              log_variance: torch.Tensor, max_normalized_residual: float = 100.0
                              ) -> torch.Tensor:
    """0.5 * (residual^2 / variance + log(variance)), mean-reduced. The standard heteroscedastic
    Gaussian negative log-likelihood -- letting the network trade off "predict tightly and be
    penalized hard when wrong" against "predict loosely and pay a log(variance) cost regardless"
    is what makes the resulting variance map meaningful rather than a fixed guess.

    `residual^2 / variance` is clamped to `max_normalized_residual` before averaging -- without
    this, UncertaintyHead's own log-variance clamp (see its docstring) bounds variance but not
    residual^2 itself, and Stage 3's own training dynamics can transiently produce a wild `pred`
    value independent of anything Stage 6 does. Hit this directly: with the log-variance clamp
    alone plus gradient clipping in the training loop, NLL still spiked to ~2e7 at one step and,
    in a follow-up run, to ~1.2e9 with grad_norm hitting ~6.5e10 -- both recovered eventually
    (gradient clipping bounds the *update*, not the forward loss value), but this clamp bounds
    the actual numerical issue at its source instead of only reacting to it after the fact.
    """
    variance = log_variance.exp()
    residual_sq = (pred - target) ** 2
    normalized_residual = (residual_sq / variance).clamp(max=max_normalized_residual)
    return 0.5 * (normalized_residual + log_variance).mean()


def predict_with_uncertainty(model: nn.Module, uncertainty_head: UncertaintyHead,
                              degradation_operator: DegradationOperator, lr: torch.Tensor):
    """End-to-end: run the Stage 3 model, compute its re-degradation residual against the real
    LR observation, upsample that residual, and predict log-variance from (pred, residual).
    Returns (pred, log_variance) -- callers combine this with observational-support features
    (temporal count, cloud distance, registration residual) if/when those exist; Core scope
    ships with just this one source, per the plan's scope gating.
    """
    pred = model(lr)
    residual = lr - degradation_operator.forward(pred)  # (B, C, h, w), LR resolution
    residual_upsampled = F.interpolate(residual, size=pred.shape[-2:], mode="nearest")
    log_variance = uncertainty_head(pred, residual_upsampled)
    return pred, log_variance


def calibrate(predicted_std: torch.Tensor, real_error: torch.Tensor, n_bins: int = 10) -> dict:
    """Bins predictions by predicted uncertainty, checks whether higher-predicted-uncertainty
    bins actually have higher real error -- the real, checkable claim behind "calibrated
    uncertainty," not just that a variance map exists. Returns per-bin (mean predicted std, mean
    real error, count) plus a monotonicity flag, since calibration in the strict statistical
    sense (predicted std matching real error magnitude 1:1) needs more held-out data than a
    handful of patches can support; monotonic ranking is the honest, checkable bar at this
    data scale -- do not claim numerically-precise calibration from this alone.
    """
    predicted_std = predicted_std.flatten().detach()
    real_error = real_error.flatten().detach()
    edges = torch.quantile(predicted_std, torch.linspace(0, 1, n_bins + 1))

    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (predicted_std >= lo) & (predicted_std <= hi if i == n_bins - 1 else predicted_std < hi)
        if mask.sum() == 0:
            continue
        bins.append({
            "mean_predicted_std": float(predicted_std[mask].mean()),
            "mean_real_error": float(real_error[mask].mean()),
            "count": int(mask.sum()),
        })

    mean_errors = [b["mean_real_error"] for b in bins]
    monotonic = all(mean_errors[i] <= mean_errors[i + 1] + 1e-6 for i in range(len(mean_errors) - 1))
    return {"bins": bins, "monotonic": monotonic}


def expected_calibration_error(predicted_confidence: torch.Tensor, correct: torch.Tensor,
                                n_bins: int = 10) -> float:
    """Standard ECE over binned confidence, per plan Section 8. `predicted_confidence` in [0,1],
    `correct` a same-shape boolean/float tensor of whether the prediction was "correct" under
    whatever thresholded definition the caller uses."""
    predicted_confidence = predicted_confidence.flatten().detach()
    correct = correct.flatten().detach().float()
    edges = torch.linspace(0, 1, n_bins + 1)

    ece = 0.0
    n = predicted_confidence.numel()
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (predicted_confidence >= lo) & (predicted_confidence <= hi if i == n_bins - 1
                                                 else predicted_confidence < hi)
        if mask.sum() == 0:
            continue
        bin_confidence = predicted_confidence[mask].mean()
        bin_accuracy = correct[mask].mean()
        ece += (mask.sum().float() / n) * (bin_confidence - bin_accuracy).abs()
    return float(ece)
