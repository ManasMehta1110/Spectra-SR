"""End-to-end inference: the complete product, not just the Stage 3 forward pass.

The problem statement asks for a framework that takes medium-resolution imagery, applies a trained
model, and produces an enhanced product that "clearly accounts for uncertainty and error
components". That is four stages, and until now only the middle one was reachable from a script:

    Stage 3  super-resolve                  SpectraHATCore
    Stage 5  enforce data consistency       projection.project_to_data_consistency
    Stage 6  per-pixel uncertainty          uncertainty.UncertaintyHead
    Stage 7  plausibility guardrails        guardrails.run_guardrails

`super_resolve()` runs all four and returns them together, so the uncertainty map and the
guardrail verdict travel with the image rather than being separately-computed afterthoughts.

On the projection defaults (`tikhonov_lambda=0.03`, `step=0.5`), measured on 200 held-out
SEN2NAIP tiles with the run-10 epoch-19 checkpoint:

    | setting              | mean PSNR | win rate vs bicubic | ||A(x)-y||        |
    |----------------------|-----------|---------------------|-------------------|
    | no projection        | 21.706 dB | 74.0%               | 0.027582          |
    | projection (default) | 21.549 dB | 82.0%               | 0.014194 (-48.5%) |

So the projection costs 0.157 dB of mean accuracy and buys 8 points of win rate plus a halving of
the re-degradation error. It is enabled by default because per-scene reliability and a defensible
consistency claim matter more for this application than a mean, but `apply_projection=False`
restores the higher-mean behaviour and the trade is stated here rather than hidden.

A full-strength projection (`step=1.0`) is deliberately NOT the default: it drives consistency to
-88.8% but costs 0.178 dB, because Stage 0's operator is still a placeholder (sigma=1.0, never
fitted to real Sentinel-2). Projecting fully through a guessed forward model over-corrects. Refit
the operator and re-run scripts/evaluate_projection.py before trusting a larger step.

IMPORTANT SCOPE NOTE: consistency here means consistency with the *modelled* degradation A, not
with physical reality. "Provably consistent with the modelled sensor degradation" is defensible;
"provably consistent with what Sentinel-2 measured" requires Stage 0's acceptance test to pass
against real data first, which has not been done.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch
import torch.nn.functional as F

from .degradation import DegradationOperator
from .guardrails import GuardrailResult, run_guardrails
from .model import SpectraHATCore
from .projection import project_to_data_consistency
from .uncertainty import UncertaintyHead


@dataclass
class SuperResolutionResult:
    """Everything the PS asks a delivered product to carry."""
    image: torch.Tensor              # (B, C, H*s, W*s) super-resolved output
    uncertainty_std: torch.Tensor    # (B, C, H*s, W*s) per-pixel predicted standard deviation
    guardrails: Optional[GuardrailResult]  # plausibility verdict, None if not requested
    projected: bool                  # whether Stage 5 was applied
    consistency_error: float         # mean |A(image) - lr|, the re-degradation residual


def load_for_inference(checkpoint_path: str, configs: dict, device=None):
    """Rebuild a model from a checkpoint, honouring the architecture values the checkpoint
    records rather than the config defaults.

    `res_scale` is a plain attribute, not a learned parameter, so `load_state_dict` silently
    accepts a mismatch and the forward pass is then simply wrong -- this cost a debugging cycle
    when a run-10 checkpoint (trained at 0.2) was reloaded under the config default of 0.1 and
    produced near-black output. Checkpoints written before that fix have no `res_scale` key, so
    pass it explicitly for those.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = configs[ckpt["config"]]
    if ckpt.get("res_scale") is not None and ckpt["res_scale"] != cfg.res_scale:
        cfg = replace(cfg, res_scale=ckpt["res_scale"])

    model = SpectraHATCore(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])

    # Infer the head's architecture from the checkpoint rather than assuming the current default.
    # `use_edge_features` added two input channels to the first conv; without this, every
    # checkpoint written before that change fails to load with a bare shape mismatch. The first
    # conv's in_channels tells us unambiguously which variant was trained.
    use_edge = True
    if "uncertainty_head" in ckpt:
        first_conv = ckpt["uncertainty_head"].get("net.0.weight")
        if first_conv is not None:
            use_edge = first_conv.shape[1] > cfg.n_bands * 2
    head = UncertaintyHead(n_bands=cfg.n_bands, use_edge_features=use_edge).to(device).eval()
    if "uncertainty_head" in ckpt:
        head.load_state_dict(ckpt["uncertainty_head"])

    degradation = DegradationOperator(n_bands=cfg.n_bands, scale=cfg.scale).to(device)
    with torch.no_grad():
        degradation.log_sigma.fill_(torch.log(torch.tensor(1.0)))
    return model, head, degradation, cfg


@torch.no_grad()
def super_resolve(lr: torch.Tensor, model: SpectraHATCore, uncertainty_head: UncertaintyHead,
                  degradation: DegradationOperator, apply_projection: bool = True,
                  tikhonov_lambda: float = 0.03, projection_step: float = 0.5,
                  run_checks: bool = True,
                  uncertainty_recalibration: float = 1.0) -> SuperResolutionResult:
    """Full pipeline on a batch of low-resolution input. See module docstring for the measured
    effect of the projection defaults.

    `uncertainty_recalibration` scales the predicted standard deviation. The head is trained by
    heteroscedastic NLL, which does not guarantee the variance is right in *magnitude* -- measured
    on the run-10 epoch-19 checkpoint it was over-confident by 1.25x, i.e. it understated real
    error, covering only 56% of residuals inside its nominal 68.3% one-sigma interval. Scaling by
    the factor fitted on a disjoint split (1.1907 for that checkpoint) brought ECE from 0.0888 to
    0.0303 and two-sigma coverage to within 0.5 points of theory.

    Left at 1.0 by default because the correct factor is checkpoint-specific: it must be measured
    per model with scripts/calibrate_uncertainty.py and passed in, never assumed. Shipping an
    unrecalibrated head as "calibrated uncertainty" would misstate risk on exactly the inferred
    detail the PS asks to be flagged.
    """
    model.eval()
    uncertainty_head.eval()

    image = model(lr)
    if apply_projection:
        image = project_to_data_consistency(image, lr, degradation,
                                             tikhonov_lambda=tikhonov_lambda,
                                             step=projection_step)

    # Uncertainty is computed from the FINAL image, after any projection -- the delivered
    # uncertainty must describe the delivered pixels, not an intermediate the caller never sees.
    residual = lr - degradation.forward(image)
    residual_up = F.interpolate(residual, size=image.shape[-2:], mode="nearest")
    log_variance = uncertainty_head(image, residual_up)
    uncertainty_std = (0.5 * log_variance).exp() * uncertainty_recalibration

    guardrails = run_guardrails(image, lr, degradation) if run_checks else None
    consistency_error = float(residual.abs().mean())

    return SuperResolutionResult(image=image, uncertainty_std=uncertainty_std,
                                 guardrails=guardrails, projected=apply_projection,
                                 consistency_error=consistency_error)
