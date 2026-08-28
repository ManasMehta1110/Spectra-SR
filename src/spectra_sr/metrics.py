"""Accuracy assessment -- reporting metrics, distinct from the training loss stack. Plan/PS
requirement: the PS text explicitly calls out "accuracy assessment" as its own deliverable,
separate from "model training." SSIM previously only existed as a *loss* (1-SSIM, differentiable,
used to update weights); PSNR/SSIM/RMSE here are plain numeric metrics for reporting results,
mirroring optical_guided_sr.losses.compute_metrics() -- same reasoning as reusing SAM/NDVI: don't
duplicate a tested pattern, port it.

Also implements a real downstream-task utility check: the PS explicitly ties the solution's
value to "improve feature visibility, support better classification, change detection" -- not
just pixel-level sharpness. Built without needing an external labeled dataset (none exists yet):
a simple NDVI-threshold vegetation/non-vegetation classification, comparing how well the SR
output's classification agrees with the true HR classification, against how well naive bicubic
upsampling's classification agrees -- a real, checkable "does SR help a downstream task" ablation
using only data already in this repo (guardrails.ndvi, already tested).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from .guardrails import ndvi, spectral_angle_mapper


@dataclass
class AccuracyMetrics:
    psnr: float
    ssim: float
    rmse: float
    sam_degrees: float


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0
                     ) -> AccuracyMetrics:
    """Plain numeric accuracy assessment -- PSNR/SSIM/RMSE per-image (via skimage, not
    differentiable, not used for training) plus SAM (reused from guardrails, already
    differentiable-safe but used here purely as a metric under no_grad).
    """
    with torch.no_grad():
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        rmse = float(np.sqrt(np.mean((pred_np - target_np) ** 2)))
        psnr = float(sk_psnr(target_np, pred_np, data_range=data_range)) if rmse > 1e-12 else 100.0

        # skimage's multichannel SSIM expects channel-last; average over the batch dim.
        ssim_vals = []
        for b in range(pred_np.shape[0]):
            p = np.moveaxis(pred_np[b], 0, -1)
            t = np.moveaxis(target_np[b], 0, -1)
            ssim_vals.append(sk_ssim(t, p, data_range=data_range, channel_axis=-1))
        ssim_val = float(np.mean(ssim_vals))

        sam_deg = float(spectral_angle_mapper(pred, target)) * 180.0 / 3.141592653589793

    return AccuracyMetrics(psnr=psnr, ssim=ssim_val, rmse=rmse, sam_degrees=sam_deg)


@dataclass
class DownstreamUtilityResult:
    sr_agreement: float           # classification agreement: SR output vs. true HR
    baseline_agreement: float     # classification agreement: naive bicubic upsample vs. true HR
    improvement: float            # sr_agreement - baseline_agreement


def _match_radiometry(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-band linear (mean/std) match of source's radiometry to reference's, over the spatial
    dims. Needed when source and reference come from genuinely different sensors (real
    Sentinel-2 vs. real NAIP, e.g. SEN2NAIPCrossSensorDataset) that were never calibrated to
    share an absolute brightness/contrast scale -- confirmed for real: a naive bicubic upsample
    of real Sentinel-2 reflectance scored 0.256 baseline_agreement (worse than a coin flip)
    against real NAIP HR purely from a ~0.16 vs. ~0.56 mean-brightness mismatch, not from lack of
    spatial detail. The NAIP-synthetic pipeline (LR generated directly from HR) doesn't have this
    problem -- source and reference are already close in scale there, so this is close to a
    no-op in that case, not a special-cased fix."""
    src_mean = source.mean(dim=(-2, -1), keepdim=True)
    src_std = source.std(dim=(-2, -1), keepdim=True) + 1e-6
    ref_mean = reference.mean(dim=(-2, -1), keepdim=True)
    ref_std = reference.std(dim=(-2, -1), keepdim=True) + 1e-6
    return (source - src_mean) / src_std * ref_std + ref_mean


def downstream_classification_agreement(sr_pred: torch.Tensor, lr_input: torch.Tensor,
                                         hr_target: torch.Tensor, ndvi_threshold: float = 0.3,
                                         match_radiometry: str = "neither"
                                         ) -> DownstreamUtilityResult:
    """Real, checkable "does SR help a downstream task" ablation, per the PS's explicit
    "support better classification" language -- without needing an external labeled dataset.

    Builds a binary vegetation/non-vegetation map (NDVI > threshold) from three sources: the SR
    model's output, a naive bicubic upsample of the same LR input (the "did nothing smart"
    baseline), and the true HR target. Reports how well each upsampling approach's
    classification agrees with the true HR classification -- if SR doesn't measurably beat naive
    upsampling here, that's a real, honest finding worth knowing, not something to hide.

    `match_radiometry` controls whether per-sample mean/std matching against hr_target is applied,
    and it must be applied SYMMETRICALLY or not at all:

      "neither" (default) -- realistic deployment. Neither side sees hr_target's statistics.
      "both"              -- isolates spatial structure by removing radiometry from the comparison.
      "baseline_only"     -- LEGACY AND UNFAIR. Retained only to reproduce older numbers.

    Why this parameter exists: an earlier version always matched the baseline and never the
    prediction, which handed bicubic the ground-truth mean and standard deviation -- information
    the model is never given. That inverted the result. Measured on 120 held-out tiles, continuous
    NDVI error:

        baseline_only : SR 0.0729 vs bicubic 0.0367  -> bicubic "wins"
        neither       : SR 0.0729 vs bicubic 0.0842  -> SR wins, 72% of tiles, p=2e-07
        both          : SR 0.0371 vs bicubic 0.0367  -> tied, p=0.15

    The "both" row is the one to keep in mind before over-claiming: our advantage under "neither"
    comes from having learned the cross-sensor radiometric mapping, not from the extra spatial
    detail. Equalise radiometry and the structural gain does not yet show up in NDVI.

    (The original motivation for matching was real -- un-matched bicubic once scored 0.256
    agreement, worse than chance, purely from a brightness offset. That is now handled upstream by
    the dataset's fixed radiometric calibration, so the per-sample fix is no longer needed.)
    """
    if match_radiometry not in {"neither", "both", "baseline_only"}:
        raise ValueError(f"match_radiometry must be neither/both/baseline_only, "
                          f"got {match_radiometry!r}")
    with torch.no_grad():
        target_size = hr_target.shape[-2:]
        bicubic = F.interpolate(lr_input, size=target_size, mode="bicubic", align_corners=False)
        if match_radiometry in {"both", "baseline_only"}:
            bicubic = _match_radiometry(bicubic, hr_target)
        if match_radiometry == "both":
            sr_pred = _match_radiometry(sr_pred, hr_target)

        true_class = ndvi(hr_target) > ndvi_threshold
        sr_class = ndvi(sr_pred) > ndvi_threshold
        baseline_class = ndvi(bicubic) > ndvi_threshold

        sr_agreement = float((sr_class == true_class).float().mean())
        baseline_agreement = float((baseline_class == true_class).float().mean())

    return DownstreamUtilityResult(
        sr_agreement=sr_agreement, baseline_agreement=baseline_agreement,
        improvement=sr_agreement - baseline_agreement,
    )
