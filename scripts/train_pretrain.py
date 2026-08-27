"""Pretrain-tier training script -- Stage 3 (SpectraHATCore) + Stage 6 (UncertaintyHead) on
real NAIP data, degraded through Stage 0's DegradationOperator. Plan Section 5's Core build
order, stitched together end to end.

Gradient clipping is not optional here -- see spectra_sr.guardrails.spectral_angle_mapper's
docstring: the SAM loss term has a real, confirmed history of exploding training to NaN within
~20-50 steps without it (arccos's unbounded gradient at its domain boundary). Both the arccos
clamp fix AND clipping are kept -- defense in depth, not redundant.

Held-out validation split + per-epoch checkpoints: a real, if modest, safeguard against
overfitting. Files are split train/val by name (deterministic, not random per-run, so the same
files are always held out regardless of seed) -- val files use a FIXED degradation sigma (no
domain randomization), since a validation number that's itself randomized per-sample would be
noisy and hard to compare across epochs. Every epoch's checkpoint is kept (not just the final
one), and every epoch's train/val loss is appended to a persistent log -- so a plateau, a
divergence between train and val, or a good-enough-already point is something you can actually
see and roll back to, not just something asserted about the final checkpoint.

Usage:
    python scripts/train_pretrain.py --config smoke_test --epochs 20 --out checkpoints/pretrain
"""
import argparse
import glob
import json
import os
import time
from typing import Optional

import torch

from spectra_sr.dataset import NAIPPretrainDataset
from spectra_sr.degradation import DegradationOperator
from spectra_sr.losses import SpectraCombinedLoss
from spectra_sr.metrics import compute_metrics, downstream_classification_agreement
from spectra_sr.model import COLAB_REALISTIC, FULL, SMOKE_TEST, SpectraHATCore
from spectra_sr.uncertainty import UncertaintyHead, heteroscedastic_nll_loss
from spectra_sr.utils import DEVICE, logger, set_seed

CONFIGS = {"smoke_test": SMOKE_TEST, "colab_realistic": COLAB_REALISTIC, "full": FULL}


def _split_train_val_files(naip_dir: str, val_fraction: float = 0.2):
    """Deterministic split by sorted filename -- NOT re-randomized per run/seed, so the same
    files are always held out regardless of what seed a given training run uses. Real NAIP
    filenames are already location-coded (e.g. az_/ca_/or_ prefixes, tile IDs), so a plain sort
    naturally spreads the split across the different AOIs rather than clustering val files from
    one region -- worth spot-checking if the file-naming pattern ever changes materially.
    """
    files = sorted(
        f for f in glob.glob(os.path.join(naip_dir, "*.tif"))
        if not os.path.basename(f).startswith("_full_")
    )
    if len(files) < 5:
        raise ValueError(
            f"Only {len(files)} real NAIP files in {naip_dir} -- too few for a meaningful "
            f"train/val split (need at least ~5 to hold out any without gutting training "
            f"diversity)."
        )
    n_val = max(1, round(len(files) * val_fraction))
    val_files = files[::len(files) // n_val][:n_val]  # spread picks across the sorted list
    train_files = [f for f in files if f not in val_files]
    return train_files, val_files


def _run_validation(model, uncertainty_head, criterion, degradation, val_loader) -> dict:
    """Everything needed to plot real curves later, not just the loss the optimizer sees:
    the differentiable loss terms (train/val comparability), PLUS plain accuracy-assessment
    metrics (PSNR/SSIM/RMSE/SAM-degrees, via spectra_sr.metrics.compute_metrics -- the actual
    "accuracy assessment" the PS calls out, distinct from the SSIM *loss*), PLUS the downstream
    NDVI-classification-agreement check against a naive bicubic baseline on the same batch.
    """
    model.eval()
    totals = {"charbonnier": 0.0, "ssim_loss": 0.0, "sam_loss": 0.0, "index_loss": 0.0,
              "cycle_loss": 0.0, "gradient_loss": 0.0, "total_loss": 0.0, "nll": 0.0,
              "psnr": 0.0, "ssim_metric": 0.0, "rmse": 0.0, "sam_degrees": 0.0,
              "downstream_sr_agreement": 0.0, "downstream_baseline_agreement": 0.0,
              "downstream_improvement": 0.0}
    n_batches = 0
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            pred = model(lr)
            terms = criterion(pred, hr, lr)
            residual = lr - degradation.forward(pred)
            residual_up = torch.nn.functional.interpolate(residual, size=pred.shape[-2:],
                                                            mode="nearest")
            log_var = uncertainty_head(pred, residual_up)
            nll = heteroscedastic_nll_loss(pred, hr, log_var)

            acc = compute_metrics(pred, hr)
            down = downstream_classification_agreement(pred, lr, hr)

            totals["charbonnier"] += float(terms["charbonnier"])
            totals["ssim_loss"] += float(terms["ssim"])
            totals["sam_loss"] += float(terms["sam"])
            totals["index_loss"] += float(terms["index"])
            totals["cycle_loss"] += float(terms["cycle"])
            totals["gradient_loss"] += float(terms["gradient"])
            totals["total_loss"] += float(terms["total"])
            totals["nll"] += float(nll)
            totals["psnr"] += acc.psnr
            totals["ssim_metric"] += acc.ssim
            totals["rmse"] += acc.rmse
            totals["sam_degrees"] += acc.sam_degrees
            totals["downstream_sr_agreement"] += down.sr_agreement
            totals["downstream_baseline_agreement"] += down.baseline_agreement
            totals["downstream_improvement"] += down.improvement
            n_batches += 1
    model.train()
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def train(config_name: str, epochs: int, batch_size: int, out_dir: str, naip_dir: str,
          seed: int = 0, val_fraction: float = 0.2, keep_last_n: int = 3,
          lr: float = 1e-3, grad_clip_norm: float = 5.0, use_lr_schedule: bool = True,
          w_gradient: float = 0.3, w_perceptual: float = 0.0,
          data_source: str = "naip_synthetic",
          sen2naip_dir: str = "data/raw/sen2naip/cross-sensor/cross-sensor",
          sen2naip_train_crops: int = 2, sen2naip_val_crops: int = 1,
          qa1_max: Optional[float] = None, qa2_max: Optional[float] = None,
          res_scale: Optional[float] = None) -> dict:
    set_seed(seed)
    cfg = CONFIGS[config_name]
    if res_scale is not None:
        from dataclasses import replace
        cfg = replace(cfg, res_scale=res_scale)
    logger.info(f'config={config_name} res_scale={cfg.res_scale} embed_dim={cfg.embed_dim} '
                f'groups={cfg.n_groups}x{cfg.n_blocks_per_group} patch={cfg.train_patch_size}')
    os.makedirs(out_dir, exist_ok=True)
    epoch_log_path = os.path.join(out_dir, "epoch_log.jsonl")

    # A DegradationOperator is still needed either way -- for "sen2naip" it's never used to
    # *simulate* LR (lr.tif already IS the real Sentinel-2 observation), only for the
    # redegradation_cycle_loss term and the uncertainty head's residual input, same role
    # Stage 0's operator plays in the naip_synthetic path.
    fixed_sigma = torch.log(torch.tensor(1.0))
    degradation = DegradationOperator(n_bands=cfg.n_bands, scale=cfg.scale).to(DEVICE)
    with torch.no_grad():
        degradation.log_sigma.fill_(fixed_sigma)

    hr_patch_size = cfg.train_patch_size * cfg.scale

    if data_source == "sen2naip":
        from spectra_sr.sen2naip_dataset import SEN2NAIPCrossSensorDataset, _split_train_val_rois
        train_rois, val_rois = _split_train_val_rois(sen2naip_dir, val_fraction, qa1_max, qa2_max)
        logger.info(f"SEN2NAIP: {len(train_rois)} train ROIs, {len(val_rois)} held-out val ROIs")
        logger.info(f"Val ROIs (never trained on): {len(val_rois)} ROIs, first 10: {val_rois[:10]}")
        # crops_per_file kept low relative to naip_synthetic's 50/10 -- SEN2NAIP has 2,281 train
        # ROIs vs. NAIP's ~41 files, so even a small crops_per_file gives a much larger, more
        # diverse epoch. A first attempt at crops_per_file=20 produced 45,620 samples/epoch
        # (~22x the naip_synthetic runs) and was still mid-epoch-0 after over an hour -- cut down
        # to keep epoch wall-clock time comparable to the earlier runs.
        train_dataset = SEN2NAIPCrossSensorDataset(sen2naip_dir, hr_patch_size=hr_patch_size,
                                                     crops_per_file=sen2naip_train_crops,
                                                     roi_list=train_rois, seed=seed)
        val_dataset = SEN2NAIPCrossSensorDataset(sen2naip_dir, hr_patch_size=hr_patch_size,
                                                   crops_per_file=sen2naip_val_crops,
                                                   roi_list=val_rois, seed=seed + 1)
    elif data_source == "naip_synthetic":
        train_files, val_files = _split_train_val_files(naip_dir, val_fraction)
        logger.info(f"Train files: {len(train_files)}, held-out val files: {len(val_files)}")
        logger.info(f"Val files (never trained on): {[os.path.basename(f) for f in val_files]}")

        # Two instances, same fixed starting sigma, different devices -- the dataset's
        # __getitem__ synthesizes LR from HR as part of CPU-side data loading (running it on
        # CUDA there caused a device-mismatch crash), while the training loop's loss/uncertainty
        # computation needs its own copy on DEVICE to match `pred`.
        dataset_degradation = DegradationOperator(n_bands=cfg.n_bands, scale=cfg.scale)
        with torch.no_grad():
            dataset_degradation.log_sigma.fill_(fixed_sigma)
        train_dataset = NAIPPretrainDataset(naip_dir, dataset_degradation, hr_patch_size=hr_patch_size,
                                             crops_per_file=50, seed=seed, file_list=train_files)
        # Val: fixed degradation (sigma_range=None), no domain randomization -- a validation
        # number that's itself randomized per-sample would be noisy and hard to compare epoch to
        # epoch. Separate DegradationOperator instance (own fixed sigma) so val's fixed-sigma
        # setting can never be clobbered by the train dataset's per-sample randomization of ITS
        # operator.
        val_degradation = DegradationOperator(n_bands=cfg.n_bands, scale=cfg.scale)
        with torch.no_grad():
            val_degradation.log_sigma.fill_(fixed_sigma)
        val_dataset = NAIPPretrainDataset(naip_dir, val_degradation, hr_patch_size=hr_patch_size,
                                           crops_per_file=10, seed=seed + 1, file_list=val_files,
                                           sigma_range=None)
    else:
        raise ValueError(f"Unknown data_source: {data_source!r}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SpectraHATCore(cfg).to(DEVICE)
    uncertainty_head = UncertaintyHead(n_bands=cfg.n_bands).to(DEVICE)
    criterion = SpectraCombinedLoss(degradation, w_gradient=w_gradient,
                                     w_perceptual=w_perceptual).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(uncertainty_head.parameters()), lr=lr)
    # Real, confirmed finding from pretrain_run1/run2/run3 (all three, despite varying data
    # volume and loss terms): grad_norm_mean sits at 8-18 every epoch against a clip norm of
    # 1.0 -- meaning virtually every step was being forced down to the same capped size
    # regardless of how large a correction was actually warranted, on top of a learning rate
    # that never decreased. That combination is a plausible mechanical explanation for why all
    # three runs plateau by epoch 1-2 and then flatline for the rest of training. Raising the
    # clip norm (default 5.0, still well under the real pathological spikes seen -- up to 3888
    # at epoch 0 of run3) and adding a decay schedule are the next isolated-variable experiment,
    # not a replacement for the arccos domain-clamp fix (guardrails.py), which remains the actual
    # NaN-prevention mechanism; clipping was always meant as backup defense-in-depth, not the
    # primary safeguard.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs) if use_lr_schedule else None

    epoch_records = []
    best_val_total = float("inf")
    best_checkpoint_path = os.path.join(out_dir, "checkpoint_best.pt")
    kept_checkpoints = []  # rolling window of recent (non-best) checkpoint paths

    for epoch in range(epochs):
        epoch_start = time.time()
        # Train side: loss terms + grad_norm only, tracked per-batch and averaged -- NOT
        # accuracy metrics (PSNR/SSIM/downstream), which are deliberately val-only. Running
        # compute_metrics on every single training batch would add real per-step overhead for
        # numbers that would be too noisy at batch_size=1 to read anyway; the val side already
        # gives a stable, comparable accuracy-assessment trend every epoch.
        train_totals = {"charbonnier": 0.0, "ssim_loss": 0.0, "sam_loss": 0.0, "index_loss": 0.0,
                         "cycle_loss": 0.0, "gradient_loss": 0.0, "total_loss": 0.0, "nll": 0.0}
        grad_norms = []
        n_batches = 0

        for lr, hr in train_loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)

            optimizer.zero_grad()
            pred = model(lr)
            loss_terms = criterion(pred, hr, lr)

            residual = lr - degradation.forward(pred)
            residual_up = torch.nn.functional.interpolate(residual, size=pred.shape[-2:],
                                                            mode="nearest")
            log_var = uncertainty_head(pred.detach(), residual_up)  # detach: the uncertainty
            # head should learn to *describe* the core's errors, not reshape the core to make
            # itself easier to predict
            nll = heteroscedastic_nll_loss(pred.detach(), hr, log_var)

            total_loss = loss_terms["total"] + nll
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(uncertainty_head.parameters()),
                max_norm=grad_clip_norm)
            optimizer.step()

            train_totals["charbonnier"] += float(loss_terms["charbonnier"])
            train_totals["ssim_loss"] += float(loss_terms["ssim"])
            train_totals["sam_loss"] += float(loss_terms["sam"])
            train_totals["index_loss"] += float(loss_terms["index"])
            train_totals["cycle_loss"] += float(loss_terms["cycle"])
            train_totals["gradient_loss"] += float(loss_terms["gradient"])
            train_totals["total_loss"] += float(loss_terms["total"])
            train_totals["nll"] += float(nll)
            grad_norms.append(float(grad_norm))
            n_batches += 1

        train_avg = {k: v / max(n_batches, 1) for k, v in train_totals.items()}
        val_avg = _run_validation(model, uncertainty_head, criterion, degradation, val_loader)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": epoch, "epoch_seconds": epoch_time, "lr": current_lr,
            "grad_norm_mean": sum(grad_norms) / max(len(grad_norms), 1),
            "grad_norm_max": max(grad_norms) if grad_norms else 0.0,
            **{f"train_{k}": v for k, v in train_avg.items()},
            **{f"val_{k}": v for k, v in val_avg.items()},
        }
        epoch_records.append(record)
        with open(epoch_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        checkpoint_state = {
            "model": model.state_dict(), "uncertainty_head": uncertainty_head.state_dict(),
            "config": config_name, "epoch": epoch, "val_total_loss": val_avg["total_loss"],
            # Record architecture values, not just the config NAME. res_scale is a plain scalar
            # attribute rather than a learned parameter, so load_state_dict() succeeds silently
            # against a model built with a different res_scale -- every group then contributes the
            # wrong residual magnitude and the output is garbage, with no error raised. Hit this
            # for real: run10 was trained at res_scale=0.2, and reloading it under the config
            # default of 0.1 produced near-black predictions.
            "res_scale": cfg.res_scale,
        }

        # Rolling window, not "keep every epoch forever" -- COLAB_REALISTIC checkpoints are
        # ~14.5MB each; unbounded accumulation over a long run adds up for no real benefit once
        # you're more than a few epochs back. Always keep the best-by-val-loss checkpoint
        # separately, regardless of the window, so a late epoch that's actually worse (early
        # sign of overfitting) never costs you the checkpoint you'd actually want to ship.
        checkpoint_path = os.path.join(out_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(checkpoint_state, checkpoint_path)
        kept_checkpoints.append(checkpoint_path)
        while len(kept_checkpoints) > keep_last_n:
            stale_path = kept_checkpoints.pop(0)
            if os.path.exists(stale_path):
                os.remove(stale_path)

        if val_avg["total_loss"] < best_val_total:
            best_val_total = val_avg["total_loss"]
            torch.save(checkpoint_state, best_checkpoint_path)
            logger.info(f"  new best val_total_loss={best_val_total:.4f} -> {best_checkpoint_path}")

        logger.info(
            f"epoch {epoch}/{epochs - 1} ({epoch_time:.1f}s) "
            f"train_loss={train_avg['total_loss']:.4f} val_loss={val_avg['total_loss']:.4f} "
            f"val_psnr={val_avg['psnr']:.2f} val_ssim={val_avg['ssim_metric']:.4f} "
            f"val_downstream_improvement={val_avg['downstream_improvement']:+.4f} "
            f"-> {checkpoint_path}"
        )

    logger.info(f"Training done. Per-epoch checkpoints + {epoch_log_path} saved to {out_dir}")
    return {"epoch_records": epoch_records}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIGS), default="smoke_test")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--out", type=str, default="checkpoints/pretrain")
    p.add_argument("--naip-dir", type=str, default="data/raw/naip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--keep-last-n", type=int, default=3,
                   help="Rolling window of recent per-epoch checkpoints to keep on disk, plus "
                        "the best-by-validation-loss checkpoint (always kept separately).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip-norm", type=float, default=5.0,
                   help="Raised from an earlier 1.0 -- pretrain_run1/run2/run3 all showed "
                        "grad_norm_mean of 8-18 against a clip norm of 1.0, meaning virtually "
                        "every step was being capped to the same size regardless of the real "
                        "signal. 5.0 lets normal steps through mostly unclipped while still "
                        "catching real spikes (seen up to ~3888).")
    p.add_argument("--no-lr-schedule", action="store_true",
                   help="Disable the cosine LR decay (default: enabled) -- kept as an escape "
                        "hatch for isolating whether the schedule or the clip-norm change is "
                        "doing the work, if it ever needs re-splitting into two experiments.")
    p.add_argument("--w-gradient", type=float, default=0.3,
                   help="Weight on the edge-aware gradient-domain loss term. Raised from the "
                        "original 0.3 default after visualize_predictions.py showed the model "
                        "converging to bicubic-equivalent blurry output -- 0.3 wasn't strong "
                        "enough to escape that basin.")
    p.add_argument("--data-source", choices=["naip_synthetic", "sen2naip"], default="naip_synthetic",
                   help="'naip_synthetic': real NAIP HR + our own DegradationOperator's "
                        "synthetic LR (the original pretrain approach). 'sen2naip': real, "
                        "same-day-acquired Sentinel-2/NAIP pairs from the SEN2NAIP cross-sensor "
                        "dataset -- no synthetic degradation involved, tests whether our "
                        "synthetic degradation model itself was contributing to the "
                        "bicubic-equivalent blur plateau.")
    p.add_argument("--sen2naip-dir", type=str,
                   default="data/raw/sen2naip/cross-sensor/cross-sensor")
    p.add_argument("--sen2naip-train-crops", type=int, default=2,
                   help="Crops per ROI for the sen2naip train set (2,281 real ROIs -- keep low, "
                        "a value tuned for naip_synthetic's ~41 files will blow up epoch time).")
    p.add_argument("--sen2naip-val-crops", type=int, default=1)
    p.add_argument("--qa1-max", type=float, default=None,
                   help="Max SEN2NAIP spatial misalignment (pixels) to keep. Median across the "
                        "full set is 0.680 px; misregistration mathematically rewards blurry "
                        "predictions, so tightening this attacks blur regression at its source. "
                        "Retention: <=0.4 keeps 303 ROIs, <=0.5 keeps 655, <=0.6 keeps 1079.")
    p.add_argument("--w-perceptual", type=float, default=0.0,
                   help="Weight on the VGG feature-space perceptual loss (0 = disabled). This is "
                        "the term that inverts the blur incentive: measured on real NAIP, "
                        "charbonnier scores a blurry candidate BETTER than a sharp one shifted by "
                        "1px (0.032 vs 0.039) while this loss scores it worse (1.584 vs 1.193).")
    p.add_argument("--res-scale", type=float, default=None,
                   help="EDSR-style residual scaling per attention group. Without it, "
                        "activations compounded ~2x per group and killed the residual branch "
                        "outright (measured). Swept on the fast capacity diagnostic.")
    p.add_argument("--qa2-max", type=float, default=None,
                   help="Max SEN2NAIP spectral angle distance (degrees) to keep. Median 1.131.")
    args = p.parse_args()

    train(args.config, args.epochs, args.batch_size, args.out, args.naip_dir, args.seed,
          args.val_fraction, args.keep_last_n, args.lr, args.grad_clip_norm,
          not args.no_lr_schedule, args.w_gradient, args.w_perceptual, args.data_source,
          args.sen2naip_dir,
          args.sen2naip_train_crops, args.sen2naip_val_crops, args.qa1_max, args.qa2_max,
          args.res_scale)
