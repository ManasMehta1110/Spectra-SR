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


def _run_validation(model, uncertainty_head, criterion, degradation, val_loader,
                    use_amp: bool = False) -> dict:
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
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                pred = model(lr)
                terms = criterion(pred, hr, lr)
                residual = lr - degradation.forward(pred)
                residual_up = torch.nn.functional.interpolate(residual, size=pred.shape[-2:],
                                                                mode="nearest")
                log_var = uncertainty_head(pred, residual_up)
                nll = heteroscedastic_nll_loss(pred, hr, log_var)
            pred = pred.float()  # metrics (PSNR/SSIM/downstream) computed in fp32 for numerical
                                  # precision regardless of the training/inference dtype

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
          w_sam: float = 0.3, w_index: float = 0.2, w_cycle: float = 0.5,
          data_source: str = "naip_synthetic",
          sen2naip_synthetic_dir: str = "data/raw/sen2naip/synthetic",
          sen2naip_synthetic_era: str = "both",
          sen2naip_dir: str = "data/raw/sen2naip/cross-sensor/cross-sensor",
          sen2naip_train_crops: int = 2, sen2naip_val_crops: int = 1,
          sen2naip_variant: str = "v1",
          qa1_max: Optional[float] = None, qa2_max: Optional[float] = None,
          res_scale: Optional[float] = None,
          val_every_n_steps: int = 0, step_val_tiles: int = 64,
          resume: Optional[str] = None, hf_backup_every_n_epochs: int = 0,
          init_from: Optional[str] = None, use_amp: bool = False) -> dict:
    set_seed(seed)
    cfg = CONFIGS[config_name]
    if res_scale is not None:
        from dataclasses import replace
        cfg = replace(cfg, res_scale=res_scale)
    logger.info(f'config={config_name} res_scale={cfg.res_scale} embed_dim={cfg.embed_dim} '
                f'groups={cfg.n_groups}x{cfg.n_blocks_per_group} patch={cfg.train_patch_size}')
    os.makedirs(out_dir, exist_ok=True)
    epoch_log_path = os.path.join(out_dir, "epoch_log.jsonl")
    # The log is opened in append mode throughout, which is what --resume needs: a resumed run
    # must continue the same record rather than start a second one. For any run that is NOT a
    # resume, appending is wrong -- relaunching into a directory that already holds a log would
    # silently interleave two runs' epochs in one file, and anything reading it back
    # (compare_runs.py, curve plots) would treat the mixture as a single run. Rotate instead of
    # truncating, so an overwritten run's record is preserved rather than destroyed.
    if not resume and os.path.exists(epoch_log_path):
        rotated = os.path.join(out_dir, f"epoch_log.{int(time.time())}.jsonl")
        os.rename(epoch_log_path, rotated)
        logger.warning(f"{epoch_log_path} already existed (previous run in this directory); "
                       f"moved it to {rotated} rather than appending to it.")

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
                                                     roi_list=train_rois, seed=seed,
                                                     variant=sen2naip_variant)
        val_dataset = SEN2NAIPCrossSensorDataset(sen2naip_dir, hr_patch_size=hr_patch_size,
                                                   crops_per_file=sen2naip_val_crops,
                                                   roi_list=val_rois, seed=seed + 1,
                                                   variant=sen2naip_variant, deterministic=True)
        # deterministic=True: this dataset is built once, here, and reused across every epoch
        # below -- without it, epoch N's validation reads different crops of the same ROIs than
        # epoch 0's did, purely from self.rng's shared state advancing, mixing crop-sampling
        # noise into every epoch-to-epoch comparison. See the flag's docstring in
        # sen2naip_dataset.py for the full reasoning. Train dataset deliberately keeps the old
        # (non-deterministic) behavior -- crops varying across epochs is desirable augmentation
        # there, not noise.
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
    elif data_source == "sen2naip_synthetic":
        # The SEN2NAIP synthetic component is HR NAIP only -- no Sentinel-2 in it at all (see
        # synthetic_component_files' docstring). So it runs through exactly the naip_synthetic
        # path: same NAIPPretrainDataset, same operator-simulated LR, same two-instance
        # degradation split. The only difference is where the file list comes from.
        from spectra_sr.sen2naip_dataset import synthetic_component_files
        train_files, val_files = synthetic_component_files(
            sen2naip_synthetic_dir, era=sen2naip_synthetic_era, val_fraction=val_fraction)

        dataset_degradation = DegradationOperator(n_bands=cfg.n_bands, scale=cfg.scale)
        with torch.no_grad():
            dataset_degradation.log_sigma.fill_(fixed_sigma)
        train_dataset = NAIPPretrainDataset(
            sen2naip_synthetic_dir, dataset_degradation, hr_patch_size=hr_patch_size,
            crops_per_file=sen2naip_train_crops, seed=seed, file_list=train_files)
        val_degradation = DegradationOperator(n_bands=cfg.n_bands, scale=cfg.scale)
        with torch.no_grad():
            val_degradation.log_sigma.fill_(fixed_sigma)
        val_dataset = NAIPPretrainDataset(
            sen2naip_synthetic_dir, val_degradation, hr_patch_size=hr_patch_size,
            crops_per_file=sen2naip_val_crops, seed=seed + 1, file_list=val_files,
            sigma_range=None)
    else:
        raise ValueError(f"Unknown data_source: {data_source!r}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SpectraHATCore(cfg).to(DEVICE)
    uncertainty_head = UncertaintyHead(n_bands=cfg.n_bands).to(DEVICE)
    criterion = SpectraCombinedLoss(degradation, w_gradient=w_gradient,
                                     w_perceptual=w_perceptual,
                                     w_sam=w_sam, w_index=w_index, w_cycle=w_cycle).to(DEVICE)
    logger.info(f'loss weights: charbonnier=1.0 ssim=0.2 sam={w_sam} index={w_index} '
                f'cycle={w_cycle} gradient={w_gradient} perceptual={w_perceptual}')
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
    start_epoch = 0

    # Warm start: load WEIGHTS ONLY, and begin a fresh run at epoch 0 with a fresh optimiser and
    # a fresh LR schedule. Deliberately distinct from --resume, which continues an interrupted
    # run and must restore optimiser/schedule state to be correct. The two would be actively
    # wrong if conflated: fine-tuning wants a fresh schedule at the new (usually lower) LR, while
    # resuming wants the original schedule continued from where it stopped.
    #
    # This is the mechanism for the plan's Phase 3 sequence (pretrain on synthetic pairs, then
    # fine-tune on real cross-sensor pairs) and for loss-weight sweeps that start from an
    # already-converged core rather than paying for convergence once per arm.
    if init_from:
        if resume:
            raise ValueError("--init-from and --resume are mutually exclusive: the first starts "
                             "a new run from borrowed weights, the second continues an old run.")
        if not os.path.exists(init_from):
            raise FileNotFoundError(f"--init-from checkpoint not found: {init_from}")
        ckpt = torch.load(init_from, map_location=DEVICE)
        if ckpt.get("res_scale") is not None and ckpt["res_scale"] != cfg.res_scale:
            raise ValueError(f"--init-from checkpoint has res_scale={ckpt['res_scale']} but this "
                             f"run uses {cfg.res_scale}; res_scale is not a learned parameter, so "
                             f"load_state_dict would accept the mismatch silently and every "
                             f"attention group would contribute the wrong residual magnitude.")
        model.load_state_dict(ckpt["model"])
        # The uncertainty head is loaded only if its architecture matches -- `use_edge_features`
        # changes the first conv's in_channels, so a checkpoint from the other variant cannot be
        # loaded and the head simply trains from scratch. It is a small, fast-converging module
        # and is decoupled from the core (see the pred.detach() below), so this costs little.
        try:
            uncertainty_head.load_state_dict(ckpt["uncertainty_head"])
        except (RuntimeError, KeyError) as exc:
            logger.warning(f"uncertainty head not loaded from --init-from ({exc.__class__.__name__}"
                           f"); training it from scratch.")
        logger.info(f"warm start from {init_from} (weights only; fresh optimiser and schedule)")

    scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=use_amp)
    if use_amp:
        logger.info("mixed precision (fp16 autocast + GradScaler) enabled")

    # Resume. Colab sessions disconnect; without this, a run that dies at epoch 15 of 25 restarts
    # from scratch and the compute already spent is simply gone. Restoring the model alone is NOT
    # enough and would be quietly wrong in two ways: Adam's first/second moment estimates would
    # restart at zero (so the first steps after a resume take badly-scaled updates), and the
    # cosine schedule would restart at the initial LR (undoing the decay already served). Both
    # optimiser and scheduler state are therefore saved and restored alongside the weights.
    if resume:
        if not os.path.exists(resume):
            raise FileNotFoundError(f"--resume checkpoint not found: {resume}")
        ckpt = torch.load(resume, map_location=DEVICE)
        if ckpt.get("config") != config_name:
            raise ValueError(f"--resume checkpoint was trained with config={ckpt.get('config')}, "
                             f"but this run specifies config={config_name}")
        if ckpt.get("res_scale") is not None and ckpt["res_scale"] != cfg.res_scale:
            raise ValueError(f"--resume checkpoint has res_scale={ckpt['res_scale']}, but this "
                             f"run uses {cfg.res_scale}. res_scale is not a learned parameter, "
                             f"so load_state_dict would accept the mismatch silently.")
        model.load_state_dict(ckpt["model"])
        uncertainty_head.load_state_dict(ckpt["uncertainty_head"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            logger.warning("--resume checkpoint has no optimizer state (written before resume "
                           "support); Adam moments restart cold.")
        if scheduler is not None and ckpt.get("scheduler") is not None:
            if ckpt.get("epochs") is not None and ckpt["epochs"] != epochs:
                raise ValueError(
                    f"--resume checkpoint was trained with --epochs {ckpt['epochs']}, but this "
                    f"run specifies {epochs}. The cosine schedule's T_max is the epoch count, so "
                    f"restoring its state into a different-length schedule produces an LR curve "
                    f"matching neither run. Re-run with --epochs {ckpt['epochs']}, or add "
                    f"--no-lr-schedule to opt out of the schedule entirely.")
            scheduler.load_state_dict(ckpt["scheduler"])
        if use_amp and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        elif use_amp:
            logger.warning("--resume checkpoint has no scaler state (written before --amp "
                           "support, or saved from a non-amp run); GradScaler starts at its "
                           "default initial scale and will recalibrate over the first few steps.")
        start_epoch = ckpt["epoch"] + 1
        best_val_total = ckpt.get("best_val_total", ckpt.get("val_total_loss", float("inf")))
        global_step_resumed = ckpt.get("global_step", 0)
        logger.info(f"resumed from {resume}: starting at epoch {start_epoch}, "
                    f"best_val_total={best_val_total:.4f}")
    else:
        global_step_resumed = 0

    # Intra-epoch validation. Per-epoch logging gave run 10 only 20 points across 45,620
    # iterations, which is too coarse to see anything forming: the epoch-1 PSNR collapse
    # (21.10 -> 18.70) was only visible 16 minutes after it began. Validating a small fixed
    # subset every N steps gives a dense curve for a small fraction of the cost, and makes a
    # divergence visible while there is still time to kill the run.
    #
    # The subset is a FIXED prefix of the held-out set, so successive points are comparable to
    # each other; the full held-out set is still scored at every epoch boundary, and only those
    # full evaluations drive checkpoint selection. Step records carry "step_eval": true so the
    # two are never mixed when plotting or when picking a best checkpoint.
    step_loader = None
    if val_every_n_steps > 0:
        step_subset = torch.utils.data.Subset(
            val_dataset, range(min(step_val_tiles, len(val_dataset))))
        step_loader = torch.utils.data.DataLoader(step_subset, batch_size=batch_size,
                                                   shuffle=False)
        logger.info(f"intra-epoch validation: every {val_every_n_steps} steps on "
                    f"{len(step_subset)} tiles (full {len(val_dataset)}-tile eval at epoch end)")
    global_step = global_step_resumed

    for epoch in range(start_epoch, epochs):
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
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
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

            # fp16 has a much narrower exponent range than fp32 -- small gradients can silently
            # underflow to zero without a scaler. GradScaler multiplies the loss up before
            # backward (keeping gradients representable in fp16), then unscales them back down
            # before anything touches raw gradient values. Order matters: gradient clipping and
            # the optimizer step must see UNSCALED gradients, so unscale_() runs before clipping,
            # never after -- clipping scaled gradients would clip against the wrong threshold
            # entirely (inflated by the scale factor, typically in the tens of thousands).
            # bf16 (no scaler needed, same exponent range as fp32) is not used here because T4
            # (Turing) has no hardware bf16 support -- only Ampere/later do.
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(uncertainty_head.parameters()),
                max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

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
            global_step += 1

            if step_loader is not None and global_step % val_every_n_steps == 0:
                step_val = _run_validation(model, uncertainty_head, criterion, degradation,
                                            step_loader, use_amp=use_amp)
                step_record = {
                    "step_eval": True, "global_step": global_step, "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_total_loss": float(loss_terms["total"]),
                    **{f"val_{k}": v for k, v in step_val.items()},
                }
                with open(epoch_log_path, "a") as f:
                    f.write(json.dumps(step_record) + "\n")
                logger.info(f"    step {global_step:>6}  val_psnr={step_val['psnr']:.3f}  "
                            f"val_loss={step_val['total_loss']:.4f}  "
                            f"val_ssim={step_val['ssim_metric']:.4f}")

        train_avg = {k: v / max(n_batches, 1) for k, v in train_totals.items()}
        val_avg = _run_validation(model, uncertainty_head, criterion, degradation, val_loader,
                                  use_amp=use_amp)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        record = {
            "step_eval": False, "global_step": global_step,
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
            # Everything below exists so --resume can restart mid-run correctly rather than
            # merely reloading weights. See the resume block above for why each is required.
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if use_amp else None,
            "best_val_total": min(best_val_total, val_avg["total_loss"]),
            "global_step": global_step,
            # Recorded so --resume can detect a changed --epochs. CosineAnnealingLR is
            # parameterised by T_max=epochs; resuming a 60-epoch schedule under --epochs 80
            # restores last_epoch into a schedule with a different period, and the LR then
            # follows a curve that matches neither run -- silently, with no error.
            "epochs": epochs,
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

        # Off-machine backup. Local checkpoints do not survive a Colab session ending, so on a
        # long run the only durable copy is the remote one. Deliberately pushes just the two
        # files worth keeping (this epoch's + the best so far) rather than the whole run dir --
        # the rolling window means most of that directory is unchanged from the previous push.
        if hf_backup_every_n_epochs and (epoch + 1) % hf_backup_every_n_epochs == 0:
            from hf_checkpoint import backup_checkpoint
            if backup_checkpoint([checkpoint_path, best_checkpoint_path, epoch_log_path],
                                 subdir=os.path.basename(out_dir.rstrip("/\\"))):
                logger.info(f"  backed up epoch {epoch} to HF Hub")

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
    p.add_argument("--amp", action="store_true",
                   help="Mixed precision (fp16 autocast + GradScaler). Roughly halves activation "
                        "memory -- measured need: FULL at batch=1 uses 97.4%% of a 16GB T4 "
                        "without this, leaving almost no headroom for validation or "
                        "fragmentation over a long run. Uses fp16 (not bf16): T4 (Turing) has no "
                        "hardware bf16 support, only Ampere/later do.")
    p.add_argument("--w-gradient", type=float, default=0.3,
                   help="Weight on the edge-aware gradient-domain loss term. Raised from the "
                        "original 0.3 default after visualize_predictions.py showed the model "
                        "converging to bicubic-equivalent blurry output -- 0.3 wasn't strong "
                        "enough to escape that basin.")
    p.add_argument("--data-source",
                   choices=["naip_synthetic", "sen2naip", "sen2naip_synthetic"],
                   default="naip_synthetic",
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
    p.add_argument("--val-every-n-steps", type=int, default=0,
                   help="Validate a small held-out subset every N optimizer steps "
                        "(0 = off). Per-epoch logging alone gave run 10 just 20 "
                        "points over 45,620 iterations -- too coarse to see a "
                        "divergence forming. Records carry step_eval=true; the full "
                        "held-out set is still scored at every epoch boundary and is "
                        "the only thing that selects checkpoints.")
    p.add_argument("--step-val-tiles", type=int, default=64,
                   help="Tiles in the intra-epoch subset. Fixed prefix of the "
                        "held-out set so successive points are comparable.")
    p.add_argument("--w-sam", type=float, default=0.3,
                   help="Spectral angle loss weight. Raising this and --w-index targets the "
                        "measured weakness: with radiometry equalised, the model is only TIED "
                        "with bicubic on NDVI (+0.0013), i.e. its extra spatial detail buys no "
                        "spectral accuracy. At 0.3/0.2 against charbonnier 1.0 the objective "
                        "is dominated by pixel fidelity.")
    p.add_argument("--w-index", type=float, default=0.2,
                   help="NDVI/NDWI preservation loss weight. See --w-sam.")
    p.add_argument("--w-cycle", type=float, default=0.5,
                   help="Re-degradation cycle-consistency loss weight: ||A(pred) - lr||^2. "
                        "Previously a buried constructor default with no CLI override. Real "
                        "caveat for --data-source sen2naip: A's sigma is a frozen PLACEHOLDER "
                        "(never fit against real Sentinel-2, never in the optimizer's parameter "
                        "list) for that data source, so this weight is currently trusting an "
                        "unvalidated forward model with real training authority. Worth running "
                        "at 0 as a diagnostic on real data before assuming this term is helping.")
    p.add_argument("--res-scale", type=float, default=None,
                   help="EDSR-style residual scaling per attention group. Without it, "
                        "activations compounded ~2x per group and killed the residual branch "
                        "outright (measured). Swept on the fast capacity diagnostic.")
    p.add_argument("--qa2-max", type=float, default=None,
                   help="Max SEN2NAIP spectral angle distance (degrees) to keep. Median 1.131.")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint to resume from. Restores model, uncertainty head, optimiser "
                        "moments, LR-schedule position, best-so-far val loss and global step -- "
                        "reloading weights alone would silently restart Adam and the cosine "
                        "schedule from zero.")
    p.add_argument("--sen2naip-variant", choices=["v1", "v2"], default="v1",
                   help="Which SEN2NAIP cross-sensor release --sen2naip-dir points at. Selects "
                        "tile geometry (v1 484/121, v2 520/130), HR scaling (v1 is uint8 NAIP "
                        "/255, v2 is uint16 Sentinel-2 reflectance /10000) and whether "
                        "radiometric calibration is applied (v2's HR is already harmonized). "
                        "Getting this wrong is SILENT: the v1 rule applied to v2 overshoots HR "
                        "by ~40x and trains on nonsense without raising anything.")
    p.add_argument("--sen2naip-synthetic-dir", type=str, default="data/raw/sen2naip/synthetic",
                   help="Root of the extracted SEN2NAIP synthetic component (ROI_*/early|late/"
                        "*.tif). HR NAIP only -- the LR side is simulated, so this is "
                        "pretraining volume, not evaluation data.")
    p.add_argument("--sen2naip-synthetic-era", choices=["early", "late", "both"], default="both",
                   help="Which acquisitions to use. 'both' treats the two dates as independent "
                        "HR tiles; the train/val split is always by ROI so the same ground never "
                        "appears on both sides.")
    p.add_argument("--init-from", type=str, default=None,
                   help="Warm-start from a checkpoint's WEIGHTS only, starting a fresh run at "
                        "epoch 0 with a fresh optimiser and LR schedule. Use this to fine-tune "
                        "(e.g. synthetic-pretrained -> real cross-sensor) or to sweep loss "
                        "weights from an already-converged core. Use --resume instead to "
                        "continue an interrupted run.")
    p.add_argument("--hf-backup-every-n-epochs", type=int, default=0,
                   help="Push the current + best checkpoint to HF Hub every N epochs (0 = off). "
                        "Requires SPECTRA_SR_HF_REPO and HF_TOKEN; verify with "
                        "`python scripts/hf_checkpoint.py check` BEFORE starting a long run. "
                        "A failed upload logs a warning and never kills training.")
    args = p.parse_args()

    # Keyword arguments throughout, deliberately. This call was positional and `train`'s
    # signature has grown past 25 parameters; inserting one in the middle silently shifted every
    # later argument by one position, with types compatible enough that nothing raised -- the
    # SEN2NAIP directory would simply have been passed as a different directory parameter.
    # Keywords make the signature order irrelevant.
    train(config_name=args.config, epochs=args.epochs, batch_size=args.batch_size,
          out_dir=args.out, naip_dir=args.naip_dir, seed=args.seed,
          val_fraction=args.val_fraction, keep_last_n=args.keep_last_n,
          lr=args.lr, grad_clip_norm=args.grad_clip_norm,
          use_lr_schedule=not args.no_lr_schedule,
          w_gradient=args.w_gradient, w_perceptual=args.w_perceptual,
          w_sam=args.w_sam, w_index=args.w_index, w_cycle=args.w_cycle,
          data_source=args.data_source,
          sen2naip_synthetic_dir=args.sen2naip_synthetic_dir,
          sen2naip_synthetic_era=args.sen2naip_synthetic_era,
          sen2naip_dir=args.sen2naip_dir,
          sen2naip_train_crops=args.sen2naip_train_crops,
          sen2naip_val_crops=args.sen2naip_val_crops,
          sen2naip_variant=args.sen2naip_variant,
          qa1_max=args.qa1_max, qa2_max=args.qa2_max,
          res_scale=args.res_scale,
          val_every_n_steps=args.val_every_n_steps, step_val_tiles=args.step_val_tiles,
          resume=args.resume, hf_backup_every_n_epochs=args.hf_backup_every_n_epochs,
          init_from=args.init_from, use_amp=args.amp)
