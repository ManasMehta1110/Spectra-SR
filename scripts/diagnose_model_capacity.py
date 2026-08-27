"""Foundational diagnostic: can SpectraHATCore beat bicubic on data it has MEMORIZED?

This should have been the first experiment run, before any of pretrain_run1-7. Every one of
those runs asked "why doesn't the model beat bicubic on held-out data" while silently assuming
the architecture is capable of beating bicubic at all. That assumption was never tested.

Why the question is sharp for this architecture specifically: SpectraHATCore predicts
`bicubic(lr) + residual` (model.py:375-377). Beating bicubic therefore requires nothing more
than a useful nonzero residual. If the model cannot produce one even when overfitting a handful
of tiles -- no generalization required, no held-out set, just memorization -- then the residual
path itself is the bottleneck, and no loss function, data volume, alignment filter, or learning
rate schedule can rescue it.

Interpreting the result:
  * Model clearly exceeds bicubic on the train tiles -> architecture is capable; the failure on
    held-out data is a generalization/objective problem, so loss design (perceptual etc.) is the
    right place to keep working.
  * Model cannot exceed bicubic even here -> the architecture or its use is broken. Stop tuning
    losses and fix that instead.

Deliberately minimal loss (charbonnier only, no SSIM/SAM/index/cycle/gradient) -- the point is to
measure raw fitting capacity, not to evaluate the loss stack. Adding terms here would confound
"can it fit?" with "does the weighted objective point where we think it does?".

Usage:
    python scripts/diagnose_model_capacity.py --n-tiles 4 --steps 600
"""
import argparse
import os

import torch
import torch.nn.functional as F

from spectra_sr.losses import charbonnier_loss
from spectra_sr.model import COLAB_REALISTIC, SMOKE_TEST, SpectraHATCore
from spectra_sr.sen2naip_dataset import SEN2NAIPCrossSensorDataset, _split_train_val_rois
from spectra_sr.utils import DEVICE, logger, set_seed

CONFIGS = {"smoke_test": SMOKE_TEST, "colab_realistic": COLAB_REALISTIC}


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1))
    return float(10 * torch.log10(1.0 / (mse + 1e-12)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIGS), default="colab_realistic")
    p.add_argument("--sen2naip-dir", default="data/raw/sen2naip/cross-sensor/cross-sensor")
    p.add_argument("--qa1-max", type=float, default=0.5)
    p.add_argument("--qa2-max", type=float, default=1.5)
    p.add_argument("--n-tiles", type=int, default=4)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--res-scale", type=float, default=None)
    p.add_argument("--cosine", action="store_true",
                   help="cosine-decay the LR to 0 over --steps")
    args = p.parse_args()

    set_seed(args.seed)
    cfg = CONFIGS[args.config]
    if args.res_scale is not None:
        from dataclasses import replace
        cfg = replace(cfg, res_scale=args.res_scale)
    hr_patch = cfg.train_patch_size * cfg.scale

    train_rois, _ = _split_train_val_rois(args.sen2naip_dir, 0.2, args.qa1_max, args.qa2_max)
    rois = train_rois[:args.n_tiles]
    # crops_per_file=1 and a fixed seed: we want the SAME few patches every step, so this is
    # genuine memorization rather than a slowly-refreshed stream of new crops.
    ds = SEN2NAIPCrossSensorDataset(args.sen2naip_dir, hr_patch_size=hr_patch,
                                     crops_per_file=1, roi_list=rois, seed=args.seed)
    lrs, hrs = [], []
    for i in range(len(ds)):
        lo, hi = ds[i]
        lrs.append(lo)
        hrs.append(hi)
    lr_batch = torch.stack(lrs).to(DEVICE)
    hr_batch = torch.stack(hrs).to(DEVICE)
    logger.info(f"Overfitting {len(rois)} tile(s): {rois}")

    with torch.no_grad():
        bic = F.interpolate(lr_batch, size=hr_batch.shape[-2:], mode="bicubic", align_corners=False)
        bicubic_psnr = psnr(bic, hr_batch)
    logger.info(f"BICUBIC baseline on these exact tiles: {bicubic_psnr:.2f} dB  <-- the bar")

    model = SpectraHATCore(cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
             if args.cosine else None)

    best = -1e9
    for step in range(args.steps):
        opt.zero_grad()
        # one tile per step, cycled -- keeps peak memory at batch=1 like real training
        i = step % lr_batch.shape[0]
        pred = model(lr_batch[i:i + 1])
        loss = charbonnier_loss(pred, hr_batch[i:i + 1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if sched is not None:
            sched.step()

        if (step + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                full = torch.cat([model(lr_batch[j:j + 1]) for j in range(lr_batch.shape[0])])
                cur = psnr(full, hr_batch)
            model.train()
            best = max(best, cur)
            delta = cur - bicubic_psnr
            flag = "BEATS bicubic" if delta > 0 else f"{delta:+.2f} dB vs bicubic"
            logger.info(f"  step {step+1:>4}  charbonnier={loss.item():.5f}  "
                        f"train-tile PSNR={cur:.2f} dB   {flag}")

    print()
    print("=" * 68)
    print(f"bicubic on these tiles : {bicubic_psnr:.2f} dB")
    print(f"best model PSNR        : {best:.2f} dB   ({best - bicubic_psnr:+.2f} dB)")
    if best > bicubic_psnr:
        print("VERDICT: architecture CAN beat bicubic -> failure is generalization/objective,")
        print("         so loss design (perceptual) is the right thing to keep working on.")
    else:
        print("VERDICT: architecture CANNOT beat bicubic even on memorized data.")
        print("         The residual path is the bottleneck. Stop tuning losses; fix this.")
    print("=" * 68)


if __name__ == "__main__":
    main()
