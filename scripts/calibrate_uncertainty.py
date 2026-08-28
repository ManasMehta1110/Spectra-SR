"""Stage 6 gate: is the predicted uncertainty actually calibrated?

Plan Section 5's gate for this stage: "Reliability diagram check on held-out data -- predicted
confidence must actually track real error before this is presented as a feature." That check had
never been run; this runs it.

Splits the held-out ROIs in two. The first half fits the scalar recalibration factor, the second
half reports calibration using it. Fitting and reporting on the same data would make any head
look perfectly calibrated, which is precisely the failure this script exists to detect.

Usage:
    python scripts/calibrate_uncertainty.py --checkpoint checkpoints/pretrain_run10/checkpoint_epoch19.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spectra_sr.calibration import apply_recalibration, evaluate_calibration
from spectra_sr.inference import load_for_inference, super_resolve
from spectra_sr.sen2naip_dataset import SEN2NAIPCrossSensorDataset, _split_train_val_rois
from spectra_sr.utils import DEVICE, logger
from train_pretrain import CONFIGS  # noqa: E402


def collect(ds, rois, model, head, degradation, apply_projection, start, end):
    preds, truths, stds = [], [], []
    for i in range(start, end):
        lo, hi = ds[i]
        lo, hi = lo.unsqueeze(0).to(DEVICE), hi.unsqueeze(0).to(DEVICE)
        r = super_resolve(lo, model, head, degradation,
                          apply_projection=apply_projection, run_checks=False)
        # subsample: 570 tiles x 4 x 384^2 would be ~336M values, far more than needed for a
        # stable calibration estimate and enough to exhaust memory.
        preds.append(r.image.flatten()[::97].cpu())
        truths.append(hi.flatten()[::97].cpu())
        stds.append(r.uncertainty_std.flatten()[::97].cpu())
    return torch.cat(preds), torch.cat(truths), torch.cat(stds)


def report(name, rep):
    print(f"\n--- {name} ---")
    print(f"  z_std                 : {rep.z_std:.4f}   (1.0 == calibrated)")
    print(f"  verdict               : {rep.verdict}")
    print(f"  ECE                   : {rep.ece:.4f}")
    print(f"  {'interval':<12}{'nominal':>10}{'empirical':>12}{'gap':>10}")
    for k, nominal, empirical in rep.sigma_coverage:
        print(f"  {f'{k:g} sigma':<12}{nominal:>10.4f}{empirical:>12.4f}{empirical - nominal:>+10.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sen2naip-dir", default="data/raw/sen2naip/cross-sensor/cross-sensor")
    p.add_argument("--n-tiles", type=int, default=120)
    p.add_argument("--res-scale", type=float, default=None)
    p.add_argument("--no-projection", action="store_true")
    p.add_argument("--sen2naip-variant", choices=["v1", "v2"], default="v1",
                   help="Which SEN2NAIP release --sen2naip-dir points at. Selects tile geometry, "
                        "HR scaling and whether radiometric calibration is applied. Wrong values "
                        "fail SILENTLY: v1's /255 HR rule applied to v2's uint16 reflectance "
                        "overshoots by ~40x, so every metric would be computed against nonsense "
                        "without anything raising.")
    args = p.parse_args()

    model, head, degradation, cfg = load_for_inference(args.checkpoint, CONFIGS, DEVICE)
    if args.res_scale is not None and args.res_scale != cfg.res_scale:
        from dataclasses import replace
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        cfg = replace(cfg, res_scale=args.res_scale)
        from spectra_sr.model import SpectraHATCore
        model = SpectraHATCore(cfg).to(DEVICE).eval()
        model.load_state_dict(ckpt["model"])
    logger.info(f"{args.checkpoint}  res_scale={cfg.res_scale}  "
                f"projection={'off' if args.no_projection else 'on'}")

    _, val_rois = _split_train_val_rois(args.sen2naip_dir, 0.2)
    ds = SEN2NAIPCrossSensorDataset(args.sen2naip_dir,
                                     hr_patch_size=cfg.train_patch_size * cfg.scale,
                                     crops_per_file=1, roi_list=val_rois, seed=1,
                                     variant=args.sen2naip_variant)
    n = min(args.n_tiles, len(val_rois))
    half = n // 2
    apply_proj = not args.no_projection

    # Disjoint splits: fit the factor on the first half, report on the second.
    fit = collect(ds, val_rois, model, head, degradation, apply_proj, 0, half)
    test = collect(ds, val_rois, model, head, degradation, apply_proj, half, n)

    print("=" * 66)
    print(f"Stage 6 calibration -- {half} fit tiles, {n - half} disjoint test tiles")
    print("=" * 66)

    raw = evaluate_calibration(*test)
    report("AS TRAINED (no recalibration)", raw)

    factor = evaluate_calibration(*fit).recalibration_factor
    print(f"\n  recalibration factor fitted on held-out fit split: {factor:.4f}")
    rec = evaluate_calibration(test[0], test[1], apply_recalibration(test[2], factor))
    report("AFTER SCALAR RECALIBRATION", rec)

    print("\n" + "=" * 66)
    if rec.ece < raw.ece:
        print(f"Recalibration improves ECE {raw.ece:.4f} -> {rec.ece:.4f}. Ship the factor with")
        print("the model; an uncalibrated variance map should not be presented as uncertainty.")
    else:
        print(f"Recalibration did NOT improve ECE ({raw.ece:.4f} -> {rec.ece:.4f}), so the error")
        print("is not a simple global scale -- investigate before claiming calibration.")
    print("=" * 66)


if __name__ == "__main__":
    main()
