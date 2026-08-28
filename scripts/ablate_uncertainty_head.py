"""Item 3 validation: does adding edge features to UncertaintyHead actually improve it?

Why this can be a cheap experiment rather than a full retrain: in train_pretrain.py the head is
fed `pred.detach()` and its NLL uses the detached prediction as the mean, so no gradient from
the head ever reaches SpectraHATCore. The head is therefore a strictly *post-hoc* module w.r.t.
the SR core, and can be trained against a FROZEN, already-validated core -- isolating the one
variable under test instead of confounding it with a differently-trained core.

Both arms get: the same frozen core, the same seed, the same data order, the same step count,
the same optimiser settings. The only difference is `use_edge_features`.

Reported on a held-out ROI split the core never trained on, further divided into calib/test:
the scalar recalibration factor is FIT on calib and APPLIED to test, because fitting and
reporting ECE on the same tensor makes any head look perfectly calibrated (calibration.py's own
`apply_recalibration` docstring makes this point).

Metrics:
  nll   -- the training objective itself, on held-out data. Lower is better.
  ece   -- expected calibration error after recalibration. Lower is better.
  r     -- per-tile Pearson correlation between predicted sigma and actual |error|. This is the
           number the edge features are meant to move: run 10's head scored only r=0.19 and was
           anti-correlated on 19% of tiles, despite being well calibrated in magnitude. A head
           can be perfectly calibrated on average and still point at the wrong pixels, and `r`
           is what separates those two things.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from spectra_sr.calibration import apply_recalibration, evaluate_calibration
from spectra_sr.inference import load_for_inference
from spectra_sr.sen2naip_dataset import SEN2NAIPCrossSensorDataset, _split_train_val_rois
from spectra_sr.uncertainty import UncertaintyHead, heteroscedastic_nll_loss
from train_pretrain import CONFIGS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson r between two flattened maps. Returns 0.0 for a degenerate (zero-variance) map
    rather than NaN -- a constant uncertainty map is uninformative, which is what r=0 means."""
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom < 1e-12:
        return 0.0
    return float((a @ b) / denom)


def train_head(use_edge: bool, model, degradation, cfg, train_rois, args):
    """Train one head arm against the frozen core. Seeded identically per arm so both see the
    same crops in the same order -- the dataset RNG is seeded, and torch's global seed is reset
    so the two heads start from identical random init."""
    torch.manual_seed(args.seed)
    head = UncertaintyHead(n_bands=cfg.n_bands, use_edge_features=use_edge).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.head_lr)

    dataset = SEN2NAIPCrossSensorDataset(
        args.sen2naip_dir, hr_patch_size=cfg.train_patch_size * cfg.scale,
        crops_per_file=args.train_crops, roi_list=train_rois, seed=args.seed,
        variant=args.sen2naip_variant)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    n_params = sum(p.numel() for p in head.parameters())
    logger.info(f"[edge={use_edge}] head params={n_params:,}  target steps={args.steps}")

    step, running = 0, 0.0
    while step < args.steps:
        for lr_img, hr_img in loader:
            if step >= args.steps:
                break
            lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)
            with torch.no_grad():
                pred = model(lr_img)
                residual = lr_img - degradation.forward(pred)
                residual_up = torch.nn.functional.interpolate(
                    residual, size=pred.shape[-2:], mode="nearest")

            optimizer.zero_grad()
            log_var = head(pred, residual_up)
            nll = heteroscedastic_nll_loss(pred, hr_img, log_var)
            nll.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()

            running += float(nll.detach())
            step += 1
            if step % args.log_every == 0:
                logger.info(f"[edge={use_edge}] step {step:>5}/{args.steps} "
                            f"nll={running / args.log_every:+.4f}")
                running = 0.0
    return head


@torch.no_grad()
def evaluate_head(head, model, degradation, cfg, rois, args, tag):
    """Collect per-tile predictions on `rois`, then score. Kept per-tile (not concatenated into
    one giant tensor) because `r` is defined per tile -- a single global correlation would be
    dominated by between-tile brightness differences rather than within-tile spatial structure,
    which is what the edge features are supposed to fix."""
    head.eval()
    dataset = SEN2NAIPCrossSensorDataset(
        args.sen2naip_dir, hr_patch_size=cfg.train_patch_size * cfg.scale,
        crops_per_file=1, roi_list=rois, seed=args.seed + 1,
        variant=args.sen2naip_variant)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    preds, truths, stds, correlations, nlls = [], [], [], [], []
    for lr_img, hr_img in loader:
        lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)
        pred = model(lr_img)
        residual = lr_img - degradation.forward(pred)
        residual_up = torch.nn.functional.interpolate(
            residual, size=pred.shape[-2:], mode="nearest")
        log_var = head(pred, residual_up)
        nlls.append(float(heteroscedastic_nll_loss(pred, hr_img, log_var)))
        std = (0.5 * log_var).exp()

        for b in range(pred.shape[0]):
            correlations.append(_pearson(std[b], (pred[b] - hr_img[b]).abs()))
        preds.append(pred.cpu())
        truths.append(hr_img.cpu())
        stds.append(std.cpu())

    preds = torch.cat(preds)
    truths = torch.cat(truths)
    stds = torch.cat(stds)

    # Split held-out tiles into calib/test. The recalibration factor is fit on calib only and
    # applied to test, so the reported ECE is an honest out-of-sample number.
    n_calib = len(preds) // 2
    calib = evaluate_calibration(preds[:n_calib], truths[:n_calib], stds[:n_calib])
    test_raw = evaluate_calibration(preds[n_calib:], truths[n_calib:], stds[n_calib:])
    test_recal = evaluate_calibration(
        preds[n_calib:], truths[n_calib:],
        apply_recalibration(stds[n_calib:], calib.recalibration_factor))

    corr = torch.tensor(correlations)
    result = {
        "tag": tag,
        "n_tiles": int(len(preds)),
        "nll": sum(nlls) / len(nlls),
        "z_std_raw": test_raw.z_std,
        "ece_raw": test_raw.ece,
        "recalibration_factor": calib.recalibration_factor,
        "ece_recalibrated": test_recal.ece,
        "sigma2_coverage_recalibrated": test_recal.sigma_coverage[1][2],
        "corr_mean": float(corr.mean()),
        "corr_median": float(corr.median()),
        "corr_frac_positive": float((corr > 0).float().mean()),
        "verdict_raw": test_raw.verdict,
    }
    return result, corr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/pretrain_run10/checkpoint_best.pt")
    p.add_argument("--sen2naip-dir", default="data/raw/sen2naip/cross-sensor/cross-sensor")
    p.add_argument("--out", default="results/uncertainty_ablation.json")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--train-crops", type=int, default=2)
    p.add_argument("--eval-rois", type=int, default=200,
                   help="Held-out ROIs to score on. One crop each.")
    p.add_argument("--head-lr", type=float, default=2e-4)
    p.add_argument("--grad-clip-norm", type=float, default=5.0)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--sen2naip-variant", choices=["v1", "v2"], default="v1",
                   help="Which SEN2NAIP release --sen2naip-dir points at; see "
                        "sen2naip_dataset.DATASET_VARIANTS. Wrong values fail silently.")
    args = p.parse_args()

    model, _, degradation, cfg = load_for_inference(args.checkpoint, CONFIGS, device=DEVICE)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    logger.info(f"frozen core: res_scale={cfg.res_scale} patch={cfg.train_patch_size} "
                f"scale={cfg.scale} n_bands={cfg.n_bands}")

    # Exactly the split train_pretrain.py uses, so the "held-out" ROIs really are ones the
    # frozen core never saw.
    train_rois, val_rois = _split_train_val_rois(args.sen2naip_dir, args.val_fraction)
    eval_rois = val_rois[:args.eval_rois]
    logger.info(f"{len(train_rois)} train ROIs, {len(val_rois)} held-out "
                f"({len(eval_rois)} used for scoring)")

    results, per_tile = [], {}
    for use_edge in (False, True):
        tag = "edge" if use_edge else "baseline"
        head = train_head(use_edge, model, degradation, cfg, train_rois, args)
        result, corr = evaluate_head(head, model, degradation, cfg, eval_rois, args, tag=tag)
        results.append(result)
        per_tile[tag] = corr.tolist()
        logger.info(json.dumps(result, indent=2))

    # Paired significance on the per-tile correlations. Both arms score the SAME tiles in the
    # same order, so the comparison is paired -- and a paired test is the only honest one here,
    # because between-tile variance in how predictable a scene's error is dwarfs the difference
    # between the two heads. An unpaired test on these numbers would be badly underpowered and
    # would report "no effect" regardless of the truth.
    from scipy import stats as scipy_stats

    base_corr = torch.tensor(per_tile["baseline"], dtype=torch.float64)
    edge_corr = torch.tensor(per_tile["edge"], dtype=torch.float64)
    diff = edge_corr - base_corr
    t_stat, t_p = scipy_stats.ttest_rel(edge_corr.numpy(), base_corr.numpy())
    try:
        w_stat, w_p = scipy_stats.wilcoxon(edge_corr.numpy(), base_corr.numpy())
    except ValueError:  # raised when every difference is exactly zero
        w_stat, w_p = float("nan"), 1.0
    # Cohen's d for paired samples: mean difference over the SD of the differences.
    cohens_d = float(diff.mean() / diff.std()) if float(diff.std()) > 0 else 0.0
    sem = float(diff.std()) / (len(diff) ** 0.5)
    significance = {
        "mean_delta_corr": float(diff.mean()),
        "ci95_low": float(diff.mean()) - 1.96 * sem,
        "ci95_high": float(diff.mean()) + 1.96 * sem,
        "paired_t_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "cohens_d": cohens_d,
        "win_rate": float((diff > 0).double().mean()),
        "n": int(len(diff)),
    }
    logger.info("significance: " + json.dumps(significance, indent=2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results,
                   "significance": significance, "per_tile_corr": per_tile}, f, indent=2)

    base, edge = results
    print("\n" + "=" * 78)
    print(f"{'metric':<34}{'baseline':>14}{'edge':>14}{'delta':>14}")
    print("-" * 78)
    for key, better_is_lower in [("nll", True), ("ece_recalibrated", True),
                                 ("corr_mean", False), ("corr_median", False),
                                 ("corr_frac_positive", False)]:
        delta = edge[key] - base[key]
        verdict = "better" if ((delta < 0) == better_is_lower and abs(delta) > 1e-9) else "worse"
        print(f"{key:<34}{base[key]:>14.4f}{edge[key]:>14.4f}{delta:>+13.4f}  {verdict}")
    print("=" * 78)
    s = significance
    print(f"per-tile correlation, paired over n={s['n']} tiles:")
    print(f"  mean delta {s['mean_delta_corr']:+.4f}  "
          f"95% CI [{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}]")
    print(f"  paired t p={s['paired_t_p']:.3e}   Wilcoxon p={s['wilcoxon_p']:.3e}   "
          f"Cohen's d={s['cohens_d']:.3f}   win rate {s['win_rate']:.1%}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
