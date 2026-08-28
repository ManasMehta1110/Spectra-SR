# SPECTRA-SR

**Physically-constrained Sentinel-2 super-resolution with calibrated per-pixel uncertainty.**
Built for SIH 2026, problem statement 26142 (NTRO) -- Deep Learning Based Super Resolution
Mapping from Medium Resolution Satellite Imagery.

> Every output pixel is either measured, or explicitly marked as inferred.

## What this is

Sentinel-2 L2A imagery (10m) super-resolved to 2.5m (4x), built around a data-consistency
(null-space) projection that makes the model structurally incapable of contradicting what
Sentinel-2 actually measured, plus a calibrated per-pixel uncertainty product -- the PS states
the uncertainty requirement twice (Description and Expected Solution), which is read here as the
actual scoring priority, not a nice-to-have.

Direct lineage from [`optical-guided-super-resolution`](../optical-guided-super-resolution)
(a GRSL-track thermal SR ablation study on NASA HLS Landsat-8/Sentinel-2 data) -- same repo
skeleton, `stats.py` reused verbatim, `DualEDSRPlus` kept as the shape-matched baseline this
project's core gets evaluated against, and the same "naive degradation modeling silently
invalidates every downstream number" lesson from that project's own findings (63.6 -> 32.9 dB
when degradation was modeled properly) is why Stage 0 here is a hard acceptance-tested gate
before anything else is trusted.

## Status

**Core pipeline built, trained, and beating its own baseline on real held-out data.** Nine
pretraining runs were invalidated by a dead residual branch (the network returning its own
bicubic skip connection unchanged, so every metric measured bicubic against bicubic) plus four
other bugs -- all fixed and documented with the measurements that found them in
[`docs/findings.md`](docs/findings.md).

The tenth run is the first real result: **+0.6948 dB over bicubic** (21.706 vs 21.011 dB),
**74% win rate** on 200 held-out tiles never trained on, p=2.2e-09, Cohen's d=0.444. Stage 5's
data-consistency projection is wired into inference (`inference.super_resolve`, on by default)
and trades a smaller mean gain for an **82% win rate** and 48.5% lower re-degradation error.
Stage 6's uncertainty head is calibrated (a measured 1.19x over-confidence corrected to
near-perfect via a scalar recalibration factor). 111 tests passing.

**Still open, stated plainly:** the downstream NDVI-classification metric remains negative --
the model's extra spatial detail does not yet translate into better vegetation classification
than naive upsampling, which maps most directly to the PS's stated use case. Guardrail checks
run but their thresholds are placeholders (real sensor NEΔρ values were never sourced -- see
`degradation.py`). `guardrails.out_of_distribution_check` remains `NotImplementedError` by
design, not oversight -- see the docstring for why faking it would be worse than leaving it
honest. Run 10 is a 3.57M-parameter model trained on US imagery (SEN2NAIP); the `FULL` config
(15.5M params, 4.34x the capacity), a larger real dataset, and Indian fine-tuning are the next
steps, not yet done.

**Why there's a next, bigger run:** run 10 was still improving when training stopped -- train
and val loss both still falling at epoch 19, the train/val gap flat at ~2% (not widening, i.e.
not yet overfitting). Doubling epochs from 9 to 19 within that same run roughly tripled the gain
(+0.27 -> +0.69 dB) and pushed win rate from 52% to 74%. That trajectory, not a guess, is why the
next run scales capacity, data, and epochs together -- see `docs/findings.md`'s pre-big-run
section for the full reasoning and the two ablations (uncertainty-head edge features, spectral
loss weights) run to settle open questions before committing the larger compute budget.

**Data**: the [SEN2NAIP cross-sensor set](https://huggingface.co/datasets/isp-uv-es/SEN2NAIP)
(v1, 2,851 real same-day Sentinel-2 L2A / NAIP pairs) is what every reported number above is
measured on. A second, larger release -- [SEN2NAIPv2](https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2)
(8,000 real pairs) -- is also available (`--sen2naip-variant v2`) and is the intended training
set for the next run, but its own held-out numbers are **not directly comparable** to v1's: its
HR was harmonized using the real Sentinel-2 as reference, making it a measurably easier
reconstruction task (avg-pooled-HR/LR correlation 0.995 vs v1's 0.850). Held-out v1 stays the
one honest basis for before/after comparison.

See [`docs/plan.md`](docs/plan.md) for the execution plan and scope gating, and
[`docs/findings.md`](docs/findings.md) for the full experimental record -- every bug, every
measured number, every retracted claim, kept rather than cleaned up after the fact.

## Setup

```bash
pip install -e ".[dev]"
pytest -v
```

For data acquisition against the Copernicus Data Space Ecosystem:

```bash
pip install -e ".[data]"
```

## Training

```bash
python scripts/train_pretrain.py --config colab_realistic --data-source sen2naip \
    --res-scale 0.2 --lr 2e-4 --epochs 20 --out checkpoints/pretrain_runN
```

`--sen2naip-variant {v1,v2}` selects which SEN2NAIP release `--sen2naip-dir` points at --
getting this wrong is silent (v2's HR is uint16 Sentinel-2 reflectance, not v1's uint8 NAIP;
applying v1's scaling to v2 overshoots by ~40x with no error raised), so always set it
explicitly rather than relying on the default. `--resume <checkpoint>` continues an interrupted
run with optimizer/scheduler state intact; `--init-from <checkpoint>` warm-starts a new run
(fresh optimizer/schedule) from another run's weights, for fine-tuning or loss-weight sweeps.

`notebooks/colab_big_run.ipynb` is the ready-to-run Colab notebook for the next, larger training
run -- pulls SEN2NAIPv2 directly from HF Hub, verifies checkpoint backup works before spending
any compute, probes GPU memory to pick a batch size, and trains with the settings validated in
`docs/findings.md`.

## Checkpoint backup

Long runs (Colab Pro) push/pull checkpoints via HF Hub rather than trusting a session to survive
disconnects. `scripts/hf_checkpoint.py check` round-trips a real write before you rely on it --
a read-scoped token authenticates fine and only fails at the first real upload, which on a long
run means finding out hours in.

```bash
export SPECTRA_SR_HF_REPO=<your-username>/spectra-sr-checkpoints
export HF_TOKEN=hf_...                                   # write-scoped
python scripts/hf_checkpoint.py check
python scripts/train_pretrain.py ... --hf-backup-every-n-epochs 1
```

## Evaluation and visualization

```bash
python scripts/visualize_product.py --checkpoint <ckpt> --res-scale 0.2 \
    --recalibration 1.1907 --n-samples 20 --out-dir <dir>
python scripts/rank_gallery_for_ppt.py --checkpoint <ckpt> --res-scale 0.2 \
    --recalibration 1.1907 --n-tiles 65
python scripts/compare_runs.py <run_dir_a> <run_dir_b> --labels a b --control a
```

`visualize_product.py` renders the full delivered product per tile (LR | bicubic | SR | truth |
uncertainty | error, uncertainty and error sharing one colour scale so they're directly
comparable). `rank_gallery_for_ppt.py` scores the same tiles by real PSNR gain over bicubic --
computed without any post-hoc radiometric matching against ground truth, since that pattern
(matching a lone baseline to the true HR) is exactly Bug 4 from `findings.md` and will silently
invert the result if reintroduced. `compare_runs.py` compares runs on weight-independent metrics
only (never `val_total_loss` across arms with different loss weights) and reports a control arm's
own drift so a change can be attributed to what was actually varied.

## Layout

```
src/spectra_sr/    core package: config, degradation, preprocessing, model, losses, projection,
                   uncertainty, calibration, inference, guardrails, dataset, sen2naip_dataset, stats
tests/             unit tests (111 passing)
scripts/           data acquisition, training, resume/backup, evaluation, visualization, ablations
notebooks/         colab_big_run.ipynb -- the next training run, ready to launch
results/           small structured outputs (ablation JSON) from one-off analysis scripts
docs/plan.md       the execution plan and scope gating
docs/findings.md   measured results, bugs found, negative results, the full experimental record
checkpoints/       (gitignored *.pt) trained weights, backed up to HF Hub -- see above
data/              (gitignored) raw/ and processed/ imagery -- see docs/plan.md Section 4
```
