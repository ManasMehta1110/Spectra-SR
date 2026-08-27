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

**Core pipeline built and training end to end** on real cross-sensor data. Stages 0 (degradation
operator), 1 (preprocessing), 3 (HAT-inspired core), 5 (data-consistency projection), 6-slim
(heteroscedastic uncertainty head) and 7 (guardrails) exist with test coverage; 78 tests passing.

Training data is the [SEN2NAIP cross-sensor
set](https://huggingface.co/datasets/isp-uv-es/SEN2NAIP) -- 2,851 same-day-acquired real
Sentinel-2 L2A / NAIP pairs, not synthetic degradation.

**Not yet clearing its own bar.** No run has beaten bicubic interpolation on held-out data. Nine
runs were invalidated by a dead residual branch (the network was returning its bicubic skip
connection unchanged, so every metric was measuring bicubic against bicubic); that and four other
bugs are fixed and documented with the measurements that found them in
[`docs/findings.md`](docs/findings.md). Post-fix, the architecture beats bicubic by **+8.21 dB**
on a memorisation diagnostic, confirming it can learn; generalisation is the current open problem.

Two components remain deliberately unfinished and should not be presented otherwise: Stage 5's
projection is implemented and unit-tested but **not yet wired into the training/inference path**,
and `guardrails.out_of_distribution_check` raises `NotImplementedError` pending real trained-model
feature statistics.

See [`docs/plan.md`](docs/plan.md) for the execution plan and scope gating, and
[`docs/findings.md`](docs/findings.md) for the experimental record.

## Setup

```bash
pip install -e ".[dev]"
pytest -v
```

For data acquisition against the Copernicus Data Space Ecosystem:

```bash
pip install -e ".[data]"
```

## Checkpoint backup

Training runs on Colab Pro (plan Section 2); checkpoints are pushed/pulled via HF Hub rather
than trusted to survive a session disconnect:

```bash
export SPECTRA_SR_HF_REPO=<your-username>/spectra-sr-checkpoints
python scripts/hf_checkpoint.py push <local_checkpoint_dir>
python scripts/hf_checkpoint.py pull <local_checkpoint_dir>
```

## Layout

```
src/spectra_sr/    core package (config, degradation, preprocessing, model, losses,
                    projection, uncertainty, guardrails, dataset, stats)
tests/              unit tests
scripts/            data acquisition, training entry points, checkpoint backup
notebooks/          exploratory / experiment notebooks
analysis/           result aggregation, figures
paper/              writeup, once there's something to write up
docs/plan.md        the full execution plan
docs/findings.md    measured results, bugs found, negative results, experimental record
data/                (gitignored) raw/ and processed/ imagery -- see docs/plan.md Section 4
```
