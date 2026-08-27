# Findings

Running record of what has actually been measured, including the failures. Written so a reader
can reconstruct *why* each decision was made, not just what the current settings are.

Companion to [`plan.md`](plan.md) (the intended build order). Where the two disagree, this file
is what really happened.

---

## Summary

Nine pretraining runs produced no model that beat bicubic interpolation on held-out data. The
cause turned out not to be the loss function, the data volume, the learning rate, or the
degradation model -- all of which were investigated in turn -- but a **dead residual branch**:
the network was structurally incapable of super-resolving, and had been returning its own
bicubic skip connection as its output. Every metric collected before that was discovered was
measuring bicubic against bicubic.

The bug and four others are described below with the measurements that identified them.

---

## Bug 1: dead residual branch (the one that mattered)

`SpectraHATCore` predicts `bicubic(lr) + residual`. Beating bicubic therefore requires only a
useful nonzero residual.

**Symptom.** Run 4's final model scored **28.25 dB** on held-out synthetic pairs. Bicubic scored
**28.25 dB**. Identical to two decimal places -- not "close", but the same number.

**Measurement.** Residual magnitude, untrained vs. trained:

| model state | \|residual\| mean | as % of image contrast |
|---|---|---|
| random init | 0.103100 | 56.94% |
| after 20 epochs | 0.000547 | 0.30% |

A 188x collapse. Tracing activations through the network located the failure exactly:

| stage | fresh | trained |
|---|---|---|
| shallow | 0.217 | 0.230 |
| group0 | 0.569 | 2.335 |
| group1 | 0.836 | 5.093 |
| group2 | 1.293 | 9.878 |
| group3 | 1.678 | 16.931 |
| deep+skip | 0.930 | 26.126 |
| upsample0 | 0.362 | 40.257 |
| **upsample1** | 0.216 | **0.000000** |
| output | 0.262 | 0.000547 |

**Cause.** Two compounding faults:

1. `LearnedUpsampler` terminated in `nn.ReLU`. A residual must be able to carry negative values
   (to darken as well as brighten). Worse, a terminal ReLU is a one-way trapdoor -- driven
   negative, it outputs exactly 0, its gradient is exactly 0, and it never recovers.
2. `ResidualHybridAttentionGroup` had no residual scaling, so activations compounded ~2x per
   group and drove that ReLU negative within the first epoch.

**Fix.** Remove the terminal activation (EDSR/SwinIR/HAT upsamplers have none); add EDSR-style
`res_scale` (Lim et al. 2017, introduced for exactly this instability); zero-initialise the
output convolution so training *starts* at bicubic and grows a residual only as it earns loss
reduction.

**Verification.** A capacity diagnostic -- can the model beat bicubic on four tiles it has
memorised? -- went from **-0.16 dB (loses)** to **+8.21 dB (wins)**.

---

## Bug 2: batch-mixing in overlapping cross-attention

`window_partition` returns queries in batch-major order (`b*n_windows + w`); the key/value
windows were assembled with `torch.cat(..., dim=0)`, which is window-major (`w*B + b`). These
coincide only at batch size 1 -- the only size ever used -- so it never surfaced.

At batch >= 2, every sample attended to keys and values belonging to a *different image*. No
crash, no shape error, no NaN.

**Measurement.** Batching three samples vs. running them one at a time (identical results are
required of any per-sample operation): max difference **0.3458**, should be ~0. After fixing to
`torch.stack(...).reshape(...)`: **0.0**.

Latent, but would have activated on the first larger-batch cloud run and presented as "the model
doesn't scale."

---

## Bug 3: cross-sensor radiometric mismatch

Real Sentinel-2 and real NAIP are not calibrated to a shared brightness scale. Per band, means
differ by 2.0-5.3x and standard deviations by 1.5-2.7x. Since the model predicts
`bicubic(lr) + residual`, the residual branch had to synthesise that entire affine transform
before modelling any detail.

**Measurement.** The model's own starting point (`bicubic(lr)` vs HR) on held-out ROIs:

| | PSNR |
|---|---|
| uncalibrated | 9.52 dB |
| calibrated | 22.00 dB |

**Fix.** Fixed per-band affine calibration, constants derived from the *training* split only.
Deliberately not a per-sample match against the paired HR -- that would leak ground truth into
the model's input and be impossible to reproduce at inference.

This is the "cross-sensor radiometric calibration" step the pairing protocol in `plan.md`
Section 4.3 already required, and which had been skipped.

---

## Bug 4: downstream metric measured brightness, not detail

The NDVI-agreement check compared the model's output against a *raw* bicubic upsample of the
input. On cross-sensor data the bicubic baseline is far darker than the HR target, so its NDVI
classification was near-random.

**Measurement.** SR 0.838 vs bicubic 0.256 -- an apparent **+58 point win**. After radiometrically
matching the baseline: SR 0.838 vs bicubic **0.928**, i.e. a **-9 point loss**. The model's own
score never changed; the entire "win" was an artefact of grading a dim image against a bright one.

**Fix.** `metrics._match_radiometry` normalises the baseline before scoring. The model needs no
such treatment -- it is already trained to predict on the target's scale.

---

## Bug 5: learning rate 5x above standard practice

`lr=1e-3` survived nine runs only because the model was frozen; a network that cannot move cannot
diverge. With the residual alive it immediately destabilised: gradient norm mean **96.57** against
a clip threshold of 5.0, cycle-consistency loss up 5000x to 1.588, val PSNR **8.31 dB**.

At `lr=2e-4` (standard for the SwinIR/HAT family): gradient norm **26.86**, cycle loss **0.185**,
val PSNR **21.86 dB**.

---

## Negative results worth keeping

Recorded because they cost real time and are worth not repeating.

- **Edge-aware gradient loss does nothing.** Added at weight 0.3, raised to 2.5 (an 8x increase).
  Across seven epochs the term moved only in the fourth decimal (0.1159-0.1167). At weight 2.5 it
  accounted for **60.9%** of total loss while being completely inert -- meaning most of the
  optimisation signal was going into a term with no effect. Now set to 0.
- **More data alone does not fix blur.** 19 -> 47 NAIP tiles gave a consistent +0.6-0.7 dB PSNR
  gain and *no* movement on the downstream task metric.
- **Alignment filtering helps, but is not sufficient.** Pairs with QA1 <= 0.5 px show 22% better
  edge correlation and +1.42 dB bicubic PSNR than pairs with QA1 > 0.85. Training only on the
  well-aligned subset improved SSIM but still plateaued 0.8 dB below bicubic -- because the
  residual was dead regardless.
- **Cycle-consistency loss contributed 0.0%** of total loss while the residual was dead.

---

## Experimental record

| run | data | key change | epochs | best val PSNR | bicubic bar | verdict |
|---|---|---|---|---|---|---|
| 1 | 19 NAIP tiles, synthetic | baseline | 20 | 27.44 | -- | tied bicubic |
| 2 | 47 NAIP tiles, synthetic | more data | 20 | 28.16 | 28.25 | lost |
| 3 | 47 NAIP, synthetic | + gradient loss 0.3 | 16 | 28.16 | 28.25 | no change |
| 4 | 47 NAIP, synthetic | + LR sched, clip 5.0 | 20 | 28.16 | 28.25 | no change |
| 5 | SEN2NAIP 2851, real | real cross-sensor pairs | 1 | 20.94 | 22.00 | lost |
| 6 | SEN2NAIP 557, filtered | QA1<=0.5, QA2<=1.5 | 7 | 21.79 | 22.61 | lost, plateaued |
| 8 | SEN2NAIP 557 | **architecture fixed**, lr 1e-3 | 1 | 8.31 | 22.61 | diverged |
| 9 | SEN2NAIP 557 | lr 2e-4 | 6 | 21.90 | 22.61 | overfitted (peak ep3) |
| 10 | SEN2NAIP 2851, unfiltered | 5x data | running | 21.10 (ep0) | 21.07 | in progress |

Runs 1-6 are all invalidated by Bug 1: the model was bicubic throughout.

Note that the bicubic bar differs per split -- poorly-aligned pairs are intrinsically harder, so
including them lowers *bicubic's* score too. Raw PSNR is not comparable across runs; only the
gap to that run's own baseline is.

---

## Open question (as of run 10)

Removing both quality filters to get 5x the data produced a specifically **spectral**
degradation between epochs 0 and 1:

| val term | ep0 | ep1 | change |
|---|---|---|---|
| index_loss (NDVI/NDWI) | 0.1537 | 0.3480 | **+126%** |
| sam_loss (spectral angle) | 0.0915 | 0.1735 | **+90%** |
| charbonnier | 0.0797 | 0.1002 | +26% |
| ssim_loss | 0.3513 | 0.3611 | +3% |

Gradient norms *improved* over the same interval (39.0 -> 17.5), so this is not general
instability -- structural terms are stable and only band consistency is collapsing.

Hypothesis: the justification for relaxing the filter applied to **QA1** (spatial alignment)
only, but **QA2** (spectral angle, up to 2.0 deg) was dropped at the same time. Pairs with
inconsistent spectral relationships may admit no learnable mapping. Proposed test: restore
QA2 <= 1.5 while keeping all alignment qualities -> 2,179 tiles, still ~4x run 9.

Not yet acted on; awaiting epochs 2-3 to distinguish a trend from noise.

---

## Method notes

- **The capacity diagnostic should have come first.** `scripts/diagnose_model_capacity.py` asks
  whether the model can beat bicubic on data it has memorised. It runs in ~3 minutes and would
  have caught Bug 1 before any of the multi-hour runs. Overfitting a tiny subset is the cheapest
  possible test of whether an architecture can learn a task at all.
- **Tune on the cheap loop.** The `res_scale` sweep (0.05/0.1/0.2/0.4, peak at 0.2) and the
  step-count/LR-schedule choices were all settled on the 3-minute diagnostic before committing
  to multi-hour runs.
- **Always measure the baseline on the same split.** Reusing a bar computed on a different subset
  produces meaningless comparisons.
- **A metric that suddenly looks excellent deserves more suspicion than one that looks bad.**
  Bug 4 presented as a +58 point win.
