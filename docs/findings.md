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

**Run 10, after the fixes, is the first model that consistently beats bicubic**: +0.6948 dB
(21.706 vs 21.011), 74% win rate on 200 held-out tiles, p=2.2e-09. Stage 5's data-consistency
projection and Stage 6's uncertainty calibration are both wired in and measured. Two further
ablations -- uncertainty-head edge features (kept, small real effect) and raised spectral loss
weights (not kept, no benefit where it matters) -- were run to settle open questions before
committing to a larger, more expensive training run. See "Pre-big-run checks" and the two
sections after it, near the end of this file, for that work; the sections between here and
there are the original nine-run debugging history, kept in full because the bugs and negative
results in it are exactly what stops the next round of experiments from re-discovering the same
dead ends.

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

- ~~**Edge-aware gradient loss does nothing.**~~ **RETRACTED -- this conclusion does not survive
  Bug 1.** It was drawn from runs 3-4, which are inside the invalidated range. If the residual is
  dead then `output == bicubic` exactly, so `gradient_loss(bicubic, hr)` has *no parameter
  dependence at all* and its gradient w.r.t. every weight is identically zero. Inertness was
  guaranteed a priori, not measured. The observed variation across 20 epochs of run 4 was 1.16%
  (0.166146-0.168076), i.e. sampling noise from random crops, not learning.

  The tell was there and went unread: a term at **60.9% of total loss** moving only in the fourth
  decimal is not a weak signal, it is a *disconnected* one. That should have prompted the question
  "is this term attached to the graph?" rather than "is this term useless?".

  Consequence: `w_gradient=0` currently removes the stated mitigation for an open risk in
  `plan.md` ("pixel-loss blurs fine linear features") on evidence that does not hold. **Pending
  re-measurement on the fixed architecture.**
- ~~**More data alone does not fix blur.**~~ **RETRACTED for the same reason** -- runs 1-2, also
  inside the invalidated range. The measured +0.6-0.7 dB PSNR gain from 19 -> 47 NAIP tiles is
  real, but "no movement on the downstream metric" is uninformative: the downstream metric was
  comparing bicubic against bicubic. Pending re-measurement.
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

**Resolved at epoch 2: it was a transient, not a trend.** Every term recovered to its epoch-0
level without any intervention:

| val term | ep0 | ep1 | ep2 | ep0 -> ep2 |
|---|---|---|---|---|
| index_loss | 0.1537 | 0.3480 | 0.1596 | +3.9% |
| sam_loss | 0.0915 | 0.1735 | 0.0919 | +0.5% |
| charbonnier | 0.0797 | 0.1002 | 0.0804 | +0.8% |
| ssim_loss | 0.3513 | 0.3611 | 0.3434 | -2.2% |

A structural objective mismatch does not self-correct while training continues. The QA2 hypothesis
is therefore **not supported** and the filter was not restored.

Two external code reviews (obtained independently) both diagnosed this spike as structural -- one
attributing it to a physically incompatible cycle-consistency constraint, the other to
unlearnable per-tile radiometric offsets. Both were written before epoch 2 existed. Measured cycle
contribution at epoch 2 is **0.1% of total loss** (raw 0.0006, falling), so that suspect is
negligible regardless. Recorded as a caution: a two-point trend is not a trend, and reviewing one
invites over-diagnosis.

---

## Run 10, epoch 9: first statistically real gain over bicubic

Measured on 150 held-out ROIs the model never trained on.

| test | result |
|---|---|
| mean paired PSNR difference | **+0.2672 dB** |
| 95% CI | +0.1107 to +0.4237 (excludes zero) |
| paired t-test | p = 0.00104 |
| Wilcoxon signed-rank | p = 0.00496 |
| Cohen's d | **0.273 (small)** |
| tiles where SR beats bicubic | **78/150 (52%)** |

Detail recovery is genuine rather than added noise -- gradient-field correlation against the true
HR is **0.4016 (SR) vs 0.3652 (bicubic)**, i.e. the model's edges land where real edges are. A
model emitting random high-frequency content would score *worse* on this, not better.

**Reporting correction.** This was first written up as "67% more edge energy than bicubic"
(0.0735 vs 0.0440 mean gradient magnitude). That figure is arithmetically right and analytically
wrong: edge *energy* measures how much high-frequency content exists, not whether it is correctly
placed. The defensible number is edge *correlation*, **+10.0% relative**. Quote that one.

**Honest reading.** Statistically real, practically marginal. A 52% win rate means bicubic still
wins on nearly half of held-out tiles; the positive mean comes from winning by more on the tiles
it wins than it loses by elsewhere. What this result establishes is that the model performs
genuine super-resolution rather than returning its bicubic skip connection -- the failure that
invalidated runs 1-6. It does not establish a model whose output one would confidently prefer.

---

## Run 10 final (epoch 19): first model that consistently beats bicubic

200 held-out ROIs, never trained on. `res_scale=0.2`, lr 2e-4 cosine, all 2,851 pairs.

| | epoch 9 | **epoch 19** |
|---|---|---|
| bicubic PSNR | 21.07 | 21.011 |
| SPECTRA-SR PSNR | 21.33 | **21.706** |
| mean gain | +0.2672 dB | **+0.6948 dB** |
| 95% CI | +0.11 to +0.42 | **+0.478 to +0.912** |
| paired t-test | p = 1.0e-3 | **p = 2.2e-09** |
| Wilcoxon | p = 5.0e-3 | **p = 1.5e-12** |
| Cohen's d | 0.273 (small) | **0.444 (medium)** |
| **win rate** | 52% | **74% (148/200)** |
| edge correlation vs HR | +10.0% | **+11.7%** |

The win rate is the meaningful number. At epoch 9 the mean gain was significant but the model beat
bicubic on barely half of individual tiles -- a better average, not a better model. At epoch 19 it
wins on roughly three of four.

Per-band radiometric accuracy also beats bicubic (mean absolute band error 0.0574 vs 0.0627,
better on all four bands), so the gain is not bought by distorting colour.

**What this does not establish.** The downstream NDVI-classification metric remains negative
(-0.0675): band *ratios* amplify the ~6% residual per-band error, so spectral fidelity is still
the weak axis, and that metric maps most directly to the PS requirement. This is also a 3.57M-param
model trained on US imagery -- the `FULL` config, real batch sizes, and Indian fine-tuning are all
still ahead.

---

## Stage 5 (projection) and Stage 6 (calibration): both gates now measured

Both existed as tested modules that no pipeline ever called. `inference.super_resolve()` now runs
Stages 3/5/6/7 together and returns the image, its uncertainty map, and the guardrail verdict as
one product.

### Stage 5 -- the projection is a trade, not free

200 held-out ROIs, run-10 epoch-19 checkpoint, `tikhonov_lambda=0.03`, `step=0.5`:

| | model raw | + projection |
|---|---|---|
| mean PSNR | 21.706 dB | 21.549 dB (**-0.157**) |
| win rate vs bicubic | 74.0% | **82.0%** |
| consistency \|\|A(x)-y\|\| | 0.027582 | **0.014194 (-48.5%)** |

It lowers the mean but raises the win rate by 8 points -- variance reduction, rescuing weak tiles
more than it costs strong ones. Enabled by default: per-scene reliability and a defensible
consistency claim matter more here than a mean.

A **partial step is essential**. Full-strength projection (`step=1.0`) drives consistency to -88.8%
but costs 0.178 dB, because Stage 0's operator is a placeholder (sigma=1.0, never fitted). A half
step takes the part of the correction the real measurement actually informs and stops before the
operator's own error dominates.

**Caution against over-claiming**: an earlier 60-tile measurement showed the projection *improving*
PSNR by +0.087 dB. At n=200 that reversed to -0.157 dB (p=0.0068). Sixty tiles was not enough.

**Scope of the claim.** \|\|A(x)-y\|\| is measured with the same placeholder operator the projection
uses, so it is partly circular. Defensible: *"provably consistent with the modelled sensor
degradation."* Not yet defensible: *"consistent with what Sentinel-2 measured"* -- that needs Stage
0's acceptance test against real data.

### Stage 6 -- the head was over-confident; a scalar factor fixes it

Plan Section 5's gate ("predicted confidence must actually track real error before this is
presented as a feature") had never been run. 60 fit tiles, 60 **disjoint** test tiles:

| | as trained | after recalibration |
|---|---|---|
| z_std (1.0 = calibrated) | **1.2513** | **1.0509** |
| verdict | over-confident 1.25x | well calibrated |
| ECE | 0.0888 | **0.0303** |
| 1 sigma coverage (want 0.6827) | 0.5600 | 0.6445 |
| 2 sigma coverage (want 0.9545) | 0.8899 | **0.9496** |
| 3 sigma coverage (want 0.9973) | 0.9890 | **0.9976** |

The head **understated real error by 25%** -- the dangerous direction, since it is exactly the
inferred detail the PS asks to be flagged. A scalar factor of **1.1907**, fitted on a disjoint
split, brings 2-sigma and 3-sigma coverage within 0.5 and 0.03 percentage points of theory.

The factor is checkpoint-specific and defaults to 1.0 in `super_resolve()`: it must be measured per
model with `scripts/calibrate_uncertainty.py`, never assumed.

### Stage 7 -- runs, but thresholds are placeholders

Guardrails now execute on real output and return per-check verdicts. On a sample held-out tile:
spectral SAM passed (1.86 deg), geometric shift passed (0.05 px), radiometric RMSE and NDVI/NDWI
delta failed. **That FAIL is not yet meaningful**: `PLACEHOLDER_NEDRHO = 0.005` has never been
sourced from ESA's Sentinel-2 handbook. The mechanism is verified; the thresholds are not.

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
- **Distinguish "is the signal present" from "is it in the right place".** Edge energy answers the
  first, edge correlation answers the second, and only the second is evidence of super-resolution.
  Reporting the first alone overstated the result by roughly 6x.
- **Report the win rate alongside the mean.** +0.27 dB with p=0.001 sounds decisive; 52% of tiles
  says otherwise. Both are true and only the pair is honest.
- **Never load a checkpoint into a config it was not trained with.** `res_scale` is a plain
  attribute, not a parameter, so `load_state_dict` accepts the mismatch silently and the forward
  pass is simply wrong. Checkpoints now record their own architecture values.
- **Do not duplicate the data-loading path.** `visualize_predictions.py` re-read the GeoTIFFs
  itself and missed the radiometric calibration added later to the dataset, feeding the model
  input ~3x too dark and producing near-black output that looked like catastrophic model failure.
  Evaluation code must consume the same dataset class training does.

## Pre-big-run checks: the model is underfit, not data-starved

Measured to decide whether the next investment should be more data or more capacity, before
committing a long Colab run to either.

Run 10's train/val gap across all 20 epochs:

| epoch | train | val | gap | gap % |
|---|---|---|---|---|
| 8 | 0.2003 | 0.2037 | +0.0034 | +1.7% |
| 12 | 0.1947 | 0.1984 | +0.0037 | +1.9% |
| 16 | 0.1903 | 0.1948 | +0.0045 | +2.4% |
| 19 | 0.1875 | 0.1919 | +0.0044 | +2.4% |

The gap is **stable at ~2% from epoch 8 onward — it is not widening** — and both train and val
loss were still falling monotonically at epoch 19, with val PSNR still climbing (21.65 -> 21.79
over the last three epochs). That is the signature of an underfit model, not an overfit one.

Consequence for the plan: at COLAB_REALISTIC's 3.57M parameters, more data would not have been
the binding constraint; capacity and training time are. **This does not generalize to FULL**,
which is roughly 5x the parameters on the same 2,851 cross-sensor tiles -- a regime where
overfitting becomes plausible and more data starts to matter. Pretraining data was therefore
still acquired, but as insurance for the capacity increase rather than as a fix for run 10.

### What the SEN2NAIP "synthetic" component actually is

Worth recording because the name is misleading and the size (177 GB total, 18 shards) makes a
wrong assumption expensive. Its metadata carries `s2_id: null`, `QA1: null`, `QA2: null`, and
each ROI holds `early/<naip>.tif` and `late/<naip>.tif` at 1100x1100, 4-band uint8, 2.5m.

**There is no Sentinel-2 imagery in it at all.** It is HR NAIP only; the LR side has to be
simulated by our own degradation operator -- which is exactly what the existing
`naip_synthetic` path already does. So it needs no new Dataset class, only a file-list helper
(`synthetic_component_files`), and it feeds `NAIPPretrainDataset` unchanged.

Two consequences:

1. A model trained on it learns to invert a degradation *we chose*, which is an easier and
   materially different task than the real cross-sensor one. Its role is pretraining volume
   (plan Section 5, phase 3); the 2,851 real cross-sensor pairs remain the only honest basis for
   any reported accuracy number.
2. Per-ROI it carries 5.4x the pixel area of a cross-sensor tile (1100^2 vs 484^2), so a single
   10 GB shard holds roughly 1.8x more unique HR pixel area than the entire cross-sensor set.
   Volume is reachable without pulling all 177 GB.

The train/val split for it is **by ROI, never by file**: the two eras cover the same ground a
decade apart, so a per-file split would put the 2011 image of a field in train and the 2021
image of that same field in val, inflating the val score with near-duplicate leakage. Covered
by `test_synthetic_split_is_by_roi_so_paired_eras_never_straddle_it`.

### Resume was missing, and weights-only resume would have been silently wrong

`train_pretrain.py` had no resume path at all, so a Colab disconnect at epoch 15 of a 60-epoch
run would have discarded all of it. Added `--resume`, but the checkpoint had to change too: it
stored only model and uncertainty-head weights. Restoring those alone leaves

* **Adam's moment estimates at zero**, so the first steps after a resume take badly-scaled
  updates into an already-converged model, and
* **the cosine schedule back at the initial LR**, undoing all decay already served.

Both now round-trip, along with `best_val_total` and `global_step`. Verified by training two
epochs, resuming, and confirming continuity with no loss discontinuity at the seam (a cold Adam
restart shows a visible spike; this did not).

A second, subtler footgun found while testing: `CosineAnnealingLR`'s `T_max` is the epoch count,
so resuming a 60-epoch run under `--epochs 80` restores `last_epoch` into a schedule with a
different period and produces an LR curve matching neither run, silently. The checkpoint now
records `epochs` and `--resume` refuses on mismatch. The first resume test ran before this guard
existed and hit exactly that case.

`--init-from` is deliberately a separate flag: it loads weights only and starts a fresh run at
epoch 0 with a fresh optimiser and schedule. Conflating the two would be wrong in both
directions -- fine-tuning wants a fresh schedule at a new LR, resuming wants the original
schedule continued.

### Item 3 measured: edge features in the uncertainty head are a real but small win

Testable cheaply because the head is decoupled from the core: `train_pretrain.py` feeds it
`pred.detach()` and the NLL uses the detached prediction as its mean, so no gradient from the
head reaches SpectraHATCore. The head is therefore strictly post-hoc and can be ablated against
a FROZEN run-10 core -- isolating the one variable instead of confounding it with a differently
trained core, and costing ~7 minutes per arm instead of a full retrain.
(`scripts/ablate_uncertainty_head.py`; both arms: same frozen core, same seed, same data order,
2000 steps, scored on 200 held-out ROIs the core never trained on.)

| metric | baseline | edge | delta |
|---|---|---|---|
| per-tile corr, mean | 0.1580 | 0.1767 | +0.0187 |
| per-tile corr, median | 0.1612 | 0.1889 | +0.0277 |
| tiles positively correlated | 75.0% | 84.5% | +9.5 pp |
| NLL | -1.8613 | -1.8477 | +0.0137 (worse) |
| ECE recalibrated | 0.0313 | 0.0337 | +0.0025 (worse) |

Paired over the same 200 tiles: mean delta +0.0187, 95% CI [+0.0062, +0.0311] (excludes zero),
paired t p=3.70e-03, Wilcoxon p=2.20e-02, **Cohen's d 0.208, win rate 55.5%**.

**Honest reading: statistically real, practically small.** d=0.208 with a 55.5% win rate means
the effect holds in aggregate but barely beats a coin flip on any individual tile. Kept anyway,
for a reason the mean does not capture: anti-correlated tiles fall from 25% to 15.5%, and an
uncertainty map that points *away* from the error on a quarter of scenes is a liability for a
problem statement that names uncertainty twice. The NLL/ECE cost is absorbed entirely by scalar
recalibration (both arms end "well calibrated"), and the features are parameter-free at the
input, costing 576 extra weights and no information unavailable at inference.

A paired test is the only honest one here: between-tile variance in how predictable a scene's
error is dwarfs the difference between the two heads, so an unpaired test would be badly
underpowered and would report "no effect" regardless of the truth.

### Incidental, and bigger than the effect being tested: train the head AFTER the core

Both arms came out at recalibration factor **1.004-1.006** ("well calibrated" with no
correction), against run 10's **1.19x over-confidence**. The only structural difference is that
these heads were trained against a frozen core, while run 10's was co-trained with a core whose
error distribution was still moving underneath it -- so run 10's head was fitting a moving
target and ended up systematically over-confident.

Recommendation for the big run: train the core, then fit the uncertainty head post-hoc against
the frozen result. It removes the need for a recalibration step and costs one short extra pass.

## SEN2NAIPv2: more real data, but a measurably easier task

Found while deciding whether to download v1's 177 GB synthetic component. The v1 README points
to a v2 release, and v2's cross-sensor variant is the highest-value data available by a wide
margin:

| | v1 cross-sensor | **v2 cross-sensor** | v1 synthetic |
|---|---|---|---|
| pairs | 2,851 | **8,000** | 17,657 |
| size | 2.2 GB | **9.7 GB** | 180 GB |
| real Sentinel-2 | yes | yes | **no -- LR is simulated** |
| pairing window | same-day | 0 or +/-1 day | n/a |
| cloud screening | -- | 0% cover (CloudSen12 UnetMob-V2) | n/a |

2.8x more of the real task for 1/18th the download of the synthetic set. All 8,000 rows were
inspected remotely via `tacoreader` before committing to the download -- the archive's metadata
is readable over HTTP without pulling the payload.

### Three silent incompatibilities with the v1 loader (all measured, none raise)

1. **Tile geometry** is 520/130 px, not 484/121.
2. **HR dtype is uint16 in Sentinel-2 reflectance units, not uint8 NAIP.** Applying v1's `/255`
   rule to v2's HR overshoots by ~40x. Nothing raises; the images are simply wrong. Measured HR
   means: v1 122.3 (uint8 range), v2 1420.2 (reflectance x10000).
3. **HR is already radiometrically harmonized to the Sentinel-2 scale.** Per-band HR/LR mean
   ratios over 20 ROIs: 1.000 / 0.999 / 0.999 / 1.000 -- max deviation 0.07%, against v1's
   2.0-5.3x per-band mismatch. Applying `calibrate_lr_to_hr_radiometry` to v2 would INTRODUCE the
   error that function exists to remove.

Handled by `DATASET_VARIANTS` in `sen2naip_dataset.py` plus a `variant=` argument threaded
through every call site (7 scripts). The variant supplies the *default* for
`radiometric_calibration`, so the correct behaviour is not something a caller has to remember.

### The important caveat: v2 numbers are NOT comparable to v1 numbers

v2's HR was harmonized *using the real Sentinel-2 as reference*, which pulls the target toward
the input. Measured, over 20 held-out tiles each:

| | corr(avgpool4(HR), LR) | median relative residual |
|---|---|---|
| v1 | 0.850 median, min 0.359 | 15.0% |
| v2 | 0.995 median, min 0.970 | 3.8% |

The LR is still genuinely independent -- a 3.8% residual means it is not merely a downsampled
HR, so there is no leakage -- but v2 is a materially easier reconstruction problem than v1.

Consequences, to be applied when reporting:

* **Model-vs-bicubic comparisons remain valid within a release**, because both face the same
  target. Those are the headline numbers.
* **Absolute PSNR must never be compared across releases.** A jump from switching to v2 would
  be an easier target, not a better model. Run 10's +0.6948 dB was measured on v1 and stays the
  reference point.
* Plan: train on v2 for the extra data, and evaluate on held-out **v1** as well -- the harder,
  more realistic pairing, and the one every earlier number was measured against.

### Item 2 measured: raising spectral loss weights tightens SAM, but the gain does not propagate downstream

Three 3-epoch fine-tunes from run 10's converged core (`--init-from`, `--lr 5e-5`,
`scripts/sweep_spectral_weights.sh`): control (w_sam=0.3, w_index=0.2, i.e. run 10's own
weights), mid (1.0/0.6), high (2.0/1.2). A control arm at the ORIGINAL weights is included
because fine-tuning itself moves every metric -- without it, any change in the other arms is
unattributable.

Best epoch per arm, weight-independent metrics only (`val_total_loss` is not comparable across
arms: each optimises a different objective by construction):

| metric | control | mid | high | control's own drift |
|---|---|---|---|---|
| val_psnr | 22.4335 | 22.4343 | 22.3553 | +0.2704 |
| val_ssim_metric | 0.6574 | 0.6567 | 0.6558 | +0.0028 |
| val_rmse | 0.0811 | 0.0813 | 0.0821 | -0.0024 |
| val_sam_degrees (lower better) | 4.5437 | 4.4097 | 4.3817 | -0.0861 |
| val_downstream_improvement | 0.0166 | 0.0171 | 0.0170 | +0.0007 |

**PSNR, SSIM, RMSE and the downstream classification metric are all noise.** Every gap between
arms on those four is smaller than the control's own drift from fine-tuning alone (+0.27 dB
PSNR from nothing but 3 epochs at a lower LR) -- so nothing beyond "more training helps" can be
claimed from them.

**`val_sam_degrees` is different.** It moves monotonically with the weight -- control 4.54 ->
mid 4.41 -> high 4.38 -- and both mid's and high's improvement (0.13, 0.16 degrees) exceed the
control's own drift (0.09 degrees). That is the signature of a real, dose-responsive effect
rather than noise, though on n=3 epochs per arm with no per-tile pairing this is suggestive
evidence, not a tested significance claim (contrast the uncertainty-head ablation, which had
200 paired tiles and a formal test).

**Read: raising w_sam/w_index buys spectral-angle accuracy specifically, at no cost to pixel
fidelity, but that gain does not convert into the downstream NDVI-classification metric.** It is
real by one measure and inert by the one closer to what the PS cares about.

**Recommendation for the big run: keep the current 0.3/0.2 default.** The metrics that matter
most (PSNR, downstream accuracy) show no benefit from raising the weights, and `high` trades a
real fidelity cost for extra SAM tightening that a downstream task does not use. If spectral
accuracy becomes a target in its own right (e.g. an NDVI-specific use case in the demo), a
`mid`-range weight is defensible; `high` is not.
