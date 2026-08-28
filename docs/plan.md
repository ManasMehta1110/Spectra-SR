# SPECTRA-SR — Full Execution Plan (SIH26142, Sentinel-2 Super-Resolution)

**Status note (post run-10):** this plan was written before any training run existed. Section 4's
data-source priority order (PlanetScope → archival Cartosat-2 → NAIP → Maxar) and the "10m
Sentinel-2 + NAIP synthetic pretraining" framing in Section 1/5 describe the *original* plan --
what actually happened is documented in `findings.md`. In short: the SEN2NAIP cross-sensor
dataset (real Sentinel-2 paired with real NAIP, not synthetic) became the primary data source and
is what run 10 was trained and evaluated on; PlanetScope E&R access is now confirmed working
(search API verified against real AOIs, download provisioning pending); the Cartosat-2/Maxar
threads below are superseded, not pursued. Sections 2/3/6/8 (compute strategy, scope gating,
verification) and the architecture note still hold as written, with the parameter counts
corrected against what was actually measured (see the architecture note at the bottom). Treat
this file as the original intent; `findings.md` as what is real.

## Context

The team has locked in two SIH 2026 problem statements to pursue — **SIH26142** (NTRO, Sentinel-2 10m → <4m super-resolution) and **SIH26170** (ISRO, component burn-in anomaly detection) — and decided to work SIH26142 first, sequentially, with the full 6-person team. A complete architecture spec for SIH26142 ("SPECTRA-SR") already exists, written by the user, and has been audited for risk (blind-kernel-estimation instability, pixel-loss blurring of fine linear features, wall-clock cost of ensembles, HR-reference-data availability, registration/temporal mismatch, tooling-complexity jump from the team's prior EDSR work). The user now wants the *whole plan* — sequencing, data strategy, compute strategy, scope gating, and reused-code mapping — fully locked and audited before any implementation begins, mirroring the ablation/verification discipline already visible in the team's prior work.

A directory scan confirmed the team's "prior thermal SR study" is a real, complete repo at `C:\Projects\optical-guided-super-resolution` (GRSL-track ablation study, `DualEDSRPlus` model on NASA HLS Landsat-8/Sentinel-2-harmonized data) — not a from-scratch situation. This plan is built around reusing that codebase's proven pieces rather than rewriting them, and treats the new work (blind degradation estimation, MISR fusion, diffusion residual, multi-source uncertainty, guardrails) as genuinely new components layered on top of a working foundation.

Team answers locked in: **fully sequential** (all 6 on 142 until it hits a stable milestone, then shift to 170), **timeline kept relative** (no hard calendar dates yet), and an open compute question (2 laptops — RTX 3050 + RTX 4050) that this plan resolves explicitly below.

**This is a hackathon submission, not a research paper — but that means pragmatic about *scope*, not about *authenticity*.** The prior project (`optical-guided-super-resolution`) is a real GRSL journal submission, and some of its defaults (5 seeds per ablation variant, 9 development regions across 5 continents, exhaustive calibration) are more than this needs. Scale those down. But the goal is not a smaller/toy version of the system -- the goal is that the team is genuinely *different from other teams* on rigor, and that whatever gets shown or claimed can actually be defended if a judge probes it at finals. Concrete implications:
- **Round 1 is a PPT submission only** — no working code required, but the architecture story and reasoning in the deck have to be airtight, since that reasoning *is* the round-1 deliverable in full.
- **Ablation seeds: 2–3, not 5; AOI count: 3–4 Indian regions + 1 held-out, not 9 across 5 continents.** Scope scaled down, not skipped — still a real paired test, still a real held-out region.
- **The demo must be the real deal, not staged.** Real trained weights, real inference on real Sentinel-2 tiles, real (even if modest) numbers — never a mocked pipeline, a filter dressed up as SR, or cherry-picked best-case examples presented without disclosure. It is completely fine for the demo to only cover Core scope with Stretch pieces still in progress by finals — that gets framed honestly ("this part ships by finals, here's the plan"), not hidden or faked. Per the spec's own deck-outline advice: the team that states the limitation first controls the framing; a team that gets caught faking a result controls nothing.
- **The bar for every Core gate below is "good enough to defend in Q&A," not "statistically airtight" — but it must be real.** Keep the acceptance tests, the one clean MISR ablation, the calibration check — they're cheap, they're the actual differentiator, and they're literally the material the team defends with if a judge asks "how do you know this works." Scale their size down; never fake their result.
- **Stretch is opportunistic, read skeptically, and always disclosed as in-progress rather than silently dropped or silently faked.**

---

## 1. What already exists and gets reused (not rebuilt)

From `C:\Projects\optical-guided-super-resolution\src\optical_guided_sr\`:

| File | Reused for SPECTRA-SR as |
|---|---|
| `stats.py` (`paired_variant_test`, `compare_all_to_reference`) | **Drop-in**, unmodified. This is exactly the seed-matched paired t-test / Wilcoxon harness SPECTRA-SR's own ablation grid (its Section 5.3) needs. Do not rewrite this. |
| `preprocessing.py` (`extract_and_save_bands`, `resample_thermal_to_optical`, `_mask_fill`, granule-grouping-by-filename-not-position) | **Pattern reused, code adapted.** Same fill-masking, `rasterio.warp.reproject`-based co-registration, and "group files by parsed ID, never by list position" discipline apply directly to Sentinel-2 band handling. Rewritten for Sentinel-2's actual band set (B2/B3/B4/B8 native 10m + optional 20m bands) instead of HLS L30/S30. |
| `model.py` (`DualEDSRPlus`, `ChannelAttention`/`SpatialAttention`/RCAB, `_icnr_init` for PixelShuffle) | **Kept as the baseline ablation variant**, not discarded. SPECTRA-SR's own methodology (Section 5.3, `full` vs. ablated variants, shape-matched) demands a real prior baseline to compare HAT against — `DualEDSRPlus` *is* that baseline, already validated on a real satellite-SR task. The ICNR PixelShuffle init is reused verbatim in any new upsampling head. |
| `losses.py` (L1 + SSIM) | **Starting point**, extended with SAM (spectral angle) and band-ratio/index-preservation terms per the SPECTRA-SR spec — the existing repo already found SSIM's loss-term contribution to be real and significant, so keep it rather than re-deriving that finding. |
| `data_acquisition.py`, `manifest.py`, `dataset.py`, `config.py`, `utils.py`, `demo_data.py`, `train.py`, `tests/` | **Structural template.** Same repo skeleton (config-driven, manifest-tracked acquisition, tested dataset/loss/model/stats modules) gets reused for the new project's scaffolding rather than inventing a new project layout. `rag-poison-robustness` uses the same `src/`+`paper/`+`analysis/`+`tests/` skeleton — this is the team's established personal project template; follow it. |

**Net effect:** Stage 3's deterministic core, Stage 1's preprocessing, and the entire ablation/statistics harness (Section 5.3 + Section 6.1 of the SPECTRA-SR spec) are adaptation work on a working base, not net-new research code. Stage 0 (blind degradation estimation), Stage 2 (MISR fusion), Stage 4 (diffusion residual), Stage 5 (data-consistency projection), and the full 3-source Stage 6 uncertainty stack are genuinely new — budget them accordingly.

---

## 2. Compute strategy (resolves the open question)

**The two laptops (RTX 3050, 4.3GB VRAM confirmed by direct measurement on the dev machine; RTX 4050, 6GB confirmed) are not sufficient for training the deterministic HAT core at a meaningful patch/batch size.** Measured on the 3050: `COLAB_REALISTIC` needs 1.86GB at batch=1, 3.70GB at batch=2 (right at that GPU's edge), and only "succeeds" at batch=4 via a slow system-RAM fallback, not real VRAM; `FULL` OOMs even at batch=4. The 4050's extra 1.7GB gives more real headroom (batch=2 comfortably, maybe batch=3 before hitting the same fallback), but neither laptop is a real training platform for this architecture. HAT-style window-attention transformers are materially more memory-hungry than `DualEDSRPlus` (2.67M params) — a ~20M-param HAT config at 128×128 LR patches with reasonable batch size typically wants far more headroom than 4-6GB, and starving it down to batch=1 with tiny crops undermines the exact reason HAT was chosen over EDSR (modeling long-range repeated structure — field grids, urban blocks — which needs a large enough window/patch to see that structure at all).

**Recommendation:**
- **Laptops (3050/4050): local dev only** — data acquisition scripting, preprocessing/co-registration pipeline (mostly CPU/IO-bound via `rasterio`/GDAL, not GPU-heavy), Stage 0 degradation-operator fitting (a numerical optimization problem, not deep learning — light enough for either laptop), unit tests (mirroring the prior repo's `tests/` coverage), and tiny-scale smoke tests of Stage 3 training code (batch=1, small crop) purely to confirm the code runs without crashing before it goes to real compute.
- **Real training: Colab Pro** (or better if it becomes available) — the same proven infra from the GroundingBench project. It already has known failure modes the team has direct experience mitigating (session disconnects, quota limits) via periodic HF Hub checkpoint backup, which should be set up for this project from day one rather than rediscovered under pressure.
- **`gpuenv`** (the existing shared virtualenv at `C:\Projects\gpuenv`, already has `rasterio`, `torch+cu128`, `transformers`, `wandb`, `statsmodels`) is the right base environment for local dev — reuse it rather than creating a new one from scratch.

This directly gates the scope decision in Section 3: stretch-goal stages (deep ensembles = 5× training cost; the diffusion residual) are the first things to defer if Colab Pro's session limits make them impractical, not things to force onto the laptops.

---

## 3. Scope: locked core vs. gated stretch

Per the earlier risk audit, the SPECTRA-SR spec's 7 stages split cleanly into a **core that is a complete, defensible, demoable submission on its own**, and a **stretch tier** that should only be built once the core is validated and there's real time/compute budget left. This directly answers "fully plan before building" — the plan itself should not commit to building all 7 stages unconditionally.

**Core (build this, in order):**
1. **Stage 0** — Degradation operator `A`, with its acceptance test (`‖A(HR_real) − S2_real‖` within sensor NEΔρ) enforced *before* anything downstream is trusted. This is the single highest-leverage step, per the team's own prior finding (63.6→32.9 dB when degradation modeling was done properly).
2. **Stage 1** — Preprocessing: band selection, cloud/shadow masking, BRDF/sun-angle normalization, sub-pixel co-registration. Adapted from `preprocessing.py`.
3. **Stage 3** — HAT deterministic core, trained with the fidelity-loss stack (Charbonnier + SSIM + SAM + index-preservation + re-degradation cycle), evaluated against `DualEDSRPlus` as the shape-matched baseline ablation.
4. **Stage 5** — Data-consistency (null-space) projection. This is what makes the "structurally incapable of contradicting the input" claim literally true, and it's cheap relative to Stages 2/4/6-full.
5. **Stage 6, slimmed** — start with *one* uncertainty source (heteroscedastic NLL head *or* a small ensemble of 2-3 models, not both, not K=32 diffusion samples) plus the deterministic observational-support features (temporal count, cloud distance, registration residual). Calibrate this on held-out data. A slimmed-but-real, calibrated uncertainty product satisfies the PS's twice-stated requirement; it does not need to be the maximal 3-source version to do that.
6. **Stage 7** — Guardrails (spectral/radiometric/index/geometric checks, OOD refusal). Cheap, high credibility, low risk.

**Stretch (defer until core is validated and time/compute allow):**
- **Stage 2 (MISR multi-temporal fusion)** — build the `no_temporal_fusion` ablation *first* as the actual baseline, since the team's own prior guided-fusion result was negative (+3.77 dB *without* the auxiliary stream) and this component is explicitly flagged as needing the same scrutiny. Only invest further if the ablation shows a real, significant gain.
- **Stage 4 (diffusion residual)** — the "how sharp does it look" differentiator, and the most training-time-expensive new component. Build only after the deterministic core (Stage 3) is fully validated and shipped as a working fallback, so a stall here never blocks having a complete submission.
- **Full Stage 6** (all 3 uncertainty sources fused, K=32 sampling, 5-model ensemble) — upgrade from the slimmed core version only if compute/time genuinely allow it; this is also the step most exposed to the Colab Pro session-limit risk (Section 2).
- **Section 10 (20m→10m band SR)** — genuinely low-risk, high-credibility optional second contribution (same sensor, zero registration error), but explicitly last in priority — it's a bonus, not core scope.

**Why gate it this way:** a team that ships Core only still has a complete, defensible, non-hallucinating submission with a real differentiator (the null-space framing + calibrated uncertainty). Everything in Stretch adds polish or extra rigor but is not required for the pitch to stand on its own — which matters given this is new tooling (attention-heavy transformers, diffusion, multi-source UQ) the team hasn't built before, on a compute budget that needs a cloud fallback.

---

## 4. Data acquisition — start immediately, in parallel with Stage 0/1 code

This is the one external-dependency risk flagged as Medium/Critical in the SPECTRA-SR spec's own risk register, and it does not get faster by writing more model code first. Kick off on day one, independent of Stage 0/1/3 implementation progress.

**Sentinel-2 (LR) source — confirmed.** The PS explicitly points to the Copernicus Data Space Ecosystem (`browser.dataspace.copernicus.eu`), matching the SPECTRA-SR spec's tech stack (`sentinelhub-py`/`openeo` + STAC access). Use the STAC API programmatically (free CDSE account, OAuth2 credentials, query by AOI/date/cloud-cover) for bulk acquisition — the browser UI alone doesn't scale to training-pair volume. This resolves the LR side; the HR reference side below remains the actual open risk.

**Update (real finding, changes this section): Cartosat-2 was deorbited February 2024 -- it is not capturing imagery anymore.** Any Cartosat-2 data on Bhuvan/Bhoonidhi is archival, pre-2019, which fails the pairing protocol below outright against 2026 Sentinel-2 (7+ year gap; real land-cover change). Its successor, Cartosat-3 (current, ~0.25-1m), is free only for Indian *government agencies* under a declaration form (Space Policy 2023) -- a hackathon team goes through NSIL commercially, ~Rs 3,860/scene. Explicitly decided against budgeting for this (team cost constraint) -- Cartosat-3 purchase is opportunistic-only if spare budget appears later, not a load-bearing part of this plan.

Revised, all-free priority order:
1. **PlanetScope education/research access** -- the primary real validation-tier HR source now. Best temporal match to Sentinel-2 (same-day pairs achievable), ~3m resolution. Apply now; approval-lag risk is real and unknown, so this should be in flight as early as possible.
2. **Archival Cartosat-2 (free, via Bhuvan/Bhoonidhi)** -- reframed, not discarded: can't serve real validation pairs (temporally stale), but is genuinely useful *pretraining-tier* volume with the correct geography (Indian terrain) that NAIP lacks. Free. Pursue as a NAIP upgrade for the pretrain tier specifically.
3. **NAIP** -- supplementary pretraining volume (already flowing, real data confirmed working). Geography-mismatched but zero-friction.
4. **Maxar Open Data** -- only needed once the disaster-response use case (Section 9 of the spec) is being demonstrated, lower priority than 1-3.

Apply the pairing protocol (Section 4.3 of the spec — same-day/≤3-day, co-registration residual <0.2 Sentinel-2 px, cross-sensor radiometric calibration) as a hard filter from the first batch of data onward, not retrofitted later — a bad pair silently teaches the network to blur, and that's exactly the kind of error that doesn't show up until much later per the Stage 0 acceptance-test discipline already established.

**Open thread:** the AOI list itself should shift from the prior project's 9-region/5-continent set (chosen for a global-generalization journal claim) toward an India-weighted, land-cover-diverse set (urban / agricultural / arid / coastal / forested) for the real fine-tune/validate tiers, given HR reference availability is concentrated in India (Bhuvan) and the audience is NTRO, not a journal reviewer — keep the same held-out-region discipline ("the Perth protocol"), just with a freshly chosen Indian region held out instead. Not yet finalized to specific AOIs.

---

## 5. Build order with explicit go/no-go gates

Each phase has a concrete stop-and-check condition before moving on — this is the "audited before building" discipline the user asked for, applied at every transition, not just at the start.

| Phase | Deliverable | Gate before proceeding |
|---|---|---|
| 0. Scaffolding | New repo (`spectra-sr`, same skeleton as `optical-guided-super-resolution`/`rag-poison-robustness`), `gpuenv` confirmed working, HF Hub checkpoint backup wired up before first real training run | Repo structure + CI mirrors the prior repo's `tests/` coverage pattern |
| 1. Degradation operator (Stage 0) | `A` as a differentiable module + fitting procedure | **Acceptance test passes**: `A(HR_real)` matches real Sentinel-2 within band NEΔρ. Do not proceed if this fails — fix `A`, don't route around it. |
| 2. Preprocessing (Stage 1) | Masking, BRDF/sun-angle normalization, co-registration, tiling | Spot-check registration residual on a sample of pairs (<0.2px target) |
| 3. Deterministic core (Stage 3) | HAT trained on synthetic pairs, then fine-tuned on real Tier 2 pairs | Beats `DualEDSRPlus` (adapted as the shape-matched baseline) on held-out synthetic pairs before spending real-pair fine-tuning budget on it |
| 4. Null-space projection (Stage 5) | Data-consistency projection wired into inference | Re-degraded output reproduces real LR input within numerical precision |
| 5. Slimmed uncertainty (Stage 6-core) | One UQ source, calibrated | Reliability diagram check on held-out data — predicted confidence must actually track real error before this is presented as a feature |
| 6. Guardrails (Stage 7) | Spectral/radiometric/geometric/index checks + OOD refusal | Guardrail thresholds validated against known-good and known-bad tiles |
| 6.5. **Demo** (runs in parallel from Phase 3 onward, not strictly sequential) | Inference script + before/after visualization, ideally the lightweight web viewer (spec Section 7) with an SR/confidence toggle -- real weights, real inference, real numbers, no mocked pipeline | Runs live end-to-end on at least one real scene before finals; anything still Stretch at that point is disclosed as in-progress with a completion plan, never hidden or faked |
| **— Core complete: a full, defensible submission exists here —** | | |
| 7. MISR ablation (Stage 2, stretch) | `no_temporal_fusion` baseline built *first*, then the fusion variant | Only keep Stage 2 in the final system if the paired significance test (reusing `stats.py`) shows a real gain — a negative result here is reported, not hidden, per the team's own established norm |
| 8. Diffusion residual (Stage 4, stretch) | Trained with the deterministic core frozen | Deterministic core (already complete) remains shippable regardless of how this goes |
| 9. Full uncertainty stack (Stage 6-full, stretch) | Ensemble + diffusion-sample UQ, all 3 sources fused | Only attempted if Colab Pro budget genuinely allows the 5× training multiplier |

---

## 6. Risk register (carried over, refined with what's now confirmed)

| Risk | Status |
|---|---|
| HR reference data unobtainable | **Resolved for non-Indian training/eval** — the SEN2NAIP cross-sensor dataset (real Sentinel-2 x real NAIP, not synthetic) supplied 2,851 pairs (v1) and a further 8,000 (v2); run 10 trained and was evaluated on v1. PlanetScope E&R access is confirmed working (search API verified against real AOIs, incl. Indian coordinates; download provisioning still pending). Indian fine-tune/validate remains open but is not currently blocking -- deprioritized per team decision to focus on the core training run first. |
| Degradation operator mis-specified | **Mitigated by design** — Stage 0 acceptance test is a hard gate (Section 5) |
| Pixel-loss blurs fine linear features (roads, boundaries) | **Open** — add an edge-aware/gradient-domain auxiliary loss term to Stage 3's loss stack; do not treat "deterministic = risk-free" |
| MISR fusion may not help (team's own prior guided-fusion result was negative) | **Mitigated by design** — ablation-first build order (Section 5, phase 7), reusing `stats.py` directly |
| GPU access insufficient for core training | **Resolved** — laptops are dev-only, Colab Pro is the training platform (Section 2); flag immediately if Colab Pro access itself becomes unavailable |
| Ensemble/diffusion wall-clock cost | **Mitigated by scope gating** — both are stretch-tier, attempted only after core ships (Section 3) |
| Tooling-complexity jump from prior EDSR work | **Mitigated by reuse** — `stats.py`, `preprocessing.py` patterns, and `DualEDSRPlus`-as-baseline all transplant directly (Section 1), so the genuinely new surface area is scoped to Stages 0/2/4/5/6/7, not the whole system |
| Cross-sensor spectral mismatch | **Open** — enforce the pairing protocol's radiometric cross-calibration step from the first data batch |

---

## 7. Deliverables per stage of the competition

- **Qualify-to-top-5 (PPT submission, no code required):** the deliverable *is* the deck -- architecture story (spec Section 13's outline), the reasoning that separates this from a generic "make it sharper" reading of the PS, the credibility slide (prior GRSL work, the degradation-modeling finding). This has to be airtight on its own, since nothing else is submitted at this stage. If a working Core-tier proof-of-concept exists by then, use it as supporting evidence -- but don't claim results in the deck that aren't backed by something real yet; state what's built vs. planned honestly.
- **Finals:** a live, real demo of at least Core (Phases 0-6.5) -- real weights, real inference, real numbers. Stretch components included only to the extent Section 3's gating allowed them to be built and genuinely validated. Anything not finished by finals gets disclosed as in-progress with a clear completion story, never faked or silently dropped -- a team that states its own limitation first controls the framing in Q&A; a team caught overclaiming loses the room.

---

## 8. Verification

- Stage 0: automated acceptance test (residual RMSE vs. band NEΔρ) as a CI-style check, not a manual eyeball.
- Stage 3: reuse `compare_all_to_reference` from `stats.py` against the `DualEDSRPlus` baseline, 5 seeds, paired significance — same rigor bar as the team's prior published work.
- Stage 5: numerical re-degradation consistency check.
- Stage 6: reliability diagram + Expected Calibration Error on held-out data.
- Full system: the Section 6 metric suite from the SPECTRA-SR spec (fidelity + hallucination metrics + downstream-task ablation + calibration), run once Core is complete, before any Stretch work begins.

---

## Architecture note — Stage 3 (locked in conversation, `src/spectra_sr/model.py`)

Shallow feature extraction → N × Residual Hybrid Attention Group (windowed self-attention +
`ChannelAttention` reused verbatim from `optical_guided_sr.model`, since its ablation already
showed a real, significant contribution — +4.41 dB in the `no_attention` variant — plus one
Overlapping Cross-Attention Block per group for long-range repeated-structure modeling, the
actual reason to pick this over `DualEDSRPlus`'s pure-CNN receptive field) → global residual
over a bicubic-upsampled copy of the input (same EDSR-family philosophy `DualEDSRPlus` already
uses) → two chained 2× `LearnedUpsampler` stages (ICNR-initialized PixelShuffle, ported
verbatim) → output conv. Deliberately deterministic throughout — no adversarial or diffusion
component in Stage 3, for the same hallucination-risk reason `DualEDSRPlus`'s own GAN
alternative was rejected on the prior project.

Two configs: `FULL` (180 embed dim, 6×6 groups/blocks, window 16, **15,514,816 params measured**)
and `COLAB_REALISTIC` (112 embed dim, 4×4, window 8, **3,572,532 params measured** -- a 4.34x
capacity ratio between the two, not the ~3x the original estimate implied) — `COLAB_REALISTIC`
cleared its own bar at run 10 (+0.69 dB over bicubic, 74% win rate, still improving when training
stopped -- see `findings.md`); `FULL` is the next run, not yet trained even once.
