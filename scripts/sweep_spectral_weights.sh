#!/usr/bin/env bash
# Item 2 validation: should the big run raise the spectral loss weights?
#
# The measured weakness this targets: with radiometry equalised, the model is only TIED with
# bicubic on NDVI classification (+0.0013). Its extra spatial detail buys no spectral accuracy.
# At w_sam=0.3 / w_index=0.2 against charbonnier=1.0, the objective is dominated by pixel
# fidelity, so that is the obvious suspect.
#
# Each arm FINE-TUNES from run 10's converged core (--init-from) rather than training from
# scratch. Three from-scratch runs would cost ~15 hours to answer a question about the direction
# and size of a trade-off; starting from a converged core makes the effect of the weight change
# visible immediately, because the only thing changing is the objective. The honest limitation:
# this measures "what happens if you re-weight a model already converged at 0.3/0.2", which is
# not identical to "what happens if you train from scratch at 1.0/0.6". It is a strong signal
# for the direction of the effect, not a substitute for the big run itself.
#
# The control arm re-runs at the ORIGINAL weights. Without it, any movement in the other arms is
# unattributable -- every arm also receives extra training at a new learning rate, and that alone
# moves the metrics.
#
# Compare with:
#   python scripts/compare_runs.py checkpoints/sweep_{control,mid,high} \
#     --labels control mid high --control control
# and compare only on weight-INDEPENDENT metrics: each arm optimises a different objective, so
# val_total_loss is not comparable across arms. compare_runs.py enforces this.
set -euo pipefail

PY="${PY:-/c/Projects/gpuenv/Scripts/python.exe}"
INIT="${INIT:-checkpoints/pretrain_run10/checkpoint_best.pt}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"   # an order below the 2e-4 the core was trained at: fine-tuning a converged
                   # model at full LR would mostly measure the disruption, not the new objective.

run_arm () {
  local name="$1" w_sam="$2" w_index="$3"
  echo "=== arm ${name}: w_sam=${w_sam} w_index=${w_index} ==="
  PYTHONPATH=src "$PY" scripts/train_pretrain.py \
    --config colab_realistic --res-scale 0.2 \
    --data-source sen2naip \
    --epochs "$EPOCHS" --batch-size 1 --lr "$LR" \
    --sen2naip-train-crops 2 --sen2naip-val-crops 1 \
    --w-sam "$w_sam" --w-index "$w_index" \
    --init-from "$INIT" \
    --keep-last-n 1 \
    --out "checkpoints/sweep_${name}"
}

run_arm control 0.3 0.2
run_arm mid     1.0 0.6
run_arm high    2.0 1.2

echo "=== comparison ==="
"$PY" scripts/compare_runs.py \
  checkpoints/sweep_control checkpoints/sweep_mid checkpoints/sweep_high \
  --labels control mid high --control control
