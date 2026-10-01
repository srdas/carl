#!/usr/bin/env bash
# long_train.sh - Extended offline training run for goddard-maxq PPO.
#
# Usage:
#     bash long_train.sh
#
# Env-var overrides (with defaults tuned for a longer horizon):
#     UPDATES       total PPO updates (default: 600, 10x in-session 60)
#     CONS_UPDATES  conservative-phase updates (default: 0)
#     CRITIC_WARMUP value-only warmup updates (default: 8)
#     SEED          RNG seed (default: 0)
#     RUN_NAME      experiments/<RUN_NAME> dir (default: rocket_long)
#     PATIENCE      early-stop patience across evals (default: 30)
#     EVAL_EVERY    updates between intermediate evals (default: 15)
#     EVAL_EPS      seeds per scenario during eval (default: 8)
#     SAVE_EVERY    updates between periodic checkpoints (default: 50)
#     INIT_FROM     path to initial model (default: experiments/pretrain/init_actor.pt)
#                   Set to "" to run BC/critic warmup fresh instead.
#
# The default INIT_FROM points at the file CARL wrote during the in-session
# run, so this long run starts from the same initial model -- the only
# difference is training budget.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${UPDATES:=1000}"
: "${CONS_UPDATES:=0}"
: "${CRITIC_WARMUP:=20}"
: "${SEED:=0}"
: "${RUN_NAME:=rocket_long}"
: "${PATIENCE:=50}"
: "${EVAL_EVERY:=10}"
: "${EVAL_EPS:=16}"
: "${SAVE_EVERY:=50}"
: "${INIT_FROM:=$SCRIPT_DIR/experiments/pretrain/init_actor.pt}"

echo "==============================================================="
echo " goddard-maxq LONG RUN"
echo "==============================================================="
echo " UPDATES=${UPDATES}  CONS_UPDATES=${CONS_UPDATES}"
echo " CRITIC_WARMUP=${CRITIC_WARMUP}"
echo " SEED=${SEED}  RUN_NAME=${RUN_NAME}"
echo " PATIENCE=${PATIENCE}  EVAL_EVERY=${EVAL_EVERY}  EVAL_EPS=${EVAL_EPS}"
echo " SAVE_EVERY=${SAVE_EVERY}"
echo " INIT_FROM=${INIT_FROM}"
echo "==============================================================="

mkdir -p "experiments/${RUN_NAME}"

INIT_FLAG=()
if [ -n "${INIT_FROM}" ]; then
    if [ ! -f "${INIT_FROM}" ]; then
        echo "ERROR: INIT_FROM=${INIT_FROM} not found."
        echo "Either run BC first (unset INIT_FROM) or point to a valid init_actor.pt"
        exit 1
    fi
    # Skip BC in train.py since we're loading pretrained weights.
    INIT_FLAG=(--init-from "${INIT_FROM}" --skip-bc)
fi

python -u train.py \
    --updates "${UPDATES}" \
    --cons-updates "${CONS_UPDATES}" \
    --critic-warmup "${CRITIC_WARMUP}" \
    --seed "${SEED}" \
    --run-name "${RUN_NAME}" \
    --patience "${PATIENCE}" \
    --eval-every "${EVAL_EVERY}" \
    --eval-eps "${EVAL_EPS}" \
    --save-every "${SAVE_EVERY}" \
    "${INIT_FLAG[@]}" \
    2>&1 | tee "experiments/${RUN_NAME}/train.log"

echo ""
echo "=== Post-training evaluation ==="
python -u evaluate.py 2>&1 | tee -a "experiments/${RUN_NAME}/train.log"

echo ""
echo "=== Generating plots ==="
python -u plots.py --run-name "${RUN_NAME}" 2>&1 | tee -a "experiments/${RUN_NAME}/train.log"

echo ""
echo "Long run complete. Results in experiments/${RUN_NAME}/"
