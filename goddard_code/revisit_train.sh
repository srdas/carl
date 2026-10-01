#!/usr/bin/env bash
set -euo pipefail

# Step 10 Revisit 1 training pipeline.
#
# Revisit 1 changes (relative to Step 9 / rocket_long):
#   1. New state features: t_remaining_frac, u_constant_hint (u_hint=1 for
#      first 5% of t_target so BC/PPO don't forget to boost hard at start).
#      OBS_DIM: 12 -> 14.
#   2. One-sided early-burnout penalty: penalty at fuel-out step scales with
#      how much earlier than t_target = 4 * full_burn_time the fuel ran out.
#      penalty = -PENALTY_COEF * deficit^2 with PENALTY_COEF = 20 (fixed
#      constant in raw_dh reward units; no per-episode scaling).
#   3. Fresh init (no --init-from): BC pretrain is rerun locally in this
#      workspace with the new obs_dim=14 and new features.
#   4. Training schedule: 500 PPO updates, LR 4e-4 -> 2e-4 (linear anneal).
#
# Everything writes to code/revisit_1/experiments/revisit_1/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_NAME="${RUN_NAME:-revisit_1}"
RUN_DIR="experiments/${RUN_NAME}"
UPDATES="${UPDATES:-500}"
CONS_UPDATES="${CONS_UPDATES:-0}"
WARMUP="${WARMUP:-10}"
EVAL_EVERY="${EVAL_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-50}"
PATIENCE="${PATIENCE:-25}"

mkdir -p "$RUN_DIR"
echo "==========================================================="
echo "Step 10 revisit run: $RUN_NAME"
echo "Workspace: $SCRIPT_DIR"
echo "Output:    $RUN_DIR"
echo "Schedule:  $UPDATES PPO updates, LR 4e-4 -> 2e-4"
echo "Initialization: fresh revisit seed; no --init-from."
echo "==========================================================="

if [[ "${TRAIN_CMD:-}" == *"--init-from"* ]]; then
  echo "Refusing TRAIN_CMD with --init-from; Step 10 revisits start fresh." >&2
  exit 2
fi

# -------------------------------------------------------------------
# Phases 1-4: BC + critic warmup + PPO (fresh, no --init-from)
# -------------------------------------------------------------------
if [[ -n "${TRAIN_CMD:-}" ]]; then
  echo "Train command override: $TRAIN_CMD"
  eval "$TRAIN_CMD" 2>&1 | tee "$RUN_DIR/train.log"
else
  python train.py \
      --updates "$UPDATES" \
      --cons-updates "$CONS_UPDATES" \
      --critic-warmup "$WARMUP" \
      --eval-every "$EVAL_EVERY" \
      --save-every "$SAVE_EVERY" \
      --patience "$PATIENCE" \
      --run-name "$RUN_NAME" \
      2>&1 | tee "$RUN_DIR/train.log"
fi

# -------------------------------------------------------------------
# Eval + plots
# -------------------------------------------------------------------
if [[ -n "${EVAL_CMD:-}" ]]; then
  echo "Eval command override: $EVAL_CMD" | tee -a "$RUN_DIR/train.log"
  eval "$EVAL_CMD" 2>&1 | tee -a "$RUN_DIR/train.log"
elif [[ -f evaluate.py ]]; then
  echo "Eval command: python evaluate.py" | tee -a "$RUN_DIR/train.log"
  python evaluate.py 2>&1 | tee -a "$RUN_DIR/train.log" || true
fi

# -------------------------------------------------------------------
# Appendix + main.pdf rebuild
# -------------------------------------------------------------------
python revisit_appendix.py --run-name "$RUN_NAME" --run-dir "$RUN_DIR" || true
