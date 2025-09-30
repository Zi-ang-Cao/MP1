#!/usr/bin/env bash
# -------------------------------------------------------------------------
# run_train.sh  GPU_ID CONFIG  [TASKS]  [SEEDS]
#
# - GPU_ID  : necessary
# - CONFIG  : necessary
# - TASKS   : option
# - SEEDS   : option 0,1,2
# -------------------------------------------------------------------------

set -euo pipefail      #

show_usage() {
  cat <<EOF
use:
  $0 <GPU_ID> <CONFIG> [TASKS] [SEEDS]

example:
  $0 0 baseline metaworld_drawer-close 42

  # explain:
  #   - run a task on GPU 0 
  #   - config file is baseline
  #   - random seed is 42
EOF
}

# ----------- parameters checking-----------------------------------------------------
if (( $# < 2 )); then
  echo "❌ parameters are not enough!"
  show_usage
  exit 1
fi

GPU_ID=$1
CONFIG=$2
TASKS_ARG=${3:-}
SEEDS_ARG=${4:-}

# ----------- default tasks and seeds ---------------------------------------------
TASKS_DEFAULT=(
  # metaworld_button-press
  # metaworld_button-press-wall
  # metaworld_dial-turn
  # metaworld_door-close
  metaworld_drawer-close

)
SEEDS_DEFAULT=(0 1 2)

# ---------- tasks ------------------------------------------------------
if [[ -n $TASKS_ARG ]]; then
  IFS=',' read -ra TASKS <<< "$TASKS_ARG"   #
else
  TASKS=("${TASKS_DEFAULT[@]}")             #
fi

# ---------- seeds ------------------------------------------------------
if [[ -n $SEEDS_ARG ]]; then
  IFS=',' read -ra SEEDS <<< "$SEEDS_ARG"
else
  SEEDS=("${SEEDS_DEFAULT[@]}")             #
fi

# ----------- train loop -----------------------------------------------------
for seed in "${SEEDS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "▶ GPU=$GPU_ID | config=$CONFIG | task=$task | seed=$seed"
    bash scripts/train_policy.sh "$CONFIG" "$task" 0000 "$seed" "$GPU_ID"
  done
done
