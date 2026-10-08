#!/usr/bin/env bash
set -euo pipefail

# Launch N independent shard processes on a single machine.
#
# Required env:
#   DATASET: dataset root
#
# Optional env:
#   OUT_DIR:  output root (default: logs)
#   EXP_NAME: experiment name (default: trajectory_selection)
#   SHARDS:   number of shards/processes (default: 1)
#   RUN_DIR:  explicit run directory (default: OUT_DIR/EXP_NAME/EXP_NAME_<timestamp>)
#   MERGE:    if "1", merge shards into RUN_DIR/merged (default: 0)
#
# Extra CLI args:
#   Pass any extra args after `--` and they will be forwarded to the python CLI.
#
# Example:
#   DATASET=data/processed SHARDS=8 OUT_DIR=logs EXP_NAME=trajsel_vlm \
#     scripts/trajectory_selection/run_eval_sharded_local.sh -- --model gemini_genai --num-workers 8

DATASET="${DATASET:-}"
if [[ -z "${DATASET}" ]]; then
  echo "Missing DATASET. Example: DATASET=data/processed SHARDS=8 $0 -- --model dummy_argmax"
  exit 2
fi

OUT_DIR="${OUT_DIR:-logs}"
EXP_NAME="${EXP_NAME:-trajectory_selection}"
SHARDS="${SHARDS:-1}"
MERGE="${MERGE:-0}"

ts="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${OUT_DIR}/${EXP_NAME}/${EXP_NAME}_${ts}}"

mkdir -p "${RUN_DIR}/shards"
mkdir -p "${RUN_DIR}/logs"

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  EXTRA_ARGS=("$@")
fi

echo "[launch] dataset=${DATASET} shards=${SHARDS} run_dir=${RUN_DIR}"

pids=()
for ((RANK=0; RANK < SHARDS; RANK++)); do
  echo "[launch] rank=${RANK}/${SHARDS}"
  shard_out="${RUN_DIR}/shards/shard${RANK}"
  WORLD_SIZE="${SHARDS}" RANK="${RANK}" \
    python -m slow_brain_fast_planner.cli.trajectory_selection \
      --dataset "${DATASET}" \
      --out "${shard_out}" \
      --num-shards "${SHARDS}" \
      --shard-id "${RANK}" \
      "${EXTRA_ARGS[@]}" \
      > "${RUN_DIR}/logs/shard${RANK}.log" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "[done] some shards failed; see ${RUN_DIR}/logs/shard*.log"
  exit 1
fi

echo "[done] all shards complete; logs: ${RUN_DIR}/logs/shard*.log"

if [[ "${MERGE}" == "1" ]]; then
  echo "[merge] merging shards into ${RUN_DIR}/merged"
  python scripts/trajectory_selection/merge_shards.py "${RUN_DIR}" --out "${RUN_DIR}/merged" --overwrite --write-report
  echo "[merge] done: ${RUN_DIR}/merged"
fi

