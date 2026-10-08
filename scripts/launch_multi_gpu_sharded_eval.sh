#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Multi-GPU Sharded Evaluation Launcher
# ──────────────────────────────────────────────────────────────────────────────
#
# This script launches sharded evaluation across multiple GPUs.
# Sharding is done by episode_id, so each process gets whole episodes.
#
# Usage:
#   # Trajectory selection on 4 GPUs:
#   GPUS=0,1,2,3 ./scripts/launch_multi_gpu_sharded_eval.sh data/processed dummy_argmax
#
#   # VLM-based trajectory selection with history frames:
#   GPUS=0,1 EXTRA_ARGS="--prompt-history-frames 3 --prompt-image-width 640" \
#     ./scripts/launch_multi_gpu_sharded_eval.sh data/processed gemini_genai
#
# Backward-compat:
#   If you pass a leading "task2" argument, it is ignored.
#
# Environment variables:
#   GPUS="0,1,2,3"   # GPU ids to use (comma-separated, default: "0")
#   SHARDS=4         # number of shards (default: number of GPUs, or 1)
#   NUM_WORKERS=4    # DataLoader workers per process (default: 4)
#   OUT_DIR=logs     # base output directory (default: logs)
#   EXTRA_ARGS="..." # extra args forwarded to the evaluation script
#
# ──────────────────────────────────────────────────────────────────────────────

ARG1="${1:-data/processed}"
ARG2="${2:-dummy_argmax}"
if [[ "${ARG1}" == "task2" ]]; then
  DATASET="${2:-data/processed}"
  MODEL_OR_STRAT="${3:-dummy_argmax}"
else
  DATASET="${ARG1}"
  MODEL_OR_STRAT="${ARG2}"
fi

GPUS="${GPUS:-0}"
OUT_DIR="${OUT_DIR:-logs}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

IFS=',' read -ra GPU_ARR <<< "${GPUS}"
GPU_COUNT="${#GPU_ARR[@]}"

# Ensure log directory exists
mkdir -p "${OUT_DIR}"

# This repo release only supports trajectory selection + closed-loop sim.

WORLD_SIZE="${SHARDS:-${GPU_COUNT}}"
[[ "${WORLD_SIZE}" -le 0 ]] && WORLD_SIZE=1

# Validate: for GPU-bound models, shards should match GPU count
if [[ "${MODEL_OR_STRAT}" == "gemini_genai" || "${MODEL_OR_STRAT}" == "openai_chat_completions" ]]; then
  # VLM models are network-bound, can have any shard count
  :
elif [[ "${WORLD_SIZE}" -ne "${GPU_COUNT}" ]]; then
  echo "[warn] SHARDS=${WORLD_SIZE} != GPU_COUNT=${GPU_COUNT}. For non-VLM models, consider matching them."
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="trajectory_selection_${MODEL_OR_STRAT}_${RUN_TAG}"

echo "[launch] trajectory_selection: model=${MODEL_OR_STRAT} gpus=${GPUS} shards=${WORLD_SIZE} out_dir=${OUT_DIR}/${EXP_NAME}"

for ((RANK=0; RANK < WORLD_SIZE; RANK++)); do
  # Assign GPU (cycle if more shards than GPUs)
  GPU_IDX=$((RANK % GPU_COUNT))
  GPU="${GPU_ARR[$GPU_IDX]}"

  export CUDA_VISIBLE_DEVICES="${GPU}"
  export WORLD_SIZE="${WORLD_SIZE}"
  export RANK="${RANK}"

  python scripts/run_trajectory_selection.py \
    --dataset "${DATASET}" \
    --model "${MODEL_OR_STRAT}" \
    --output-dir "${OUT_DIR}" \
    --exp-name "${EXP_NAME}" \
    --num-workers "${NUM_WORKERS}" \
    ${EXTRA_ARGS} \
    > "${OUT_DIR}/${EXP_NAME}_rank${RANK}.log" 2>&1 &
done

wait
echo "[done] Trajectory selection complete. Logs: ${OUT_DIR}/${EXP_NAME}_rank*.log"
echo ""
echo "To merge shard results, run:"
echo "  python -c \"from slow_brain_fast_planner.benchmarks.sharding import merge_shard_metrics; merge_shard_metrics('${OUT_DIR}/${EXP_NAME}')\""
