#!/bin/bash
# Serve Qwen2.5-VL-32B with vLLM
# Usage: ./serve.sh [--port PORT] [--tensor-parallel-size TP]

set -e

PORT=${PORT:-8000}
TP=${TP:-8}  # Adjust based on your GPU count (32B can fit in 1-2 GPUs, but use 8 to utilize all)

MODEL="Qwen/Qwen2.5-VL-32B-Instruct"

echo "Starting vLLM server for $MODEL"
echo "Port: $PORT, Tensor Parallel: $TP"

uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --port "$PORT" \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    "$@"
