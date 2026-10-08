# vLLM Serving for Qwen2.5-VL

Standalone vLLM environment for serving Qwen2.5-VL-32B-Instruct as an OpenAI-compatible endpoint.

## Setup

```bash
cd vllm
uv sync
```

## Serve the Model

```bash
# Default: port 8000, tensor-parallel 8
./serve.sh

# Custom configuration
PORT=8001 TP=4 ./serve.sh

# Or run directly with uv
uv run python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-32B-Instruct \
    --tensor-parallel-size 8 \
    --port 8000 \
    --trust-remote-code
```

## Test the Server

```bash
curl http://localhost:8000/v1/models
```

## Notes

- Qwen2.5-VL-32B fits in 1-2 A100-80GB GPUs; `serve.sh` defaults to `TP=8` to utilize a full node
- Adjust `TP` (tensor parallel size) based on your GPU configuration
- The model is easily swappable: edit `MODEL` in `serve.sh` or pass `--model` to the direct invocation
- The `--max-model-len` is set to 32768 by default; adjust if needed
