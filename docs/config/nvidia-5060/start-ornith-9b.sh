#!/usr/bin/env bash
#
# Start llama-server on the RTX 5060 Ti 16 GB with Ornith-1.0-9B (Qwen-3.5 dense),
# the small-card Ornith coder — the intended model for 12-16 GB boxes (see
# ../../plan-ornith-models.md). Fits whole on 16 GB with huge headroom; despite
# its size it scores SWE-bench Verified 69.4.
#
#   Endpoint : http://127.0.0.1:8080/v1
#   API key  : "local"          (must match opencode.json provider.options.apiKey)
#   Model id : ornith-9b         (opencode refs it as llamacpp/ornith-9b)
#
# WHY 9B HERE (not the 35B): the 9B-Dense fits 16 GB whole at Q8_0 (8.88 GiB).
# The 35B-MoE (Q6_K 28 GiB) does NOT fit 16 GB and would need heavy `--n-cpu-moe`
# offload — run the 35B on the 24 GB 3090 / 64 GB M1 Max instead.
#
# NO FP4 win here: the 9B is a K-quant, not MXFP4, so Blackwell's FP4 lane is
# idle — this is a plain bandwidth/quality run. Ornith is a reasoning model
# (<think> blocks); gate 1 (native tool_calls) passes via llama.cpp's Qwen parser.
#
# Download first (see download-models.sh; uses the uv hf CLI):
#   hf download bartowski/deepreinforce-ai_Ornith-1.0-9B-GGUF \
#     deepreinforce-ai_Ornith-1.0-9B-Q8_0.gguf --local-dir /disk1/models/ornith-9b
#
# Run ONE server at a time (all bind :8080). Point opencode at the alias.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/disk1/build/llama.cpp/build/bin/llama-server}"   # CUDA sm_120 build
MODEL="${MODEL:-/disk1/models/ornith-9b/deepreinforce-ai_Ornith-1.0-9B-Q8_0.gguf}"  # 8.88 GiB dense
ALIAS="${ALIAS:-ornith-9b}"     # MUST match the opencode.json model key (llamacpp/ornith-9b)
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"  # 99 = whole model on the GPU (fits 16 GB easily)
CTX="${CTX:-32768}"             # total context (matches opencode limit.context)
PARALLEL="${PARALLEL:-1}"

echo "Starting llama-server (RTX 5060 Ti 16 GB — Ornith-1.0-9B Q8_0, Qwen-3.5 dense)"
echo "  binary   : ${LLAMA_SERVER}"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo

exec "${LLAMA_SERVER}" \
  -m "${MODEL}" \
  --alias "${ALIAS}" \
  --gpu-layers "${GPU_LAYERS}" \
  --ctx-size "${CTX}" \
  --parallel "${PARALLEL}" \
  --cont-batching \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --host "${HOST}" \
  --port "${PORT}" \
  --n-predict 4096 \
  --api-key "${API_KEY}"
