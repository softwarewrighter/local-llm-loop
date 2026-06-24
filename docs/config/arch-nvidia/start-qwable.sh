#!/usr/bin/env bash
#
# Start llama-server for Qwable-v1 (IQ4_XS) on Arch Linux with an NVIDIA GPU
# (llama.cpp built with CUDA). Serves an OpenAI-compatible endpoint for opencode.
#
#   Endpoint : http://127.0.0.1:8080/v1
#   API key  : "local"   (must match opencode.json provider.options.apiKey)
#   Model id : qwable    (the --alias; opencode refs it as llamacpp/qwable)
#
# Requires: llama-server built with CUDA on $PATH (see docs/config/README.md),
# the NVIDIA driver + CUDA runtime, and the GGUF model file present locally.
#
# VRAM NOTE: unlike the Mac's unified memory, a single consumer GPU usually
# cannot hold the model PLUS a 128k-token KV cache. CTX defaults to 32768 here
# (matching opencode's limit.context). Raise it only if you have the VRAM;
# lower GPU_LAYERS to offload fewer layers if you run out.
#
# Override any default via env var, e.g.:
#   CTX=65536 ./start-qwable.sh
#   GPU_LAYERS=60 ./start-qwable.sh        # partial offload on a small GPU
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
MODEL="${MODEL:-$HOME/models/Qwable-v1.IQ4_XS.gguf}"  # local GGUF path (edit me)
ALIAS="${ALIAS:-qwable}"        # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"       # 0.0.0.0 only if a remote/Docker client needs it
PORT="${PORT:-8080}"            # MUST match opencode.json baseURL port
API_KEY="${API_KEY:-local}"     # MUST match opencode.json apiKey
GPU_LAYERS="${GPU_LAYERS:-99}"  # 99 = offload all layers to the GPU (VRAM permitting)
CTX="${CTX:-32768}"             # total context; bounded by VRAM, see note above
PARALLEL="${PARALLEL:-1}"       # concurrent slots
# ----------------------------------------------------------------------------

echo "Starting llama-server (Arch / NVIDIA CUDA)"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo "  context  : ${CTX} total / ${PARALLEL} slot(s) = $((CTX / PARALLEL)) per slot"
echo

# Flags are identical to the macOS script — only the build backend (CUDA vs
# Metal) and the CTX/VRAM budget differ. --gpu-layers, --flash-attn, and the
# q8_0 KV cache all work on CUDA.
exec llama-server \
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
