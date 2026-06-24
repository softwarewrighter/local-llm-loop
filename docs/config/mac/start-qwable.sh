#!/usr/bin/env bash
#
# Start llama-server for Qwable-v1 (IQ4_XS) on macOS (Apple Silicon / Metal).
# Serves an OpenAI-compatible endpoint that opencode talks to.
#
#   Endpoint : http://127.0.0.1:8080/v1
#   API key  : "local"   (must match opencode.json provider.options.apiKey)
#   Model id : qwable    (the --alias; opencode refs it as llamacpp/qwable)
#
# Requires: llama-server (llama.cpp built with Metal — the default on Apple
# Silicon) on $PATH, and the GGUF model file present locally.
#
# Override any default via env var, e.g.:
#   PORT=9000 CTX=65536 ./start-qwable.sh
#   PARALLEL=2 ./start-qwable.sh           # add a 2nd slot for another client
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
MODEL="${MODEL:-$HOME/tmp-hf/Qwable-v1.IQ4_XS.gguf}"  # local GGUF path (edit me)
ALIAS="${ALIAS:-qwable}"        # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"       # 0.0.0.0 only if a remote/Docker client needs it
PORT="${PORT:-8080}"            # MUST match opencode.json baseURL port
API_KEY="${API_KEY:-local}"     # MUST match opencode.json apiKey
GPU_LAYERS="${GPU_LAYERS:-99}"  # 99 = offload all layers to Metal (unified memory)
CTX="${CTX:-131072}"            # total context, split across PARALLEL slots
PARALLEL="${PARALLEL:-1}"       # concurrent slots
# ----------------------------------------------------------------------------

echo "Starting llama-server (macOS / Metal)"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo "  context  : ${CTX} total / ${PARALLEL} slot(s) = $((CTX / PARALLEL)) per slot"
echo

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
