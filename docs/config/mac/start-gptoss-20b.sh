#!/usr/bin/env bash
#
# Start llama-server for gpt-oss-20b (native MXFP4) on macOS (Apple Silicon /
# Metal). This is the cross-platform comparison point: gpt-oss-20b is the 5060
# Ti's speed champ (~138 t/s, FP4-ACCELERATED on Blackwell). On the M1 Max there
# is NO native FP4 path (Metal upcasts MXFP4 to FP16 on the shader ALUs), so this
# run measures what the same model does WITHOUT FP4 acceleration — decode is then
# a pure bandwidth story on the ~3.6B active params, and it fits whole in the
# 64 GB unified memory.
#
#   Endpoint : http://127.0.0.1:8080/v1
#   API key  : "local"        (must match opencode.json provider.options.apiKey)
#   Model id : gptoss-20b      (the --alias; opencode refs it as llamacpp/gptoss-20b)
#
# Override via env, e.g.:  CTX=16384 ./start-gptoss-20b.sh
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
MODEL="${MODEL:-$HOME/tmp-hf/gpt-oss-20b-mxfp4.gguf}"
ALIAS="${ALIAS:-gptoss-20b}"    # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"  # whole model on Metal (unified memory)
CTX="${CTX:-32768}"             # matches opencode limit.context
PARALLEL="${PARALLEL:-1}"
# ----------------------------------------------------------------------------

echo "Starting llama-server (macOS / Metal — gpt-oss-20b MXFP4, no FP4 accel here)"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo "  context  : ${CTX} total / ${PARALLEL} slot(s)"
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
