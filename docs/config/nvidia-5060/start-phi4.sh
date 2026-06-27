#!/usr/bin/env bash
#
# Start llama-server on the RTX 5060 Ti 16 GB (Blackwell, sm_120) with
# Phi-4-14B (dense), run fully GPU-resident.
#
#   Endpoint : http://127.0.0.1:8080/v1
#   Model id : phi-4   (opencode refs llamacpp/phi-4)
#
# WHY: dense 14B at Q4_K_M is ~8.9 GB — fits 16 GB with lots of headroom (room
# for a big KV cache). Microsoft tuned it for structured output / function
# calling, so it is a strong "real coder that fits whole" candidate. Untested in
# this harness → verify BOTH gates (native tool_calls + strict JSON); 14B is the
# size band where the strict-JSON gate starts to bite, so this is a real test of
# whether a sub-15B model can complete the loop on clean JSON.
#
# Override any default via env var.
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-/disk1/build/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/disk1/models/phi-4/phi-4-Q4_K_M.gguf}"  # ~8.9 GB dense
ALIAS="${ALIAS:-phi-4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"
CTX="${CTX:-32768}"
PARALLEL="${PARALLEL:-1}"
# ----------------------------------------------------------------------------

echo "Starting llama-server (RTX 5060 Ti 16 GB — Phi-4-14B dense, fully resident)"
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
