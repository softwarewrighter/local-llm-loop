#!/usr/bin/env bash
#
# Start llama-server on the RTX 5060 Ti 16 GB (Blackwell, sm_120) with
# gpt-oss-20b in its NATIVE MXFP4 quant — the FP4 showcase for this node.
#
#   Endpoint : http://127.0.0.1:8080/v1
#   API key  : "local"          (must match opencode.json provider.options.apiKey)
#   Model id : gptoss-20b        (the --alias; opencode refs llamacpp/gptoss-20b)
#
# WHY THIS MODEL ON THIS BOX (see ../../nvidia-5060-poc-summary.md):
#   - FP4: Blackwell (5th-gen tensor cores) runs MXFP4 *accelerated* rather than
#     upcasting it (as Ampere/the 3060 must). gpt-oss-20b ships native MXFP4, so
#     this is the natural place to measure whether FP4 acceleration is real.
#   - FITS: ~12.1 GB weights + q8_0 KV fit fully in 16 GB → -ngl 99, no offload.
#   - GATE RISK: on the 3090 gpt-oss-20b cleared tool-calling but FAILED the
#     clean-JSON gate (a literal control char in its envelope). The 5060 does not
#     change that; treat this primarily as the raw-speed / FP4 benchmark and
#     re-test the loop in case a server/template update fixed the JSON.
#
# Run ONE server at a time (they all bind :8080). Point opencode at the alias.
# Requires: a CUDA llama-server built for sm_120 (see ../../nvidia-5060-poc-summary.md),
#           opencode >= 1.17, and the GGUF present locally.
#
# Override any default via env var, e.g.:  CTX=16384 ./start-gptoss-20b.sh
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-/disk1/build/llama.cpp/build/bin/llama-server}"   # CUDA sm_120 build
MODEL="${MODEL:-/disk1/models/gpt-oss-20b/gpt-oss-20b-mxfp4.gguf}"  # ~12.1 GB native MXFP4
ALIAS="${ALIAS:-gptoss-20b}"    # MUST match the opencode.json model key (llamacpp/gptoss-20b)
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"  # 99 = whole model on the GPU (fits 16 GB)
CTX="${CTX:-32768}"             # total context (matches opencode limit.context)
PARALLEL="${PARALLEL:-1}"
# ----------------------------------------------------------------------------

echo "Starting llama-server (RTX 5060 Ti 16 GB — gpt-oss-20b MXFP4, FP4-accelerated, solo)"
echo "  binary   : ${LLAMA_SERVER}"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo "  context  : ${CTX} total / ${PARALLEL} slot(s)"
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
