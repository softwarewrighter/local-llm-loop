#!/usr/bin/env bash
#
# Start llama-server on the RTX 5060 Ti 16 GB (Blackwell, sm_120) with Qwen3-8B
# (dense), optionally with a Qwen3-0.6B DRAFT for speculative decoding.
#
#   Endpoint : http://127.0.0.1:8080/v1
#   Model id : qwen3-8b   (opencode refs llamacpp/qwen3-8b)
#
# WHY: Qwen3 tool-calling is what llama.cpp parses natively (the family the 3090
# work blessed). At 8B it is the small, fast end — fits whole with enormous
# headroom (~5 GB). Two uses:
#   1. Loop test — but 8B is below the ~15B strict-JSON floor seen on the 3090
#      (Qwen3-14B already failed gate 2), so expect this to be a SPEED study more
#      than a loop-completer. Higher-precision (Q5/Q8) is the cheapest gate-2 lever.
#   2. Spec-decode A/B — Qwen3 shares ONE 151936-token vocab across all sizes, so
#      Qwen3-8B pairs EXACTLY with a Qwen3-0.6B draft. Set DRAFT to engage it.
#
# Speculative decoding REQUIRES --spec-type draft-simple (llama.cpp defaults to
# none; --model-draft alone loads the draft but engages nothing).
#
# Usage:
#   ./start-qwen3-8b.sh                              # solo
#   DRAFT=/disk1/models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf ./start-qwen3-8b.sh   # spec-decode
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-/disk1/build/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/disk1/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf}"  # ~5.0 GB dense
DRAFT="${DRAFT:-}"              # set to the Qwen3-0.6B GGUF to enable spec-decode
ALIAS="${ALIAS:-qwen3-8b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"
CTX="${CTX:-32768}"
PARALLEL="${PARALLEL:-1}"
# ----------------------------------------------------------------------------

SPEC_ARGS=()
if [ -n "${DRAFT}" ]; then
  SPEC_ARGS=(--model-draft "${DRAFT}" --spec-type draft-simple --gpu-layers-draft 99)
  echo "Starting Qwen3-8B + Qwen3-0.6B DRAFT (speculative decoding, draft-simple)"
else
  echo "Starting Qwen3-8B (solo, no draft)"
fi
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
[ -n "${DRAFT}" ] && echo "  draft    : ${DRAFT}"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo

exec "${LLAMA_SERVER}" \
  -m "${MODEL}" \
  --alias "${ALIAS}" \
  --gpu-layers "${GPU_LAYERS}" \
  "${SPEC_ARGS[@]}" \
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
