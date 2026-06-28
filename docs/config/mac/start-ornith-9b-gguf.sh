#!/usr/bin/env bash
#
# Start llama-server for Ornith-1.0-9B (GGUF Q8_0) on macOS (Apple Silicon /
# Metal). The 9B-Dense (Qwen 3.5) is the **small-card** Ornith — the intended
# model for the RTX 3060 12 GB / RTX 5060 Ti 16 GB; benched here on the M1 Max as
# the stand-in. Despite its size it scores SWE-bench Verified 69.4 (see
# plan-ornith-models.md).
#
#   Endpoint : http://127.0.0.1:8080/v1
#   Model id : ornith-9b-gguf   (opencode refs it as llamacpp/ornith-9b-gguf)
#
# Dense 9B → all params active per token, so decode is slower than the A3B MoEs
# (~33 t/s Q8_0 here). vocab 248320, reasoning <think> blocks. 8.88 GiB fits any
# of the four target boxes whole.
set -euo pipefail

MODEL="${MODEL:-$HOME/tmp-hf/ornith-1.0-9b-Q8_0.gguf}"
ALIAS="${ALIAS:-ornith-9b-gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"
CTX="${CTX:-32768}"
PARALLEL="${PARALLEL:-1}"

echo "Starting llama-server (macOS / Metal — Ornith-1.0-9B Q8_0, GGUF)"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
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
