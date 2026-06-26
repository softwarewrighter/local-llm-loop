#!/usr/bin/env bash
#
# Start llama-server on the RTX 3090 24 GB with Gemma-4-31B-it (DENSE) + a tiny
# Gemma-4-E2B draft — the one verified DENSE speculative-decoding coding pair for
# this harness (see docs/nvidia-3090-poc-summary.md). All Gemma 4 share one
# 262144-token vocab, so target and draft match exactly.
#
#   Endpoint : http://127.0.0.1:8080/v1
#   API key  : "local"      (must match opencode.json provider.options.apiKey)
#   Model id : gemma4-31b    (the --alias; opencode refs it as llamacpp/gemma4-31b)
#
# NOTES:
#   - DENSE 31B → ~36 t/s solo; the E2B draft lifts decode to ~46 t/s (~1.3×).
#     Still far slower than the MoE coders — use when you specifically want a dense
#     model. `--spec-type draft-simple` is REQUIRED to enable the draft (llama.cpp
#     b9728 defaults to none).
#   - CTX=16384 (not 32768) so the 31B weights + E2B draft + KV fit 24 GB. Keep
#     opencode.json's limit.context for gemma4-31b at 16384 to match.
#
# One server binds :8080 at a time — point opencode at the matching alias.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/usr/bin/llama-server}"
MODEL="${MODEL:-/disk1/models/gemma4-31b/gemma-4-31B-it-Q4_K_M.gguf}"        # ~17.4 GB dense
DRAFT="${DRAFT:-/disk1/models/gemma4-31b/gemma-4-E2B-it-Q4_K_M.gguf}"        # ~2.9 GB draft (exact vocab)
ALIAS="${ALIAS:-gemma4-31b}"    # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"
GPU_LAYERS_DRAFT="${GPU_LAYERS_DRAFT:-99}"
NMAX="${NMAX:-16}"; NMIN="${NMIN:-4}"; PMIN="${PMIN:-0.6}"
CTX="${CTX:-16384}"             # reduced so target+draft fit 24 GB
PARALLEL="${PARALLEL:-1}"

echo "Starting llama-server (RTX 3090 24 GB — Gemma-4-31B-it + E2B draft, dense spec-decode)"
echo "  target   : ${MODEL}  (alias: ${ALIAS})"
echo "  draft    : ${DRAFT}"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo "  context  : ${CTX} / ${PARALLEL} slot(s)   spec: n-max ${NMAX}/n-min ${NMIN}/p-min ${PMIN}"
echo

exec "${LLAMA_SERVER}" \
  -m "${MODEL}" \
  --model-draft "${DRAFT}" \
  --spec-type draft-simple \
  --alias "${ALIAS}" \
  --gpu-layers "${GPU_LAYERS}" \
  --gpu-layers-draft "${GPU_LAYERS_DRAFT}" \
  --spec-draft-n-max "${NMAX}" \
  --spec-draft-n-min "${NMIN}" \
  --spec-draft-p-min "${PMIN}" \
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
