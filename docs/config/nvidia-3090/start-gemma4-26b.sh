#!/usr/bin/env bash
#
# Start llama-server on the RTX 3090 24 GB with Gemma-4-26B-A4B-it — a verified
# agentic-coder for this harness (see docs/nvidia-3090-poc-summary.md). MoE,
# ~3.8B active, run SOLO, fully GPU-resident. Produced the highest-quality greeter
# (`Hello, {name}!`) of all models tested.
#
#   Endpoint : http://127.0.0.1:8080/v1
#   API key  : "local"      (must match opencode.json provider.options.apiKey)
#   Model id : gemma4-26b    (the --alias; opencode refs it as llamacpp/gemma4-26b)
#
# Gemma 4 ships native function-calling (dedicated tool-call tokens) + structured
# JSON output, which is exactly what the agentic loop needs. As an A4B MoE it runs
# solo (a draft gives no net gain on a low-active-param MoE).
#
# One server binds :8080 at a time — point opencode at the matching alias.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/usr/bin/llama-server}"
MODEL="${MODEL:-/disk1/models/gemma4-26b/gemma-4-26B-A4B-it-Q4_K_M.gguf}"   # ~15.6 GB MoE
ALIAS="${ALIAS:-gemma4-26b}"    # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"
CTX="${CTX:-32768}"
PARALLEL="${PARALLEL:-1}"

echo "Starting llama-server (RTX 3090 24 GB — Gemma-4-26B-A4B-it, MoE, solo)"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo "  context  : ${CTX} / ${PARALLEL} slot(s)"
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
