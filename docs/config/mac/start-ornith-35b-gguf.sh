#!/usr/bin/env bash
#
# Start llama-server for Ornith-1.0-35B (GGUF Q6_K) on macOS (Apple Silicon /
# Metal). This is the GGUF side of the MLX-vs-GGUF comparison — slower decode than
# MLX on this box (40.7 vs 55.6 t/s) but the standard llama.cpp path, directly
# comparable to every other GGUF row in performance-analysis.md.
#
#   Endpoint : http://127.0.0.1:8080/v1
#   Model id : ornith-35b-gguf   (opencode refs it as llamacpp/ornith-35b-gguf)
#
# Ornith is a Qwen-3.5-based MoE (~3B active) reasoning coder, vocab 248320,
# 256k context. Gate 1 (native tool_calls) passes. The 27.98 GiB Q6_K fits whole
# in the M1 Max's 64 GB unified memory.
#
# Requires: llama-server (Metal) on $PATH and the GGUF present locally
# (bartowski/deepreinforce-ai_Ornith-1.0-35B-GGUF, Q6_K).
set -euo pipefail

MODEL="${MODEL:-$HOME/tmp-hf/ornith-1.0-35b-Q6_K.gguf}"
ALIAS="${ALIAS:-ornith-35b-gguf}"   # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"
CTX="${CTX:-32768}"
PARALLEL="${PARALLEL:-1}"

echo "Starting llama-server (macOS / Metal — Ornith-1.0-35B Q6_K, GGUF)"
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
