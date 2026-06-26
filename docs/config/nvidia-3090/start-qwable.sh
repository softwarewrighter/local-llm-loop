#!/usr/bin/env bash
#
# Start llama-server for Qwable-v1 (IQ4_XS) on the RTX 3090 24 GB — the original
# baseline model, FULLY GPU-resident (the 18.9 GB IQ4_XS fits 24 GB whole, no
# expert offload needed, unlike the 12 GB 3060 profile in ../arch-nvidia-3060).
#
#   Endpoint : http://127.0.0.1:8080/v1
#   API key  : "local"   (must match opencode.json provider.options.apiKey)
#   Model id : qwable    (the --alias; opencode refs it as llamacpp/qwable)
#
# This is the QWABLE script. Its sibling start-coder-specdecode.sh in this same
# dir is the separate Qwen2.5-Coder spec-decode profile. Run ONE at a time — they
# both bind :8080 and both want most of the 24 GB. Point opencode's
# provider.model at whichever alias is currently served.
#
# --reasoning-budget 0: this Qwable build is reasoning-distilled; left unbounded
# it burns the whole --n-predict budget in a <think> stream before answering.
#
# Requires: a CUDA llama-server (system /usr/bin/llama-server works) and the GGUF
# present locally.
#
# Override any default via env var, e.g.:
#   CTX=65536 ./start-qwable.sh
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-/usr/bin/llama-server}"   # system CUDA build
# The Qwable GGUF lives in the HF cache on this box (snapshot symlink):
MODEL="${MODEL:-/disk1/hf-cache/hub/models--lordx64--Qwable-v1-GGUF/snapshots/f35ea1502056a2886dd88fb8a29272f8f3c9c3a5/Qwable-v1.IQ4_XS.gguf}"
ALIAS="${ALIAS:-qwable}"        # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"       # 0.0.0.0 only if a remote/Docker client needs it
PORT="${PORT:-8080}"            # MUST match opencode.json baseURL port
API_KEY="${API_KEY:-local}"     # MUST match opencode.json apiKey
GPU_LAYERS="${GPU_LAYERS:-99}"  # 99 = all layers on the 3090 (whole model fits 24 GB)
CTX="${CTX:-32768}"             # total context (matches opencode limit.context)
PARALLEL="${PARALLEL:-1}"       # concurrent slots
# ----------------------------------------------------------------------------

echo "Starting llama-server (RTX 3090 24 GB — Qwable-v1 IQ4_XS, full-GPU-resident)"
echo "  binary   : ${LLAMA_SERVER}"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo "  context  : ${CTX} total / ${PARALLEL} slot(s)"
echo

exec "${LLAMA_SERVER}" \
  -m "${MODEL}" \
  --alias "${ALIAS}" \
  --gpu-layers "${GPU_LAYERS}" \
  --reasoning-budget 0 \
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
