#!/usr/bin/env bash
#
# Start llama-server on the RTX 3090 24 GB with the WINNING agentic-coder config
# from docs/nvidia-3090-poc-summary.md: Qwen3-Coder-30B-A3B-Instruct, MoE, run
# SOLO (no speculative-decoding draft), fully GPU-resident.
#
#   Endpoint : http://127.0.0.1:8080/v1
#   API key  : "local"      (must match opencode.json provider.options.apiKey)
#   Model id : qwen-coder   (the --alias; opencode refs it as llamacpp/qwen-coder)
#
# WHY THIS MODEL (see the POC summary's model-search matrix):
#   - TOOL CALLING: opencode is agentic — it only works if llama.cpp parses the
#     model's tool calls into native `tool_calls`. Qwen3-Coder does this correctly;
#     Qwen2.5-Coder (any quant) emits the wrong delimiter (<tools>) and breaks.
#   - STRICT JSON: the harness parses serde_json — needs valid JSON for plan/step/
#     review envelopes. Qwen3-Coder emits clean JSON; Qwen3-14B emits trailing
#     commas / bad escaping and fails.
#   - SPEED: as an MoE with only ~3.3B active params (30.5B total), decode is
#     ~156 t/s here — faster than any dense + speculative-decoding pair tried — so
#     NO draft is needed (spec-decode gives no net gain on A3B MoE). That is why
#     this is a solo run, not the start-coder-specdecode.sh draft-pair config.
#
# This is the QWEN3-CODER (solo MoE) script. Siblings in this dir:
#   start-qwable.sh            — the baseline Qwable model (full-GPU-resident)
#   start-coder-specdecode.sh  — dense target + draft, speculative decoding (slower)
# Run ONE at a time — they all bind :8080. Point opencode at the matching alias.
#
# Requires: a CUDA llama-server (system /usr/bin/llama-server works), the NVIDIA
# driver + CUDA runtime, opencode >= 1.17, and the GGUF present locally.
#
# Override any default via env var, e.g.:
#   CTX=16384 ./start-qwen3-coder.sh
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-/usr/bin/llama-server}"   # system CUDA build (b9728)
MODEL="${MODEL:-/disk1/models/qwen-coder/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}"  # ~18.6 GB MoE
ALIAS="${ALIAS:-qwen-coder}"    # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"       # 0.0.0.0 only if a remote/Docker client needs it
PORT="${PORT:-8080}"            # MUST match opencode.json baseURL port
API_KEY="${API_KEY:-local}"     # MUST match opencode.json apiKey
GPU_LAYERS="${GPU_LAYERS:-99}"  # 99 = whole MoE on the 3090 (fits 24 GB, ~20.8 GB used)
CTX="${CTX:-32768}"             # total context (matches opencode limit.context)
PARALLEL="${PARALLEL:-1}"       # concurrent slots
# ----------------------------------------------------------------------------

echo "Starting llama-server (RTX 3090 24 GB — Qwen3-Coder-30B-A3B-Instruct, MoE, solo)"
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
