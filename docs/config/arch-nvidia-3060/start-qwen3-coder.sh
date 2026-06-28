#!/usr/bin/env bash
#
# Start llama-server on the RTX 3060 12 GB (Ampere, sm_86) with the verified
# agentic-coder reference: Qwen3-Coder-30B-A3B-Instruct (MoE, ~3.3B active), run
# SOLO via MoE expert-offload to system RAM. The reliable workhorse for this box
# (see ../../plan-rtx3060-12.md).
#
#   Endpoint : http://127.0.0.1:8081/v1
#   API key  : "local"        (must match opencode.json provider.options.apiKey)
#   Model id : qwen3-coder     (opencode refs it as llamacpp/qwen3-coder)
#
# WHY THIS MODEL (see ../../nvidia-3090-poc-summary.md / ../../nvidia-5060-poc-summary.md):
#   The one model verified to clear BOTH gates (native tool_calls AND strict JSON)
#   and complete the loop with passing `cargo test` on multiple boxes. No MTP head
#   (A3B spec-decode is moot), but the highest-confidence coder here.
#
# FIT ON 12 GB: the Q4_K_M GGUF is ~17.3 GB > 12 GB — NOT resident. Push the MoE
# experts of the first N layers to the 503 GB system RAM (--n-cpu-moe N); decode
# survives (~3.3B active), prefill takes the hit. The 5060 (16 GB) served it at
# N=16 -> ~55 t/s; with only 12 GB here START HIGHER (N=24) and tune toward
# ~11 GB VRAM. Smaller N = faster, until 12 GB OOMs.
#
# THE TOOL-CALL CATCH (without this the agentic loop silently does nothing):
# Qwen3-Coder uses NON-STANDARD tool-call syntax (<function=name><parameter=x>..
# </parameter></function> XML, not <tool_call>{json}). llama.cpp's generic parser
# only half-parses it from the embedded template, so tool calls LEAK into
# assistant text and opencode never executes them (finish_reason never
# `tool_calls`). Starting with llama.cpp's dedicated Qwen3-Coder.jinja template
# (--chat-template-file below) fixes parsing. This is a GATE-1 PREREQUISITE.
#
# PORT 8081, not 8080: 8080 is the opencode UI Docker stack here.
# Override any default via env var, e.g.:  NCPUMOE=28 ./start-qwen3-coder.sh
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-/mnt/storage1/src/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/mnt/storage1/qwen3-coder/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}"  # ~17.3 GB MoE
ALIAS="${ALIAS:-qwen3-coder}"    # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"             # 8080 = opencode UI Docker on this box; server uses 8081
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"   # 99 = all layers nominally to GPU; experts pulled back below
NCPUMOE="${NCPUMOE:-24}"         # experts of first N (of 48) layers -> RAM; tune to ~11 GB VRAM
CTX="${CTX:-32768}"              # total context (matches opencode limit.context)
PARALLEL="${PARALLEL:-1}"
THREADS="${THREADS:-36}"         # physical cores (E5-2697 v4: 2x18c) — CPU experts decode here
# REQUIRED for tool-calling — see "THE TOOL-CALL CATCH". Empty TEMPLATE= falls back to the embedded template (tools leak).
TEMPLATE="${TEMPLATE:-/mnt/storage1/src/llama.cpp/models/templates/Qwen3-Coder.jinja}"
# ----------------------------------------------------------------------------

echo "Starting llama-server (Arch / NVIDIA CUDA — RTX 3060 12 GB — Qwen3-Coder-30B-A3B MoE, --n-cpu-moe ${NCPUMOE})"
echo "  binary   : ${LLAMA_SERVER}"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo "  context  : ${CTX} total / ${PARALLEL} slot(s)"
echo "  offload  : -ngl ${GPU_LAYERS} --n-cpu-moe ${NCPUMOE}  (experts of first ${NCPUMOE}/48 layers -> RAM; -t ${THREADS})"
echo

# NUMA note (dual-socket box): CPU-resident experts make decode RAM-bandwidth-bound.
# If decode is slow:  numactl --interleave=all ./start-qwen3-coder.sh

exec "${LLAMA_SERVER}" \
  -m "${MODEL}" \
  --alias "${ALIAS}" \
  --gpu-layers "${GPU_LAYERS}" \
  --n-cpu-moe "${NCPUMOE}" \
  --ctx-size "${CTX}" \
  --parallel "${PARALLEL}" \
  --threads "${THREADS}" \
  --cont-batching \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  ${TEMPLATE:+--chat-template-file "${TEMPLATE}"} \
  --host "${HOST}" \
  --port "${PORT}" \
  --n-predict 4096 \
  --api-key "${API_KEY}"
