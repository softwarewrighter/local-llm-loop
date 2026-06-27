#!/usr/bin/env bash
#
# Start llama-server on the RTX 5060 Ti 16 GB (Blackwell, sm_120) with the
# proven agentic-coder winner from the 3090 work: Qwen3-Coder-30B-A3B-Instruct
# (MoE, ~3.3B active), run SOLO (no draft — spec-decode is moot on A3B).
#
#   Endpoint : http://127.0.0.1:8080/v1
#   Model id : qwen3-coder   (opencode refs llamacpp/qwen3-coder)
#
# WHY THIS MODEL (see ../../nvidia-3090-poc-summary.md):
#   It is the one model verified to clear BOTH gates — native tool_calls AND
#   strict JSON — and complete the loop with passing `cargo test`. It is the
#   single most likely model to produce a verified Rust CLI on this box.
#
# THE 16 GB CATCH: the Q4_K_M GGUF is ~18.6 GB — it does NOT fit 16 GB whole.
# Use MoE expert-offload to system RAM (this host has 251 GB): keep attention +
# a few expert layers on the GPU, push the rest to RAM with --n-cpu-moe N.
# Decode survives (only ~3.3B active params are read per token); prefill takes
# the hit (RAM-resident experts compute on the CPU) — same regime as the 3060
# POC. MEASURED on this box: --n-cpu-moe 16 -> 14.2 GB VRAM (1.7 GB free),
# ~55 tok/s. Lower N for more speed if you have headroom; raise it if it OOMs.
#
# THE TOOL-CALL CATCH (without this the agentic loop silently does nothing):
# Qwen3-Coder uses a NON-STANDARD tool-call syntax — <function=name>
# <parameter=x>..</parameter></function> XML, not <tool_call>{json}. llama.cpp's
# generic "peg-native" parser only half-parses it from the GGUF's embedded
# template, so tool calls LEAK into assistant text and opencode never executes
# them (finish_reason is never `tool_calls`). Starting with llama.cpp's dedicated
# Qwen3-Coder.jinja template (--chat-template-file below) fixes parsing. VERIFIED:
# with it, opencode completed the spec->code->`cargo test` loop (tests passed).
#
# Override any default via env var, e.g.:  N_CPU_MOE=16 ./start-qwen3-coder.sh
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-/disk1/build/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/disk1/models/qwen3-coder-30b/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}"  # ~18.6 GB MoE
ALIAS="${ALIAS:-qwen3-coder}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"
N_CPU_MOE="${N_CPU_MOE:-16}"    # expert layers (of 48) whose FFNs live in RAM. MEASURED: 16 -> 14.2 GB VRAM / ~55 tok/s
CTX="${CTX:-32768}"
PARALLEL="${PARALLEL:-1}"
# REQUIRED for tool-calling — see "THE TOOL-CALL CATCH" above. Empty TEMPLATE= falls back to the embedded template (tools will leak).
TEMPLATE="${TEMPLATE:-/disk1/build/llama.cpp/models/templates/Qwen3-Coder.jinja}"
# ----------------------------------------------------------------------------

echo "Starting llama-server (RTX 5060 Ti 16 GB — Qwen3-Coder-30B-A3B MoE, solo, --n-cpu-moe ${N_CPU_MOE})"
echo "  binary   : ${LLAMA_SERVER}"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo "  context  : ${CTX} total / ${PARALLEL} slot(s) / --n-cpu-moe ${N_CPU_MOE}"
echo

exec "${LLAMA_SERVER}" \
  -m "${MODEL}" \
  --alias "${ALIAS}" \
  --gpu-layers "${GPU_LAYERS}" \
  --n-cpu-moe "${N_CPU_MOE}" \
  --ctx-size "${CTX}" \
  --parallel "${PARALLEL}" \
  --cont-batching \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  ${TEMPLATE:+--chat-template-file "${TEMPLATE}"} \
  --host "${HOST}" \
  --port "${PORT}" \
  --n-predict 4096 \
  --api-key "${API_KEY}"
