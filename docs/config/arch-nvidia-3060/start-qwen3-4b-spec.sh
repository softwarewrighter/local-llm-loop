#!/usr/bin/env bash
#
# Start llama-server on the RTX 3060 12 GB with Qwen3-4B (dense) + a Qwen3-0.6B
# DRAFT for classic speculative decoding — the "draft+target" arm of the smaller-
# model experiments (see ../../plan-smaller-models-3060.md, experiment 3).
#
#   Endpoint : http://127.0.0.1:8081/v1
#   Model id : qwen3-4b   (opencode refs it as llamacpp/qwen3-4b)
#
# CLASSIC SPEC-DECODE (not MTP): a separate small model drafts, the target
# verifies. Qwen3 shares one 151936-token vocab across all sizes, so Qwen3-0.6B
# pairs EXACTLY with Qwen3-4B (high acceptance, no vocab gap). Engage with
# --model-draft + --spec-type draft-simple (draft-simple is REQUIRED; --model-draft
# alone engages nothing).
#
# Tests two things: (1) does a 4B dense *general* model cross our code floor
# (prior: no — Qwen3-8B emitted non-building code), and (2) classic draft-simple
# acceptance/speedup on Ampere for a dense target (where spec-decode *should* pay,
# unlike A3B MoEs). Both models resident, tiny footprint (~3 GB + ~0.5 GB).
#
# Qwen3 is a reasoning model (<think>); set REASONING_BUDGET=0 if it burns the
# n-predict budget thinking. A/B: SPEC_OFF=1 disables the draft.
# PORT 8081 (8080 = opencode UI Docker on this box).
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/data2/llm/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/data2/llm/models/qwen3-4b/Qwen3-4B-Q4_K_M.gguf}"
DRAFT="${DRAFT:-/data2/llm/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf}"
ALIAS="${ALIAS:-qwen3-4b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"
CTX="${CTX:-32768}"
PARALLEL="${PARALLEL:-1}"
SPEC_N="${SPEC_N:-4}"
REASONING_BUDGET="${REASONING_BUDGET:-}"

SPEC_ARGS=(--model-draft "${DRAFT}" --spec-type draft-simple --spec-draft-n-max "${SPEC_N}" --gpu-layers-draft 99)
if [ "${SPEC_OFF:-0}" = "1" ]; then SPEC_ARGS=(); fi
RB_ARGS=()
if [ -n "${REASONING_BUDGET}" ]; then RB_ARGS=(--reasoning-budget "${REASONING_BUDGET}"); fi

echo "Starting llama-server (RTX 3060 12 GB — Qwen3-4B + Qwen3-0.6B draft, classic spec-decode)"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  draft    : ${DRAFT}"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
if [ "${SPEC_OFF:-0}" = "1" ]; then echo "  spec     : OFF (A/B baseline)"; else echo "  spec     : draft-simple, n=${SPEC_N}"; fi
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
  --api-key "${API_KEY}" \
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
  ${RB_ARGS[@]+"${RB_ARGS[@]}"}
