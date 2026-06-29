#!/usr/bin/env bash
#
# Start llama-server on the RTX 5060 Ti 16 GB (Blackwell, sm_120) with
# Ornith-1.0-9B PLUS a baked-in MTP next-token head — RESIDENT dense coder with
# SELF-speculative decoding (no draft model, no MoE offload). This is the
# **MTP variant** of start-ornith-9b.sh: prefer it when the head loads (free
# decode uplift), per the project's "prefer MTP when available" rule.
#
#   Endpoint : http://127.0.0.1:8080/v1
#   API key  : "local"            (must match opencode.json provider.options.apiKey)
#   Model id : ornith-9b-mtp       (opencode refs it as llamacpp/ornith-9b-mtp)
#
# WHY MTP HERE: spec-decode rewards DENSE models (every token reads all weights →
# lots to amortize), so MTP pays off on this dense 9B — unlike the A3B MoEs where
# only ~3B params are active per token. Ornith's custom 248320 vocab also rules
# out an external Qwen3 draft, so the bundled MTP head is the ONLY spec route.
# Plain (no-MTP) Ornith-9B did the 5060 loop in 4m42s; this should beat it. The
# 3060 measured the MTP head at ~1.3–1.7× decode (4m29s loop, Q6_K).
#
# THE MTP GGUF: protoLabsAI/Ornith-1.0-9B-MTP-GGUF bakes a KL-distilled next-token
# head onto the 9B trunk for llama.cpp's `--spec-type draft-mtp`, distribution-
# lossless (drafts are verified against the target). Needs llama.cpp ≥ b9616 with
# Qwen3.5 arch (this box's sm_120 build qualifies).
#
#   hf download protoLabsAI/Ornith-1.0-9B-MTP-GGUF <file>.gguf \
#     --local-dir /disk1/models/ornith-9b-mtp
#
# FIT ON 16 GB: dense, RESIDENT, big headroom (Q8_0 ~9.5 GB + the head + q8_0 KV).
# No --n-cpu-moe (it is dense, not MoE).
#
# A/B the MTP uplift:  SPEC_OFF=1 ./start-ornith-9b-mtp.sh   (same model, MTP off).
# Ornith is a reasoning model (<think>); set REASONING_BUDGET=0 if it spends the
# whole --n-predict budget thinking.
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-/disk1/build/llama.cpp/build/bin/llama-server}"   # CUDA sm_120 build
MODEL="${MODEL:-/disk1/models/ornith-9b-mtp/Ornith-1.0-9B-MTP-Q8_0.gguf}"  # dense + MTP head, resident
ALIAS="${ALIAS:-ornith-9b-mtp}"  # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"   # 99 = whole model on the GPU (fits 16 GB easily)
CTX="${CTX:-32768}"
PARALLEL="${PARALLEL:-1}"
SPEC_N="${SPEC_N:-3}"            # speculative tokens per step (3 = max throughput, 2 = max accept)
REASONING_BUDGET="${REASONING_BUDGET:-}"  # set 0 if <think> eats the n-predict budget
# ----------------------------------------------------------------------------

# BUNDLED MTP: the head is baked into the GGUF, so engage it with --spec-type
# draft-mtp ALONE. Do NOT pass --spec-draft-model <same file> (loads a 2nd copy).
SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max "${SPEC_N}")
if [ "${SPEC_OFF:-0}" = "1" ]; then SPEC_ARGS=(); fi   # A/B: serve the same model with MTP off

RB_ARGS=()
if [ -n "${REASONING_BUDGET}" ]; then RB_ARGS=(--reasoning-budget "${REASONING_BUDGET}"); fi

echo "Starting llama-server (RTX 5060 Ti 16 GB — Ornith-1.0-9B + MTP, dense, resident, self-spec)"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
if [ "${SPEC_OFF:-0}" = "1" ]; then echo "  MTP      : OFF (A/B baseline)"; else echo "  MTP      : on (draft-mtp, n=${SPEC_N})"; fi
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
