#!/usr/bin/env bash
#
# Start llama-server on the RTX 3060 12 GB (Ampere, sm_86) with Ornith-1.0-9B
# (Qwen-3.5 dense) — the small-card agentic coder and the ANCHOR model for this
# box (see ../../plan-rtx3060-12.md and ../../plan-ornith-models.md).
#
#   Endpoint : http://127.0.0.1:8081/v1
#   API key  : "local"          (must match opencode.json provider.options.apiKey)
#   Model id : ornith-9b-gguf    (opencode refs it as llamacpp/ornith-9b-gguf)
#
# WHY THIS MODEL: Ornith-1.0-9B is the only sub-14B model that reliably codes on
# this harness — every other general model <=14B fails a gate or emits
# non-building code (see ../../nvidia-5060-poc-summary.md). RL-trained for agentic
# coding (SWE-bench Verified 69.4); on the 5060 it ran the full loop in 4m42s with
# 0 retries.
#
# FIT ON 12 GB: this is the comfortable one. Q6_K is 7.36 GiB + ~1.5 GB q8_0 KV
# ~= 9 GB, so it is GPU-RESIDENT with headroom — no --n-cpu-moe offload needed
# (it is dense, not MoE). Q8_0 (9.53 GiB) also fits but is tight with ctx; use
# Q6_K by default and bump to Q8_0 only if code quality needs it.
#
# NO MTP HERE: the 9B is dense (all params active per token), so it has no
# self-spec MTP head and no faster same-vocab draft (custom 248320 vocab). It
# runs solo — MTP is the angle for the offloaded 35B-A3B MoE (start-qwen36-mtp.sh).
#
# Ornith is a reasoning model (<think> blocks). It ran fine WITHOUT a reasoning
# budget on the 5060; if it spends the whole --n-predict budget thinking before
# answering, add --reasoning-budget 0 (REASONING_BUDGET=0 ./start-ornith-9b.sh).
#
# Requires a CUDA (sm_86) llama-server build and the GGUF present locally.
# Override any default via env var, e.g.:  PORT=8080 ./start-ornith-9b.sh
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-/mnt/storage1/src/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/mnt/storage1/ornith-9b/Ornith-1.0-9B-Q6_K.gguf}"  # 7.36 GiB dense, resident
ALIAS="${ALIAS:-ornith-9b-gguf}"  # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"              # 8080 = opencode UI Docker on this box; server uses 8081
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"   # 99 = whole model on the GPU (fits 12 GB easily)
CTX="${CTX:-32768}"              # total context (matches opencode limit.context)
PARALLEL="${PARALLEL:-1}"
REASONING_BUDGET="${REASONING_BUDGET:-}"  # set 0 if <think> eats the n-predict budget
# ----------------------------------------------------------------------------

RB_ARGS=()
if [ -n "${REASONING_BUDGET}" ]; then RB_ARGS=(--reasoning-budget "${REASONING_BUDGET}"); fi

echo "Starting llama-server (Arch / NVIDIA CUDA — RTX 3060 12 GB — Ornith-1.0-9B Q6_K, dense, resident)"
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
  --api-key "${API_KEY}" \
  ${RB_ARGS[@]+"${RB_ARGS[@]}"}
