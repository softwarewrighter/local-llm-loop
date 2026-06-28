#!/usr/bin/env bash
#
# Start llama-server on the RTX 3060 12 GB (Ampere, sm_86) with
# Qwen3.6-35B-A3B-MTP (Qwen3.5-style MoE, ~3B active) — the MTP candidate for
# this box (see ../../plan-rtx3060-12.md). The built-in MTP next-token head drives
# SELF-speculative decoding: no separate draft model.
#
#   Endpoint : http://127.0.0.1:8081/v1
#   API key  : "local"        (must match opencode.json provider.options.apiKey)
#   Model id : qwen36-mtp      (opencode refs it as llamacpp/qwen36-mtp)
#
# WHY MTP HERE (and only via an offloaded MoE): on 12 GB the dense 9B has no MTP
# fit gain and no faster draft, so MTP only pays off on an A3B MoE whose experts
# spill to RAM. Decode rides only the ~3B active params (survives offload) and the
# MTP head claws back what the spill costs (~85% accept, ~1.24x on the M1 Max).
# This is the same offloaded-MoE regime already PROVEN to complete the loop on
# THIS box (Qwable, start-qwable.sh, --n-cpu-moe 18, ~49 t/s) — here with a real
# coder + the MTP head.
#
# FIT ON 12 GB: the Q4_K_M GGUF is ~21 GB > 12 GB, so it CANNOT be resident.
# Keep all layers nominally on the GPU (-ngl 99) but push the MoE experts of the
# first N layers back to the 503 GB system RAM (--n-cpu-moe N). Qwable's IQ4_XS
# (17.6 GB) sat at ~11.2 GB VRAM with N=18; this Q4_K_M is bigger, so START
# HIGHER (N=24) and tune toward ~11 GB VRAM. The MTP head adds a little VRAM, so
# leave margin. Smaller N = faster, until 12 GB OOMs.
#
# MTP flags: --spec-type draft-mtp engages the embedded next-token head; the main
# GGUF is also the draft (--spec-draft-model = same file), so no second model is
# loaded. A/B the uplift with SPEC_OFF=1 ./start-qwen36-mtp.sh (MTP off).
#
# Qwen3.6 is a reasoning model (<think>). If it spends the whole --n-predict
# budget thinking before answering, add --reasoning-budget 0
# (REASONING_BUDGET=0 ./start-qwen36-mtp.sh) — as Qwable needed on this box.
#
# PORT 8081, not 8080: 8080 is the opencode UI Docker stack here.
# Override any default via env var, e.g.:  NCPUMOE=28 ./start-qwen36-mtp.sh
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-/mnt/storage1/src/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/mnt/storage1/qwen36-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"  # ~21 GB MoE, offloaded
ALIAS="${ALIAS:-qwen36-mtp}"     # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"             # 8080 = opencode UI Docker on this box; server uses 8081
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"   # 99 = all layers nominally to GPU; experts pulled back below
NCPUMOE="${NCPUMOE:-24}"         # experts of first N (of 40) layers -> RAM; tune to ~11 GB VRAM
CTX="${CTX:-32768}"              # total context (matches opencode limit.context)
PARALLEL="${PARALLEL:-1}"
THREADS="${THREADS:-36}"         # physical cores (E5-2697 v4: 2x18c) — CPU experts decode here
SPEC_N="${SPEC_N:-4}"            # speculative tokens per step
REASONING_BUDGET="${REASONING_BUDGET:-}"  # set 0 if <think> eats the n-predict budget
# ----------------------------------------------------------------------------

# BUNDLED MTP: the next-token head ships inside the Qwen3.6 GGUF, so engage it
# with --spec-type draft-mtp ALONE. Do NOT pass --spec-draft-model <same file> —
# that loads a SECOND full copy of the model as the draft (OOMs the GPU; pointless
# RAM blowup under offload).
SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max "${SPEC_N}")
if [ "${SPEC_OFF:-0}" = "1" ]; then SPEC_ARGS=(); fi   # A/B: serve the same model with MTP off

RB_ARGS=()
if [ -n "${REASONING_BUDGET}" ]; then RB_ARGS=(--reasoning-budget "${REASONING_BUDGET}"); fi

echo "Starting llama-server (Arch / NVIDIA CUDA — RTX 3060 12 GB — Qwen3.6-35B-A3B-MTP, MoE offload + self-spec)"
echo "  binary   : ${LLAMA_SERVER}"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo "  context  : ${CTX} total / ${PARALLEL} slot(s)"
echo "  offload  : -ngl ${GPU_LAYERS} --n-cpu-moe ${NCPUMOE}  (experts of first ${NCPUMOE}/40 layers -> RAM; -t ${THREADS})"
if [ "${SPEC_OFF:-0}" = "1" ]; then echo "  MTP      : OFF (A/B baseline)"; else echo "  MTP      : on (draft-mtp, n=${SPEC_N})"; fi
echo

# NUMA note (dual-socket box): CPU-resident experts make decode RAM-bandwidth-bound,
# so cross-socket access hurts. If decode is slow:  numactl --interleave=all ./start-qwen36-mtp.sh

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
  --host "${HOST}" \
  --port "${PORT}" \
  --n-predict 4096 \
  --api-key "${API_KEY}" \
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
  ${RB_ARGS[@]+"${RB_ARGS[@]}"}
