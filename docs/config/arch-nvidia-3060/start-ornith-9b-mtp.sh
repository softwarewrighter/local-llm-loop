#!/usr/bin/env bash
#
# Start llama-server on the RTX 3060 12 GB (Ampere, sm_86) with Ornith-1.0-9B
# PLUS a baked-in MTP next-token head — RESIDENT dense coder with SELF-speculative
# decoding, no draft model and no MoE offload. This is the FAST FALLBACK for this
# box (see ../../plan-rtx3060-12.md "Fallback tier"): if the offloaded A3B MoEs
# (start-qwen36-mtp.sh / start-qwen3-coder.sh) run too slow, drop to this.
#
#   Endpoint : http://127.0.0.1:8081/v1
#   API key  : "local"            (must match opencode.json provider.options.apiKey)
#   Model id : ornith-9b-mtp       (opencode refs it as llamacpp/ornith-9b-mtp)
#
# WHY THIS BEATS A DRAFT PAIR ON THIS BOX:
#   - Spec-decode rewards DENSE models (every token reads all weights -> lots to
#     amortize). On an A3B MoE it is break-even-to-NEGATIVE on Ampere (measured
#     -40%..-52% on a 3090: thc1006/qwen3.6-speculative-decoding-rtx3090), because
#     only ~3B params are active per token already. So MTP pays off HERE (dense
#     9B), not on the offloaded MoEs.
#   - Ornith's custom 248320 vocab rules out borrowing an external Qwen3-0.6B
#     draft, so the MTP self-spec head is the ONLY spec-decode route for the 9B.
#   - It is a REAL coder (SWE-bench Verified 69.4, 0 retries on the 5060) — the
#     only sub-14B that clears the harness gates.
#
# THE MTP GGUF: protoLabsAI/Ornith-1.0-9B-MTP-GGUF bakes a KL-distilled next-token
# head onto the 9B trunk for llama.cpp's --spec-type draft-mtp. Measured ~1.4-1.7x
# single-stream decode speedup on an A6000 (Ampere), per-token acceptance ~0.766,
# distribution-lossless (drafts are verified against the target). Requires
# llama.cpp >= b9616 with Qwen3.5 arch support (this box's b9784+ build qualifies).
#
#   hf download protoLabsAI/Ornith-1.0-9B-MTP-GGUF <file>.gguf \
#     --local-dir /mnt/storage1/ornith-9b-mtp
#
# FIT ON 12 GB: dense, RESIDENT. Q6_K 7.36 GiB + ~1.5 GB q8_0 KV ~= 9 GB; the MTP
# head adds a little. Drop to Q5_K_M (6.47) / Q4_K_M (5.63) for more headroom/speed
# if quality holds. NO --n-cpu-moe (it is dense, not MoE).
#
# A/B the MTP uplift:  SPEC_OFF=1 ./start-ornith-9b-mtp.sh   (serves the same model
# with MTP off — the speculative head is simply not engaged).
#
# Ornith is a reasoning model (<think>); add --reasoning-budget 0
# (REASONING_BUDGET=0 ...) if it spends the whole --n-predict budget thinking.
#
# PORT 8081, not 8080: 8080 is the opencode UI Docker stack here.
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-/mnt/storage1/src/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/mnt/storage1/ornith-9b-mtp/Ornith-1.0-9B-MTP-Q6_K.gguf}"  # dense + MTP head, resident
ALIAS="${ALIAS:-ornith-9b-mtp}"  # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"             # 8080 = opencode UI Docker on this box; server uses 8081
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"   # 99 = whole model on the GPU (fits 12 GB easily)
CTX="${CTX:-32768}"              # total context (matches opencode limit.context)
PARALLEL="${PARALLEL:-1}"
SPEC_N="${SPEC_N:-3}"            # speculative tokens per step (card: 3 = max throughput, 2 = max accept)
REASONING_BUDGET="${REASONING_BUDGET:-}"  # set 0 if <think> eats the n-predict budget
# ----------------------------------------------------------------------------

# BUNDLED MTP: the next-token head is baked into the GGUF, so engage it with
# --spec-type draft-mtp ALONE. Do NOT pass --spec-draft-model <same file> — that
# loads a SECOND full copy of the model as the draft and OOMs a 12 GB card.
SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max "${SPEC_N}")
if [ "${SPEC_OFF:-0}" = "1" ]; then SPEC_ARGS=(); fi   # A/B: serve the same model with MTP off

RB_ARGS=()
if [ -n "${REASONING_BUDGET}" ]; then RB_ARGS=(--reasoning-budget "${REASONING_BUDGET}"); fi

echo "Starting llama-server (Arch / NVIDIA CUDA — RTX 3060 12 GB — Ornith-1.0-9B + MTP, dense, resident, self-spec)"
echo "  binary   : ${LLAMA_SERVER}"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo "  context  : ${CTX} total / ${PARALLEL} slot(s)"
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
