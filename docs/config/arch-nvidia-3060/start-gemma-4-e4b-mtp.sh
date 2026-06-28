#!/usr/bin/env bash
#
# Start llama-server on the RTX 3060 12 GB (Ampere, sm_86) with Gemma-4-E4B-it
# (effective ~4B) + its MTP head for SELF-speculative decoding — a "go smaller"
# experiment below Ornith-9B (see ../../plan-smaller-models-3060.md).
#
#   Endpoint : http://127.0.0.1:8081/v1
#   Model id : gemma-4-e4b-mtp   (opencode refs it as llamacpp/gemma-4-e4b-mtp)
#
# GEMMA MTP ≠ ORNITH MTP. Ornith's head is *bundled* in the model GGUF
# (--spec-type draft-mtp alone). Gemma-4 ships the head as a SEPARATE ~94 MB file
# (mtp-gemma-4-E4B-it.gguf) passed via --model-draft. Per the unsloth MTP guide:
#   llama-server -m <base>.gguf --model-draft mtp-...gguf --spec-type draft-mtp
#   --spec-draft-n-max 2   (test 1-6). Needs a recent llama.cpp (PR #22673).
#
# WHY TEST IT: it answers two open questions at once — (1) does llama.cpp's MTP
# support cover Gemma-4 at all, and (2) does a ~4B model cross our code floor (our
# measured prior: general <=14B fails; only purpose-trained Ornith-9B passed). If
# it can't code, it's logged as draft-grade, not coder-grade.
#
# FIT: Q4_K_M base ~4.7 GB + 94 MB head, resident, huge KV headroom on 12 GB.
# A/B the MTP uplift:  SPEC_OFF=1 ./start-gemma-4-e4b-mtp.sh
# PORT 8081 (8080 = opencode UI Docker on this box).
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/data2/llm/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/data2/llm/models/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf}"
DRAFT="${DRAFT:-/data2/llm/models/gemma-4-e4b/mtp-gemma-4-E4B-it.gguf}"  # the MTP head
ALIAS="${ALIAS:-gemma-4-e4b-mtp}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"
CTX="${CTX:-32768}"
PARALLEL="${PARALLEL:-1}"
SPEC_N="${SPEC_N:-2}"            # unsloth guide: start at 2, test 1-6
# MEASURED: Gemma-4 MTP + flash-attn ON crashes (CUDA fatal in ggml-cuda/fattn.cu).
# MTP requires flash-attn OFF on this build (commit dbdaece). fa on/off barely
# affects decode here (~79 t/s either way), so default off when MTP is engaged.
FLASH_ATTN="${FLASH_ATTN:-off}"

SPEC_ARGS=(--model-draft "${DRAFT}" --spec-type draft-mtp --spec-draft-n-max "${SPEC_N}")
if [ "${SPEC_OFF:-0}" = "1" ]; then SPEC_ARGS=(); FLASH_ATTN="${FLASH_ATTN_BASE:-on}"; fi  # A/B: base, MTP off

echo "Starting llama-server (RTX 3060 12 GB — Gemma-4-E4B-it + MTP head, resident, self-spec)"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  draft    : ${DRAFT}"
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
  --flash-attn "${FLASH_ATTN}" \
  --jinja \
  --host "${HOST}" \
  --port "${PORT}" \
  --n-predict 4096 \
  --api-key "${API_KEY}" \
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"}
