#!/usr/bin/env bash
#
# Start llama-server for Qwen3.6-35B-A3B-MTP on macOS (Apple Silicon / Metal),
# with the built-in MTP head driving SELF-speculative decoding — no separate
# draft model. This is the Apple-Silicon pick from performance-analysis.md: MTP
# is the preferred speedup on bandwidth-bound boxes (M1 Max), ~85% accept.
#
#   Endpoint : http://127.0.0.1:8080/v1
#   API key  : "local"        (must match opencode.json provider.options.apiKey)
#   Model id : qwen36-mtp      (the --alias; opencode refs it as llamacpp/qwen36-mtp)
#
# The 64 GB unified memory holds this ~21 GB MoE WHOLE — no expert offload (the
# 16 GB 5060 Ti had to spill to system RAM). Decode rides only the ~3B active
# params, so it stays fast; MTP self-speculation lifts it further.
#
# MTP flags: `--spec-type draft-mtp` engages the embedded next-token head. The
# main GGUF is also the draft (-md), so no second model is loaded.
#
# Override via env, e.g.:  SPEC_OFF=1 ./start-qwen36-mtp.sh   (disable MTP, A/B)
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
MODEL="${MODEL:-$HOME/tmp-hf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
ALIAS="${ALIAS:-qwen36-mtp}"    # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"  # whole model on Metal (unified memory)
CTX="${CTX:-32768}"             # matches opencode limit.context
PARALLEL="${PARALLEL:-1}"
SPEC_N="${SPEC_N:-4}"           # speculative tokens per step
# ----------------------------------------------------------------------------

SPEC_ARGS=(--spec-type draft-mtp --spec-draft-model "${MODEL}" --spec-draft-n-max "${SPEC_N}")
if [ "${SPEC_OFF:-0}" = "1" ]; then
  SPEC_ARGS=()   # A/B: serve the same model with MTP off
fi

echo "Starting llama-server (macOS / Metal — Qwen3.6-35B-A3B-MTP, self-spec-decode)"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo "  context  : ${CTX} total / ${PARALLEL} slot(s)"
if [ "${SPEC_OFF:-0}" = "1" ]; then echo "  MTP      : OFF (A/B baseline)"; else echo "  MTP      : on (draft-mtp, n=${SPEC_N})"; fi
echo

exec llama-server \
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
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"}
