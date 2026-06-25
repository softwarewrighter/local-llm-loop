#!/usr/bin/env bash
#
# Start llama-server for GLM-5.2 (3-bit UD-IQ3_XXS) on this node: RTX 3060 12 GB +
# 503 GB system RAM. GLM-5.2 is a ~754B MoE (~40B active) — far too big for any
# single GPU — so it is a big-RAM, CPU-resident job with the GPU offloading only
# the dense/attention layers it can hold. Same expert-offload pattern as the
# Qwable 3060 config (start-qwable.sh), just much larger. See docs/glm-models.md.
#
#   Endpoint : http://127.0.0.1:8082/v1          (8080 = opencode UI, 8081 = qwable)
#   API key  : "local"   (must match the opencode.json provider apiKey)
#   Model id : glm-5.2   (the --alias; opencode refs it as llamacpp-glm/glm-5.2)
#
# PREREQUISITE — the model is NOT downloaded yet. Fetch the 3-bit shards into the
# dir below (≈282 GB), then run this script:
#
#   huggingface-cli download unsloth/GLM-5.2-GGUF \
#     --include "*UD-IQ3_XXS*" --local-dir /mnt/storage1/models/GLM-5.2
#
# MEMORY MATH (this box): 503 GB RAM holds the 282 GB model + KV + OS headroom
# comfortably. The 3060's 12 GB holds the dense/attention weights + KV; for a 754B
# model that dense slice may not fit at -ngl 99 — if the GPU OOMs, LOWER GPU_LAYERS
# (e.g. 40, then 20…) until the dense layers + KV fit 12 GB. The experts live in
# RAM regardless; the GPU mainly accelerates PREFILL (the CPU's weak spot here —
# E5-2697 v4 has AVX2 but no AMX/tensor cores). Expect single-digit tok/s decode:
# this is an OVERNIGHT-BATCH model on this hardware, not interactive.
#
# Override any default via env var, e.g.:
#   GPU_LAYERS=40 ./start-glm.sh          # offload fewer dense layers if 12 GB OOMs
#   REASONING_BUDGET=0 ./start-glm.sh     # disable thinking (see note below)
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-/mnt/storage1/src/llama.cpp/build/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-/mnt/storage1/models/GLM-5.2}"
# First shard of the multi-part GGUF; llama.cpp loads the rest automatically.
# (|| true so a glob miss doesn't trip `set -e` before the friendly guard below.)
if [ -z "${MODEL:-}" ]; then
  MODEL="$(ls "${MODEL_DIR}"/*UD-IQ3_XXS*-00001-of-*.gguf 2>/dev/null | head -1 || true)"
fi
ALIAS="${ALIAS:-glm-5.2}"       # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8082}"            # distinct from qwable (8081) and the opencode UI (8080)
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"  # all dense/attention layers to GPU; LOWER if 12 GB OOMs
EXPERT_OT="${EXPERT_OT:-.ffn_.*_exps.=CPU}"   # route MoE experts to CPU/RAM (size-independent)
CTX="${CTX:-32768}"             # modest ctx protects RAM for experts (GLM supports 1M)
PARALLEL="${PARALLEL:-1}"
THREADS="${THREADS:-36}"        # physical cores (E5-2697 v4: 18c x 2 sockets) — experts decode here
NPREDICT="${NPREDICT:-16384}"   # generous: GLM reasons before answering (see below)
# REASONING_BUDGET: leave UNSET to keep GLM's reasoning (its main value for hard
# tasks). Set to 0 if the planner/reviewer JSON gets truncated by long thinking.
# ----------------------------------------------------------------------------

if [ -z "${MODEL}" ] || [ ! -f "${MODEL}" ]; then
  echo "ERROR: GLM-5.2 GGUF not found in ${MODEL_DIR}" >&2
  echo "Download it first (see the header of this script), then re-run." >&2
  exit 1
fi

echo "Starting llama-server (GLM-5.2 UD-IQ3_XXS — RTX 3060 12 GB + 503 GB RAM, expert offload)"
echo "  binary   : ${LLAMA_SERVER}"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo "  context  : ${CTX} total / ${PARALLEL} slot(s)"
echo "  offload  : -ngl ${GPU_LAYERS}, experts -> CPU via -ot '${EXPERT_OT}'  (-t ${THREADS})"
echo

# NUMA note (dual-socket): the 282 GB of experts span both sockets' RAM; decode is
# RAM-bandwidth-bound, so try interleaving for steadier throughput:
#   numactl --interleave=all ./start-glm.sh

ARGS=(
  -m "${MODEL}"
  --alias "${ALIAS}"
  --gpu-layers "${GPU_LAYERS}"
  -ot "${EXPERT_OT}"
  --ctx-size "${CTX}"
  --parallel "${PARALLEL}"
  --threads "${THREADS}"
  --cont-batching
  --flash-attn on
  --cache-type-k q8_0 --cache-type-v q8_0
  --jinja
  --host "${HOST}"
  --port "${PORT}"
  --n-predict "${NPREDICT}"
  --api-key "${API_KEY}"
)
if [ -n "${REASONING_BUDGET:-}" ]; then
  ARGS+=(--reasoning-budget "${REASONING_BUDGET}")
fi

exec "${LLAMA_SERVER}" "${ARGS[@]}"
