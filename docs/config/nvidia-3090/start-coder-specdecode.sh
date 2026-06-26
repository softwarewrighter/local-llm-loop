#!/usr/bin/env bash
#
# Start llama-server on the RTX 3090 24 GB with a TWO-MODEL speculative-decoding
# coder profile: a 14B target verified by a tiny same-family 1.5B draft, both
# fully GPU-resident, with comfortable VRAM headroom for a 32k context. This is
# the "fast, low-power, GPU-resident" config from
# docs/optimizations-plan.md — the opposite regime from the RAM-offload GLM/3060
# setups (../arch-nvidia-3060). The whole working set lives in VRAM, so decode is
# fast (~50–90 t/s expected) and energy-cheap (GPU for minutes, not the whole box
# for hours).
#
#   Endpoint : http://127.0.0.1:8080/v1
#   API key  : "local"      (must match opencode.json provider.options.apiKey)
#   Model id : qwen-coder   (the --alias; opencode refs it as llamacpp/qwen-coder)
#
# WHY SPECULATIVE DECODING:
#   The draft (0.6B) proposes several tokens per step; the target (14B) verifies
#   them in one forward pass. Verification accepts/resamples, so the OUTPUT IS
#   IDENTICAL to sampling the 14B alone — lossless w.r.t. quality, only faster.
#   Coding tasks draft well (predictable text) so acceptance is high. The draft
#   and target are both Qwen3, which uses one shared 151936-token vocab across all
#   sizes → exact tokenizer match, which spec-decode requires.
#
# --reasoning-budget 0: Qwen3 is a hybrid reasoning model that, left unbounded,
# emits a long <think> stream and can spend the whole --n-predict budget thinking
# before it answers (same failure mode as the Qwable build). Budget 0 forces
# direct answers — which is what we want for fast agentic coding (fewest
# tokens/task). The draft (also Qwen3) follows the same template.
#
# FLAG NAMES target llama.cpp build ~9728 (the system /usr/bin/llama-server here),
# which renamed the older --draft-max/--gpu-layers-draft flags:
#   --spec-draft-n-max  (was --draft-max)   --spec-draft-n-min (was --draft-min)
#   --gpu-layers-draft / -ngld  still accepted.   --model-draft / -md  unchanged.
#
# Requires: a CUDA llama-server (sm_86), NVIDIA driver + CUDA runtime, and both
# GGUFs present locally.
#
# Override any default via env var, e.g.:
#   CTX=65536 ./start-coder-specdecode.sh
#   NMAX=12 PMIN=0.5 ./start-coder-specdecode.sh    # retune the draft window
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-/usr/bin/llama-server}"             # system CUDA build (v9728, has spec-decode)
# MODEL CHOICE — Qwen3, not Qwen2.5-Coder. Two reasons:
#   1. TOOL CALLING. opencode is agentic: every role must emit *native* tool calls
#      so opencode actually executes file writes / cargo, etc. Qwen3 does this
#      correctly (llama-server returns finish_reason=tool_calls). Qwen2.5-Coder-14B
#      Q4 here emits its call with the WRONG delimiter (<tools> instead of
#      <tool_call>), so llama-server can't parse it → opencode never runs the tool
#      → the executor "narrates" work it never did. That breaks the loop.
#   2. EXACT VOCAB. Qwen3 uses one 151936-token vocab across all sizes, so the 14B
#      target + 0.6B draft match exactly and spec-decode engages cleanly.
# (Qwen2.5-Coder drafts faster — ~84% acceptance vs ~55% — but its broken tool
# calling makes it unusable for the agentic loop. Speed is moot if no tools fire.)
MODEL="${MODEL:-/disk1/models/qwen-coder/Qwen_Qwen3-14B-Q4_K_M.gguf}"   # ~9 GB target
DRAFT="${DRAFT:-/disk1/models/qwen-coder/Qwen_Qwen3-0.6B-Q8_0.gguf}"    # ~0.6 GB draft (exact vocab match)
ALIAS="${ALIAS:-qwen-coder}"    # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"       # 0.0.0.0 only if a remote/Docker client needs it
PORT="${PORT:-8080}"            # MUST match opencode.json baseURL port (free on this box)
API_KEY="${API_KEY:-local}"     # MUST match opencode.json apiKey
GPU_LAYERS="${GPU_LAYERS:-99}"  # all target layers on the 3090
GPU_LAYERS_DRAFT="${GPU_LAYERS_DRAFT:-99}"   # all draft layers on the 3090 too
NMAX="${NMAX:-16}"              # tokens to draft per step (code accepts well → high)
NMIN="${NMIN:-4}"               # floor on drafted tokens
PMIN="${PMIN:-0.6}"             # min draft probability to keep speculating (greedy)
CTX="${CTX:-32768}"             # total context (matches opencode limit.context)
PARALLEL="${PARALLEL:-1}"       # concurrent slots
# ----------------------------------------------------------------------------

echo "Starting llama-server (RTX 3090 24 GB — Qwen2.5-Coder + 1.5B draft, speculative decoding)"
echo "  binary   : ${LLAMA_SERVER}"
echo "  target   : ${MODEL}  (alias: ${ALIAS})"
echo "  draft    : ${DRAFT}"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo "  context  : ${CTX} total / ${PARALLEL} slot(s)"
echo "  spec     : n-max ${NMAX} / n-min ${NMIN} / p-min ${PMIN}  (target -ngl ${GPU_LAYERS}, draft -ngld ${GPU_LAYERS_DRAFT})"
echo

exec "${LLAMA_SERVER}" \
  -m "${MODEL}" \
  --model-draft "${DRAFT}" \
  --spec-type draft-simple \
  --alias "${ALIAS}" \
  --gpu-layers "${GPU_LAYERS}" \
  --gpu-layers-draft "${GPU_LAYERS_DRAFT}" \
  --spec-draft-n-max "${NMAX}" \
  --spec-draft-n-min "${NMIN}" \
  --spec-draft-p-min "${PMIN}" \
  --reasoning-budget 0 \
  --ctx-size "${CTX}" \
  --parallel "${PARALLEL}" \
  --cont-batching \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --host "${HOST}" \
  --port "${PORT}" \
  --n-predict 4096 \
  --api-key "${API_KEY}"
