#!/usr/bin/env bash
#
# Start llama-server on the RTX 5060 Ti 16 GB (Blackwell, sm_120) with
# Ornith-1.0-35B (Qwen-3.5 MoE, 256 experts / 8 active ≈ 3B active) — the
# **stretch** config for this 16 GB box: the 35B does NOT fit whole (Q4_K_M is
# ~20 GB), so push expert FFNs to system RAM with --n-cpu-moe (decode mostly
# survives since only ~3B params are active per token; prefill takes the hit).
# Same MoE-offload regime as start-gemma4-26b.sh / start-qwen3-coder.sh.
#
#   Endpoint : http://127.0.0.1:8080/v1
#   Model id : ornith-35b   (opencode refs llamacpp/ornith-35b)
#
# ARCH: qwen35moe (35B.A3B) — confirmed supported by the sm_120 build (3fc4e10)
# via llama-bench. Gate 1 (native tool_calls) uses the same Ornith/Qwen parser
# that passed for Ornith-9B.
#
# Download first (~20 GB):
#   hf download bartowski/deepreinforce-ai_Ornith-1.0-35B-GGUF \
#     deepreinforce-ai_Ornith-1.0-35B-Q4_K_M.gguf --local-dir /disk1/models/ornith-35b
#
# Tune N_CPU_MOE: lower = faster (more on GPU) but risks OOM; raise if it OOMs.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-/disk1/build/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/disk1/models/ornith-35b/deepreinforce-ai_Ornith-1.0-35B-Q4_K_M.gguf}"  # ~20 GB MoE
ALIAS="${ALIAS:-ornith-35b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"
N_CPU_MOE="${N_CPU_MOE:-28}"     # expert layers whose FFNs live in RAM; lower → faster, raise if OOM
CTX="${CTX:-16384}"
PARALLEL="${PARALLEL:-1}"

echo "Starting llama-server (RTX 5060 Ti 16 GB — Ornith-1.0-35B qwen35moe, --n-cpu-moe ${N_CPU_MOE})"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
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
  --host "${HOST}" \
  --port "${PORT}" \
  --n-predict 4096 \
  --api-key "${API_KEY}"
