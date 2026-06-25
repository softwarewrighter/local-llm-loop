#!/usr/bin/env bash
#
# Start llama-server for Qwable-v1 (IQ4_XS) on Arch Linux with a *small* NVIDIA
# GPU — the RTX 3060 12 GB — using MoE expert offload to system RAM.
#
#   Endpoint : http://127.0.0.1:8081/v1
#   API key  : "local"   (must match opencode.json provider.options.apiKey)
#   Model id : qwable    (the --alias; opencode refs it as llamacpp/qwable)
#
# WHY THIS DIFFERS FROM ../arch-nvidia (the 24 GB 3090 profile):
#
#   1. EXPERT OFFLOAD. Qwable-v1.IQ4_XS is ~17.6 GiB but the 3060 has only 12 GB
#      VRAM, so the whole model cannot be GPU-resident (`--gpu-layers 99` alone
#      OOMs). Qwable is a Qwen3.5-style MoE (qwen35moe: 34.66B total / ~3B active,
#      256 experts, 8 used per token, 40 layers). The giant *expert* tensors
#      dominate the file; the dense/attention layers are small. So we keep all
#      layers nominally on the GPU (-ngl 99) but push the MoE experts of the first
#      N layers back to RAM (--n-cpu-moe N). The GPU then holds the dense/attention
#      weights, the KV cache, and the experts of the remaining (40-N) layers.
#      Tune N: smaller N = more on the GPU = faster, until the 12 GB OOMs.
#      N=18 fills ~11.2 GB here (22 layers' experts on the GPU). Raise N if VRAM
#      is tight; lower it if you have headroom. This is the same 12 GB-card +
#      CPU/RAM-offload pattern GLM-5.2 uses (docs/glm-models.md), just smaller.
#
#   2. PORT 8081, not 8080. On this box 8080 (and 3000) are taken by the opencode
#      web UI Docker stack, so the model server uses 8081. opencode.json's baseURL
#      must match (http://localhost:8081/v1).
#
#   3. --reasoning-budget 0. This Qwable build is a "reasoning-distilled" model
#      that, left unbounded, emits a long <think> stream and can spend the whole
#      --n-predict budget reasoning before it answers. Budget 0 bounds that so the
#      actual answer (and the harness's JSON) comes through.
#
# Requires: a llama.cpp llama-server built with CUDA (see this dir's README),
# the NVIDIA driver + CUDA runtime, and the GGUF present locally.
#
# Override any default via env var, e.g.:
#   CTX=65536 ./start-qwable.sh
#   NCPUMOE=24 ./start-qwable.sh      # push more experts to RAM if 12 GB is tight
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-/mnt/storage1/src/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/mnt/storage1/qwable/Qwable-v1.IQ4_XS.gguf}"  # local GGUF (on the roomy mount)
ALIAS="${ALIAS:-qwable}"        # MUST match the opencode.json model key
HOST="${HOST:-127.0.0.1}"       # 0.0.0.0 only if a remote/Docker client needs it
PORT="${PORT:-8081}"            # MUST match opencode.json baseURL port (8080 = opencode UI)
API_KEY="${API_KEY:-local}"     # MUST match opencode.json apiKey
GPU_LAYERS="${GPU_LAYERS:-99}"  # 99 = all layers nominally to GPU; experts pulled back below
NCPUMOE="${NCPUMOE:-18}"        # experts of the first N (of 40) layers -> CPU/RAM; tune to VRAM
CTX="${CTX:-32768}"             # total context (matches opencode limit.context)
PARALLEL="${PARALLEL:-1}"       # concurrent slots
THREADS="${THREADS:-36}"        # physical cores (E5-2697 v4: 18c x 2 sockets) — CPU experts decode here
# ----------------------------------------------------------------------------

echo "Starting llama-server (Arch / NVIDIA CUDA — RTX 3060 12 GB, MoE expert offload)"
echo "  binary   : ${LLAMA_SERVER}"
echo "  model    : ${MODEL}  (alias: ${ALIAS})"
echo "  endpoint : http://${HOST}:${PORT}/v1   (api key: ${API_KEY})"
echo "  context  : ${CTX} total / ${PARALLEL} slot(s)"
echo "  offload  : -ngl ${GPU_LAYERS} --n-cpu-moe ${NCPUMOE}  (experts of first ${NCPUMOE}/40 layers -> RAM; -t ${THREADS})"
echo

# NUMA note (dual-socket box): the CPU-resident experts make decode partly
# RAM-bandwidth-bound, so cross-socket access hurts. If decode is slow, try:
#   numactl --interleave=all ./start-qwable.sh
# or pin to one node: numactl --membind=0 --cpunodebind=0 ./start-qwable.sh

exec "${LLAMA_SERVER}" \
  -m "${MODEL}" \
  --alias "${ALIAS}" \
  --gpu-layers "${GPU_LAYERS}" \
  --n-cpu-moe "${NCPUMOE}" \
  --reasoning-budget 0 \
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
  --api-key "${API_KEY}"
