#!/usr/bin/env bash
#
# Start llama-server on the RTX 5060 Ti 16 GB (Blackwell, sm_120) with
# Gemma-4-26B-A4B-it — MoE (~4B active), the MXFP4 quant so the expert weights
# run on Blackwell's NATIVE FP4 path (like gpt-oss-20b).
#
#   Endpoint : http://127.0.0.1:8080/v1
#   Model id : gemma4-26b   (opencode refs llamacpp/gemma4-26b)
#
# WHY: this is the **third verified fast+good coder** on 16 GB (see
# ../../nvidia-5060-poc-summary.md). It completed the loop cleanly (single-crate,
# `cargo test` 2/2), used the `bootstrap emit` helper correctly with ZERO retries,
# and decodes ~61 t/s (bench). It was also the best-code model on the 3090.
#
# FIT: the MXFP4 GGUF is ~15.4 GB — it does NOT fit 16 GB whole. Being MoE, push
# expert layers to system RAM with --n-cpu-moe N; decode survives (only ~4B
# active params/token). N=12 fits serving at ctx 16384 with headroom; lower N
# (toward 6–8) is faster if VRAM allows — measured ~61 t/s at N=6 (llama-bench,
# weights-only). Gate 1 works with the EMBEDDED template (no --chat-template-file).
#
# Override any default via env var.
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-/disk1/build/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/disk1/models/gemma4-26b/gemma-4-26B-A4B-it-MXFP4_MOE.gguf}"  # ~15.4 GB MXFP4 MoE
ALIAS="${ALIAS:-gemma4-26b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-local}"
GPU_LAYERS="${GPU_LAYERS:-99}"
N_CPU_MOE="${N_CPU_MOE:-12}"    # expert layers whose FFNs live in RAM; lower → faster, raise if OOM
CTX="${CTX:-16384}"
PARALLEL="${PARALLEL:-1}"
# ----------------------------------------------------------------------------

echo "Starting llama-server (RTX 5060 Ti 16 GB — Gemma-4-26B-A4B MoE, MXFP4/FP4-native, --n-cpu-moe ${N_CPU_MOE})"
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
