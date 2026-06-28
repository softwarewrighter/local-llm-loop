#!/usr/bin/env bash
#
# Start mlx_lm.server for Ornith-1.0-35B (MLX 6-bit) on macOS (Apple Silicon).
# MLX is Apple's framework — on the M1 Max it decodes ~1.35x faster than the
# equivalent GGUF on llama.cpp/Metal (measured: 55.6 vs 40.7 t/s, see
# performance-analysis.md / plan-ornith-models.md). This is the **winner** of the
# MLX-vs-GGUF comparison for the 35B on Apple Silicon.
#
#   Endpoint : http://127.0.0.1:8080/v1   (OpenAI-compatible)
#   Model id : ornith35mlx   (opencode refs it as mlx/ornith35mlx)
#
# Ornith is a Qwen-3.5-based MoE (~3B active) reasoning coder (<think> blocks),
# vocab 248320, 256k context. Gate 1 (native tool_calls) passes via mlx_lm.server.
#
# Requires: an mlx-lm venv. Create once with uv:
#   uv venv --python 3.12 ~/tmp-hf/mlxenv
#   VIRTUAL_ENV=~/tmp-hf/mlxenv uv pip install mlx-lm
# and the MLX weights downloaded to $MODEL (e.g. via huggingface_hub
# snapshot_download "mlx-community/Ornith-1.0-35B-6bit").
#
# mlx_lm.server has no --alias: it serves the model under its --model STRING, so
# we point --model at a short relative symlink and reference that exact id.
set -euo pipefail

VENV="${VENV:-$HOME/tmp-hf/mlxenv}"
MODEL_DIR="${MODEL_DIR:-$HOME/tmp-hf/ornith-35b-mlx-6bit}"   # snapshot_download target
ALIAS="${ALIAS:-ornith35mlx}"   # the served model id (MUST match opencode.json)
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

# Serve under a short relative name so opencode can reference `mlx/ornith35mlx`.
ln -sfn "$(basename "${MODEL_DIR}")" "$(dirname "${MODEL_DIR}")/${ALIAS}"
cd "$(dirname "${MODEL_DIR}")"

echo "Starting mlx_lm.server (macOS / MLX — Ornith-1.0-35B 6-bit)"
echo "  model id : ${ALIAS}  ->  ${MODEL_DIR}"
echo "  endpoint : http://${HOST}:${PORT}/v1"
echo

exec "${VENV}/bin/python" -m mlx_lm server \
  --model "${ALIAS}" \
  --host "${HOST}" \
  --port "${PORT}"
# For MTP / draft self-speculation, add: --draft-model <mlx-mtp-or-9b> --num-draft-tokens 4
