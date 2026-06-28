#!/usr/bin/env bash
#
# Download the RTX 5060 Ti 16 GB candidate GGUFs to /disk1/models.
#
# Keeps everything off the small root partition: models land on /disk1 (most
# free space) and the HF cache is pinned to /disk1/hf-cache. Uses the `hf` CLI
# from a uv venv (NOT system pip / conda):
#   uv venv /disk1/venvs/llm-loop --python 3.12
#   source /disk1/venvs/llm-loop/bin/activate && uv pip install "huggingface_hub[hf_transfer]"
#
# Models are downloaded in PRIORITY order (most likely to complete the loop, and
# the FP4 showcase, first). Set HF_TOKEN for higher rate limits / faster pulls.
#
# Usage:  ./download-models.sh            # all candidates, in priority order
#         ONLY=gptoss ./download-models.sh   # just one (gptoss|coder|phi4|qwen3-8b|qwen36|gemma3)
set -euo pipefail

export HF_HOME="${HF_HOME:-/disk1/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/disk1/hf-cache/hub}"
MODELS_DIR="${MODELS_DIR:-/disk1/models}"
VENV="${VENV:-/disk1/venvs/llm-loop}"

[ -f "${VENV}/bin/activate" ] && source "${VENV}/bin/activate"
command -v hf >/dev/null || { echo "hf CLI not found — create the uv venv first (see header)"; exit 1; }

dl() { # repo  file  subdir
  local repo="$1" file="$2" sub="$3"
  echo ">>> ${repo} :: ${file} -> ${MODELS_DIR}/${sub}"
  mkdir -p "${MODELS_DIR}/${sub}"
  hf download "${repo}" "${file}" --local-dir "${MODELS_DIR}/${sub}"
}

want() { [ -z "${ONLY:-}" ] || [ "${ONLY}" = "$1" ]; }

# 1. FP4 showcase — fits 16 GB whole, fastest decode candidate on Blackwell
want gptoss   && dl ggml-org/gpt-oss-20b-GGUF                  gpt-oss-20b-mxfp4.gguf                       gpt-oss-20b
# 2. Proven loop-completer (MoE, needs light --n-cpu-moe on 16 GB)
want coder    && dl unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF  Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf    qwen3-coder-30b
# 3. Dense 14B that fits whole — structured-output candidate
want phi4     && dl unsloth/phi-4-GGUF                         phi-4-Q4_K_M.gguf                            phi-4
# 4. Small Qwen3 + draft — spec-decode study (exact 151936 vocab)
want qwen3-8b && { dl unsloth/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf qwen3-8b
                   dl unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf qwen3-0.6b; }
# 5. Dense 27B (cleanest structure on the 3090) — tight fit, needs small offload/ctx
want qwen36   && dl unsloth/Qwen3.6-27B-GGUF                   Qwen3.6-27B-Q4_K_M.gguf                      qwen3.6-27b
# 5b. Gemma-4-26B-A4B MoE, MXFP4 — VERIFIED 3rd fast+good coder (FP4-native, ~61 t/s, clean loop)
want gemma4   && dl unsloth/gemma-4-26B-A4B-it-GGUF           gemma-4-26B-A4B-it-MXFP4_MOE.gguf            gemma4-26b
# 6. Gemma quality candidate (the plan wanted "Gemma-4"; Gemma-3-27B is the available analog)
want gemma3   && dl unsloth/gemma-3-27b-it-GGUF               gemma-3-27b-it-Q4_K_M.gguf                   gemma-3-27b

echo "done. models under ${MODELS_DIR}:"
du -sh "${MODELS_DIR}"/*/ 2>/dev/null || true
