#!/usr/bin/env bash
#
# Demo: build the bootstrap binary and run a real tool-using query against
# opencode, served locally by llama-server with Qwable-v1 (provider: llamacpp/qwable).
#
# Prereqs:
#   1. llama-server is running:   ~/tmp-hf/start-qwable.sh
#   2. opencode.json has the `llamacpp` provider + `qwable` model configured.
#
# Override the prompt or model via env:
#   MODEL=llamacpp/qwable PROMPT="..." ./scripts/demo.sh
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-llamacpp/qwable}"
# A task that forces actual tool use (list + read files), not just chat:
PROMPT="${PROMPT:-List the files in the current directory, then read Cargo.toml and tell me the package name, edition, and every dependency with its version.}"

echo "[demo] building bootstrap (release)..."
cargo build --release

echo "[demo] checking opencode can see the model..."
if ! opencode models 2>/dev/null | grep -q "${MODEL}"; then
  echo "[demo] WARNING: '${MODEL}' not listed by 'opencode models'."
  echo "[demo]          Is llama-server running and opencode.json configured?"
fi

echo "[demo] running tool-using query against ${MODEL}:"
echo "       \"${PROMPT}\""
echo "------------------------------------------------------------"
exec ./target/release/bootstrap exec "${PROMPT}" --model "${MODEL}" --verbose
