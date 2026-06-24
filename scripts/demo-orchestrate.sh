#!/usr/bin/env bash
#
# Demo: the full planner -> executor -> reviewer loop.
#
# `bootstrap orchestrate <spec>` reads a spec file, asks the LLM to turn it into
# a design + serialized step list, then iterates the steps — executing each
# (tool-using) and feeding the structured result back to the LLM, which decides
# to continue / insert / skip / stop after every step.
#
# The build happens in an isolated temp dir so it never touches this repo.
#
# Prereqs:
#   1. llama-server running:   ~/tmp-hf/start-qwable.sh
#   2. opencode.json has the `llamacpp` provider + `qwable` model.
#
# Env overrides:
#   MODEL=llamacpp/qwable   SPEC=examples/spec-greeter.txt   PLAN_ONLY=1
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-llamacpp/qwable}"
SPEC="${SPEC:-examples/spec-greeter.txt}"
BIN=./target/release/bootstrap

echo "[demo] building bootstrap (release)..."
cargo build --release

WORKDIR="$(mktemp -d -t bootstrap-greeter.XXXXXX)"
echo "[demo] build workspace: ${WORKDIR}"
cp "${SPEC}" "${WORKDIR}/spec.txt"

ARGS=(orchestrate "${WORKDIR}/spec.txt" --model "${MODEL}" --work-dir "${WORKDIR}" --verbose)
if [ "${PLAN_ONLY:-0}" = "1" ]; then
  ARGS+=(--plan-only)
  echo "[demo] PLAN_ONLY=1 — will stop after planning."
fi

echo "[demo] running orchestrator against ${MODEL}..."
echo "------------------------------------------------------------"
"${BIN}" "${ARGS[@]}"

echo
echo "============================================================"
echo "[demo] artifacts (plan / per-step results & reviews):"
ls -1 "${WORKDIR}/.bootstrap" 2>/dev/null || echo "  (none)"
echo "[demo] produced files in workspace:"
( cd "${WORKDIR}" && find . -type f -not -path './.bootstrap/*' -not -path './target/*' | sort )
echo "[demo] workspace left at: ${WORKDIR}"
