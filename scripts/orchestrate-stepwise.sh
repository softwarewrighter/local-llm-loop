#!/usr/bin/env bash
#
# Run the orchestrate loop ONE STEP AT A TIME against a (slow) model, so a
# multi-hour build can be spread over many short, separately-verifiable runs.
#
# Unlike demo-orchestrate.sh (fresh temp dir, whole loop in one shot), this uses
# a PERSISTENT workspace and the harness's --resume so each invocation does just
# one step and saves state. Verify each step's artifacts before running the next.
#
# Usage:
#   scripts/orchestrate-stepwise.sh plan       # plan only; saves resumable state
#   scripts/orchestrate-stepwise.sh step        # execute the next 1 step, then review
#   scripts/orchestrate-stepwise.sh status      # show plan + progress + next step
#   scripts/orchestrate-stepwise.sh reset       # delete the workspace and start over
#   scripts/orchestrate-stepwise.sh full        # run the whole loop (no chunking)
#
# Env overrides:
#   MODEL=llamacpp-glm/glm-5.2   (default)   provider/model
#   SPEC=examples/spec-greeter.txt           the spec file
#   WS=/mnt/storage1/glm-runs/greeter        persistent workspace (state lives here)
#   N=1                                       steps per `step` invocation
set -euo pipefail
cd "$(dirname "$0")/.."

CMD="${1:-help}"
MODEL="${MODEL:-llamacpp-glm/glm-5.2}"
SPEC="${SPEC:-examples/spec-greeter.txt}"
WS="${WS:-/mnt/storage1/glm-runs/greeter}"
N="${N:-1}"
BIN=./target/release/bootstrap

build() { echo "[stepwise] building bootstrap (release)…"; cargo build --release >/dev/null; }

show_status() {
  local st="${WS}/.bootstrap/state.json"
  if [ ! -f "$st" ]; then echo "[stepwise] no state yet at ${st} — run 'plan' first."; return; fi
  python3 - "$st" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
steps = s.get("steps", [])
cur = s.get("cursor", 0)
print(f"workspace state: {s.get('executed',0)} executed | cursor {cur}/{len(steps)} | finished={s.get('finished')}")
if s.get("stopped_reason"): print("  stopped:", s["stopped_reason"])
print("plan steps:")
for i, st in enumerate(steps):
    mark = "✓ done " if i < cur else ("→ NEXT " if i == cur else "  todo ")
    print(f"  {mark}[{st['id']}] {st['title']}")
PY
}

list_artifacts() {
  echo "[stepwise] artifacts in ${WS}/.bootstrap:"
  ls -1 "${WS}/.bootstrap" 2>/dev/null | grep -vE '\.llm\.json$' | sed 's/^/    /' || echo "    (none)"
  echo "[stepwise] files produced in workspace (excluding .bootstrap/target/.git):"
  ( cd "$WS" && find . -type f \
      -not -path './.bootstrap/*' -not -path './target/*' -not -path './*/target/*' \
      -not -path './.git/*' -not -path './*/.git/*' | sort | sed 's/^/    /' ) 2>/dev/null || true
}

case "$CMD" in
  plan)
    build
    mkdir -p "$WS"
    cp "$SPEC" "${WS}/spec.txt"
    echo "[stepwise] planning with ${MODEL} into ${WS} (this can take a while on a slow model)…"
    "$BIN" orchestrate "${WS}/spec.txt" --model "$MODEL" --work-dir "$WS" --plan-only --verbose
    echo; show_status
    ;;
  step)
    build
    [ -f "${WS}/.bootstrap/state.json" ] || { echo "[stepwise] no saved state — run 'plan' first."; exit 1; }
    echo "[stepwise] executing next ${N} step(s) with ${MODEL} (resume)…"
    "$BIN" orchestrate "${WS}/spec.txt" --model "$MODEL" --work-dir "$WS" --resume --max-steps "$N" --verbose
    echo; show_status; echo; list_artifacts
    ;;
  status) show_status ;;
  reset)  echo "[stepwise] removing ${WS}"; rm -rf "$WS"; echo "done." ;;
  full)
    build
    mkdir -p "$WS"; cp "$SPEC" "${WS}/spec.txt"
    "$BIN" orchestrate "${WS}/spec.txt" --model "$MODEL" --work-dir "$WS" --max-steps 20 --verbose
    echo; list_artifacts
    ;;
  *)
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
