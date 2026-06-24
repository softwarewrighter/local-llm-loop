#!/usr/bin/env bash
#
# Demo: ONE `bootstrap` CLI call processes a list of 3 tasks, making 3 opencode
# (tool-using) calls against Qwable-v1 (provider llamacpp/qwable):
#
#   1. create a test
#   2. run the test
#   3. validate the results
#
# bootstrap chains the 3 tasks into a single opencode session (--continue after
# the first), so step 3 ("validate") remembers the test created in step 1.
#
# Prereqs:
#   1. llama-server running:   ~/tmp-hf/start-qwable.sh
#   2. opencode.json has the `llamacpp` provider + `qwable` model.
#
# Override the model via env:  MODEL=llamacpp/qwable ./scripts/demo-3-steps.sh
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-llamacpp/qwable}"
BIN=./target/release/bootstrap
TEST_FILE="tests/demo_add.rs"

echo "[demo] building bootstrap (release)..."
cargo build --release

echo "[demo] checking opencode can see the model..."
if ! opencode models 2>/dev/null | grep -q "${MODEL}"; then
  echo "[demo] WARNING: '${MODEL}' not listed by 'opencode models'."
  echo "[demo]          Is llama-server running and opencode.json configured?"
fi

# Start clean so task 1 genuinely creates the file.
rm -f "${TEST_FILE}"

# The 3 tasks. Because bootstrap chains them in one session, tasks 2 and 3 can
# refer back to "the test you created" rather than re-deriving everything.
TASK1="Create a Rust integration test at ${TEST_FILE}. It must contain exactly one test function named adds_two_and_two, annotated with #[test], that asserts 2 + 2 == 4. Create only that file; do not modify any other files."
TASK2="Run the test you just created with 'cargo test --test demo_add' and report the full, exact output."
TASK3="Validate the result of the test you just ran: state PASS if adds_two_and_two passed and FAIL otherwise, and quote the exact cargo output line that justifies your verdict."

echo "[demo] one CLI call, three chained opencode tasks against ${MODEL}..."
echo "------------------------------------------------------------"
"${BIN}" exec "${TASK1}" "${TASK2}" "${TASK3}" --model "${MODEL}" --verbose

echo
echo "============================================================"
echo "[demo] done. created test file (${TEST_FILE}):"
echo "------------------------------------------------------------"
cat "${TEST_FILE}" 2>/dev/null || echo "[demo] WARNING: ${TEST_FILE} was not created."
