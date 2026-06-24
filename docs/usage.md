# Usage

How to build and run `bootstrap`. See the [README](../README.md) for an
overview and [orchestrator.md](orchestrator.md) for the design.

## Prerequisites

- Rust toolchain (edition 2024; `rustc` ≥ 1.85) via [rustup](https://rustup.rs/).
- The `opencode` CLI on your `$PATH`.
- A model reachable by opencode. The defaults target a local `llama-server`
  (OpenAI-compatible endpoint) configured in `opencode.json` as the
  provider/model `llamacpp/qwable`.

## Building

```bash
cargo build --release
# binary: target/release/bootstrap
```

## `orchestrate` — plan / execute / review from a spec

Give it a spec file (goal + description + architectural decisions). The model
designs a plan, then implements and reviews it step by step.

```bash
bootstrap orchestrate path/to/spec.txt \
  --model llamacpp/qwable \
  --work-dir /path/to/target/project \
  --max-steps 20 \
  --verbose
```

| Option | Description | Default |
|--------|-------------|---------|
| `<SPEC_FILE>` | Spec file path | (required) |
| `--model` | Model in `provider/model` form | `llamacpp/qwable` |
| `--work-dir` | Directory the agent works in; `.bootstrap/` artifacts land here | `.` |
| `--max-steps` | Safety cap on total executed steps | `20` |
| `--plan-only` | Produce + print the plan, then stop (cheap planner test) | off |
| `--verbose` | Verbose logging | off |

Cheap dry run of just the planner:

```bash
bootstrap orchestrate path/to/spec.txt --plan-only
```

After a run, inspect `<work-dir>/.bootstrap/`: `plan.json`,
`step-NN-result.json`, and `step-NN-review.json`.

### Writing a spec

Plain text with a goal, a description, and the architectural decisions to honor.
See `examples/spec-greeter.txt`. Keep decisions explicit (language/edition,
dependencies, structure, tests) — the planner turns each into steps and
acceptance criteria.

## `exec` — run an explicit task list

Lower-level mode: one `opencode run` per task. By default tasks chain into a
single session so later tasks can reference earlier work; `--no-chain` isolates
them.

```bash
bootstrap exec "create a test" "run the test" "validate the results" \
  --model llamacpp/qwable --verbose
```

## Demo scripts

```bash
PLAN_ONLY=1 ./scripts/demo-orchestrate.sh   # one cheap planner call
./scripts/demo-orchestrate.sh               # full loop in an isolated temp workspace
./scripts/demo.sh                           # single tool-using exec call
./scripts/demo-3-steps.sh                   # 3 chained exec tasks
```

## Troubleshooting

- **`opencode` not found:** ensure it's installed and on your `$PATH`.
- **Model not found / connection refused:** confirm `llama-server` is running and
  `opencode.json` defines the `llamacpp/qwable` provider/model. Check with
  `opencode models`.
- **Many steps report `unknown` / an early `stop`:** the executor did the work
  but didn't emit the JSON result envelope, so the reviewer couldn't confirm it.
  Re-run, or inspect the `.bootstrap/step-NN-result.json` files. This is a known
  limitation tracked in [plan.md](plan.md).
- **Build errors:** verify `rustc --version` ≥ 1.85 (edition 2024).
