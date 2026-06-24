# local-llm-loop

A small Rust harness that drives [opencode](https://opencode.ai) against a
**local** LLM (e.g. Qwable-v1 served by `llama-server`) in an autonomous
**plan → execute → review** loop.

Give it a short spec file — a goal plus architectural decisions — and it asks
the model to design a plan, then iterates the steps: a tool-using agent
implements each step and a supervising model reviews the result and decides
whether to continue, insert new steps, skip a step, or stop and request human
intervention.

> The binary and crate are named `bootstrap`; the git repository is
> `local-llm-loop`.

## Overview

```
spec.txt
   │
   ▼
[PLAN]   planner LLM → { design, notes, steps[] }              (.bootstrap/plan.json)
   │
   ▼
for each step (Rust owns the cursor + history):
   ├─ [EXECUTE]  tool-using LLM performs the step, emits a JSON result envelope
   │             { summary, new_or_changed_files[], changes, features[],
   │               bugs[], issues[], status }                  (.bootstrap/step-NN-result.json)
   └─ [REVIEW]   supervising LLM sees the result + remaining steps and returns:
                   continue | insert | skip | stop             (.bootstrap/step-NN-review.json)
```

Rust owns all orchestration state; every LLM call is a stateless `opencode run`
with the needed context passed in the prompt. Structured hand-offs use
sentinel-delimited JSON that the harness extracts robustly. See
[docs/orchestrator.md](docs/orchestrator.md) for the full design and the
annotated end-to-end runs ([macOS](docs/mac-poc-summary.md) ·
[Arch/NVIDIA](docs/arch-poc-summary.md)).

## Prerequisites

- A Rust toolchain (edition 2024; `rustc` ≥ 1.85) via [rustup](https://rustup.rs/).
- The `opencode` CLI on your `$PATH`.
- A model reachable by opencode. The defaults target a local `llama-server`
  exposing an OpenAI-compatible endpoint, configured in `opencode.json` as the
  provider/model `llamacpp/qwable`. Ready-to-use server and opencode configs for
  macOS and Arch/NVIDIA, plus the compatibility matrix, are in
  [docs/config/](docs/config/README.md).

## Build

```bash
cargo build --release
# binary: target/release/bootstrap
```

## Usage

The binary has two subcommands.

### `orchestrate` — plan/execute/review from a spec file

```bash
bootstrap orchestrate path/to/spec.txt \
  --model llamacpp/qwable \
  --work-dir /path/to/target/project \
  --max-steps 20 \
  --verbose
```

| Option | Description | Default |
|--------|-------------|---------|
| `<SPEC_FILE>` | Spec file: goal + description + architectural decisions | (required) |
| `--model` | Model in `provider/model` form | `llamacpp/qwable` |
| `--work-dir` | Directory the agent works in; `.bootstrap/` artifacts go here | `.` |
| `--max-steps` | Safety cap on total executed steps | `20` |
| `--plan-only` | Produce and print the plan, then stop (cheap planner test) | off |
| `--verbose` | Verbose logging | off |

### `exec` — run an explicit list of tasks

```bash
bootstrap exec "create a test" "run the test" "validate the results" \
  --model llamacpp/qwable --verbose
```

Each task is a separate `opencode run`; by default they chain into one session
(`--no-chain` to isolate them).

### Demo scripts

```bash
PLAN_ONLY=1 ./scripts/demo-orchestrate.sh   # one cheap planner call
./scripts/demo-orchestrate.sh               # full loop in an isolated temp workspace
./scripts/demo.sh                           # single tool-using exec call
./scripts/demo-3-steps.sh                   # 3 chained exec tasks
```

## Documentation

- [docs/orchestrator.md](docs/orchestrator.md) — control flow and rationale
- [docs/architecture.md](docs/architecture.md) — components and data flow
- [docs/design.md](docs/design.md) — CLI, JSON contracts, sentinels, artifacts
- [docs/prd.md](docs/prd.md) — product requirements
- [docs/plan.md](docs/plan.md) — roadmap and current status
- [docs/usage.md](docs/usage.md) — build and run instructions
- [docs/config/](docs/config/README.md) — local model setup: `llama-server` + `opencode.json`, compatibility matrix, macOS & Arch/NVIDIA
- [docs/mac-poc-summary.md](docs/mac-poc-summary.md) — annotated end-to-end proof-of-concept run (macOS / Apple Silicon)
- [docs/arch-poc-summary.md](docs/arch-poc-summary.md) — annotated end-to-end proof-of-concept run (Arch Linux / NVIDIA)
- [docs/performance-analysis.md](docs/performance-analysis.md) — throughput benchmarks and RTX 3090 vs M1 Max analysis
- [docs/older-hardware.md](docs/older-hardware.md) — older 24 GB GPUs (Tesla M40 / K80) for an overnight batch fleet
- [docs/fleet-strategy.md](docs/fleet-strategy.md) — deploying across heterogeneous nodes (GPU/CPU tiers, scheduling, power/$) for overnight batch runs

## License

Released under the [MIT License](LICENSE).

Copyright (c) 2026 Michael A. Wright
