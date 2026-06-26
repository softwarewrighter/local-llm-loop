# local-llm-loop

A small Rust harness that drives [opencode](https://opencode.ai) against a
**local** LLM in an autonomous **plan → execute → review** loop.

Give it a short spec file — a goal plus architectural decisions — and it asks
the model to design a plan, then iterates the steps: a tool-using agent
implements each step and a supervising model reviews the result and decides
whether to continue, insert new steps, skip a step, or stop and request human
intervention.

The harness is **model-agnostic** — it shells out to `opencode run --model
<provider/model>`, so any model opencode can reach works. It started against
Qwable-v1 but has since driven the loop to a verified, working Rust CLI with
several local models. **Verified-working models** (complete the loop and produce
code that passes `cargo test`), measured on an RTX 3090:

| Model | Notes |
|-------|-------|
| **Gemma-4-26B-A4B-it** | MoE, best greeter quality (`Hello, X!`) — see [3090 POC](docs/nvidia-3090-poc-summary.md) |
| **Qwen3-Coder-30B-A3B-Instruct** | MoE, fastest verified coder (~189 tok/s decode) |
| **Qwen3.6-35B-A3B-MTP** | MoE with built-in **MTP self-speculative decode** (85% accept, no separate draft) |
| **Qwen3.6-27B** | dense; cleanest output structure (pure `lib.rs` + `main.rs`) |
| **Gemma-4-31B-it + Gemma-4-E2B draft** | dense + speculative decoding (exact-vocab pair); the verified dense spec-decode pair |
| **Qwable-v1 IQ4_XS** | the original baseline |

A model needs two things to work here: **native tool-calling that llama.cpp
parses** and **strict-JSON output**. Larger MoE/dense coders (≥26B) clear both;
several otherwise-capable models fail one gate — Qwen2.5-Coder (tool-call
delimiter), gpt-oss-20b and Granite-4.1-8B (JSON), Qwen3-14B (JSON). The
[3090 POC](docs/nvidia-3090-poc-summary.md) has the full model-search matrix and
[performance-analysis.md](docs/performance-analysis.md) the throughput comparison.

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
- [docs/arch-poc-summary.md](docs/arch-poc-summary.md) — annotated end-to-end proof-of-concept run (Arch Linux / NVIDIA RTX 3090, full GPU)
- [docs/arch-nvidia-3060-poc-summary.md](docs/arch-nvidia-3060-poc-summary.md) — end-to-end run on a 12 GB RTX 3060 with MoE experts offloaded to system RAM
- [docs/nvidia-3090-poc-summary.md](docs/nvidia-3090-poc-summary.md) — RTX 3090 fast-GPU-resident coder run (Qwen3-Coder-30B-A3B MoE); the model-search matrix + speculative-decoding findings
- [docs/plan-rtx5060-16.md](docs/plan-rtx5060-16.md) — plan: testing small coders (±spec-decode) on a 16 GB RTX 5060 (Blackwell / FP4)
- [docs/plan-rtx3060-12.md](docs/plan-rtx3060-12.md) — plan: testing small coders (±spec-decode) on a 12 GB RTX 3060 (Ampere + RAM offload)
- [docs/performance-analysis.md](docs/performance-analysis.md) — throughput benchmarks: RTX 3090 vs RTX 3060 (CPU-offload) vs M1 Max
- [docs/older-hardware.md](docs/older-hardware.md) — older 24 GB GPUs (Tesla M40 / K80) for an overnight batch fleet
- [docs/fleet-strategy.md](docs/fleet-strategy.md) — deploying across heterogeneous nodes (GPU/CPU tiers, scheduling, power/$) for overnight batch runs
- [docs/glm-models.md](docs/glm-models.md) — pointing the harness at GLM-5.2 (3-bit, big-RAM CPU + A2 offload)

## License

Released under the [MIT License](LICENSE).

Copyright (c) 2026 Michael A. Wright
