# local-llm-loop

A small Rust harness that drives [opencode](https://opencode.ai) against a
**local** LLM in an autonomous **plan → execute → review** loop.

Give it a short spec file — a goal plus architectural decisions — and it asks
the model to design a plan, then iterates the steps: a tool-using agent
implements each step and a supervising model reviews the result and decides
whether to continue, insert new steps, skip a step, or stop and request human
intervention.

The harness is **model-agnostic** — it shells out to `opencode run --model
<provider/model>`, so any model opencode can reach works.

## Which model on which box

The table below groups **measured loop results by GPU tier**, smallest → largest
VRAM. They come from a growing **test fleet of 8–9 machines** (several share a GPU
type but differ in CPU/RAM/OS, and several are owned but not yet benched) — see
**[test-fleet.md](docs/test-fleet.md)** for the full inventory and per-box role.
Each box runs the biggest model it can hold; pick by **loop wall-clock** —
measured time to a `cargo test`-green crate (*quality* ranking is future work;
this is **speed only**). All served by a local `llama-server`, model loaded once
and reused across every call.

**Priorities:** 24 GB is the sweet spot (holds a 30–35B MoE coder whole — primary
focus); 16 GB is great where it works (5060 Ti ✅, A2-16 GB next); **12 GB is
weak for *coding*** and may be better aimed at **admin / non-coding** agent tasks;
Apple Silicon spans M1 8 GB → M1 Max 64 GB.

| Box (VRAM) | Best models, by loop wall-clock | Measured? |
|------------|---------------------------------|-----------|
| **RTX 3060 · 12 GB** | **Ornith-1.0-9B+MTP 4m29s** › Qwen3-Coder-30B 7m20s › Ornith-1.0-9B 10m58s › Qwen3.6-35B 13m40s | ✅ **fully loop-timed** |
| **RTX 5060 Ti · 16 GB** | **gpt-oss-20b 1m13s** › Ornith-1.0-9B+MTP 2m15s › Ornith-1.0-9B 4m42s › Qwen3-Coder-30B 6m38s › Gemma-4-26B 9m02s › Ornith-1.0-35B 9m56s | ✅ **fully loop-timed** |
| **RTX 3090 · 24 GB** | **Qwen3-Coder-30B ~3m30s**; also Gemma-4-26B (*best code*), Qwen3.6-35B-MTP, Gemma-4-31B+E2B draft | qwen3-coder loop-timed; rest throughput-verified |
| **M1 Max · 64 GB** | **gpt-oss-20b 3m34s** › Ornith-1.0-35B (MLX) 6m06s › Qwen3.6-35B-MTP 8m01s | ✅ loop-timed; holds *any* model |

Notes: **12 GB** runs the dense 9B Ornith whole (+ its MTP head) and A3B MoE coders
≤35B via `--n-cpu-moe` (experts→RAM); **16 GB** runs MoE coders ≤30B (light
`--n-cpu-moe`) + the dense 9B Ornith; **24 GB** fits 30–35B whole; **64 GB** fits
anything (and MLX is ~1.35× GGUF decode on Apple Silicon). The **same model is
faster on a bigger/faster box** (Qwen3-Coder: ~3m30s on the 3090 vs 6m38s on the
5060; gpt-oss: 1m13s on the 5060 vs 3m34s on the M1 Max). Per-box detail:
[5060](docs/nvidia-5060-poc-summary.md) · [3090](docs/nvidia-3090-poc-summary.md) ·
[3060 (measured)](docs/nvidia-3060-12-results.md) · [M1 Max](docs/mac-poc-summary.md).

**August 2026 candidates:** Laguna S 2.1 and Nemotron 3.5 Lightning 30B-A3B have
now been attempted on the M1 Max. Laguna stalls in step 2 and leaves a
non-compiling crate after 13m03s; Nemotron passes native tool use but fails all
three structured-plan attempts. Neither is in the successful ranking.
Qwen3.8-27B remains queued. See the
[evaluation plan and failure records](docs/plan-august-2026-models.md).

### Every measured loop run — slowest → fastest

All boxes, all models that produced a `cargo test`-green crate, by measured loop
wall-clock. Reads top-down **slow → fast**; the **box** column is the HW tier
(**low → high VRAM: 3060 12 GB ‹ 5060 Ti 16 GB ‹ 3090 24 GB ‹ M1 Max 64 GB**).

| Model | Box (VRAM tier) | Placement | **Loop wall-clock** |
|-------|-----------------|-----------|--------------------:|
| Qwen3.6-27B (dense) | 5060 Ti · 16 GB | offload | ~75 min 🐌 |
| Qwopus3.6-27B-Coder (dense) + MTP | M1 Max · 64 GB | whole + MTP | 19m43s |
| Qwen3.6-35B-A3B (no MTP) | **3060 · 12 GB** | `--n-cpu-moe` | 13m40s |
| Ornith-1.0-9B | **3060 · 12 GB** | resident | 10m58s |
| Ornith-1.0-35B | 5060 Ti · 16 GB | `--n-cpu-moe` | 9m56s |
| Gemma-4-26B-A4B | 5060 Ti · 16 GB | `--n-cpu-moe` | 9m02s |
| Qwen3.6-35B-A3B-MTP | M1 Max · 64 GB | whole + MTP | 8m01s |
| Qwen3-Coder-30B-A3B | **3060 · 12 GB** | `--n-cpu-moe` | 7m20s |
| Qwen3-Coder-30B-A3B | 5060 Ti · 16 GB | `--n-cpu-moe` | 6m38s |
| Ornith-1.0-35B (MLX) | M1 Max · 64 GB | whole | 6m06s |
| Ornith-1.0-9B | 5060 Ti · 16 GB | resident | 4m42s |
| **Ornith-1.0-9B + MTP** | **3060 · 12 GB** | resident + MTP | **4m29s** |
| gpt-oss-20b | M1 Max · 64 GB | whole | 3m34s |
| Qwen3-Coder-30B-A3B | 3090 · 24 GB | whole | ~3m30s |
| **Ornith-1.0-9B + MTP** | 5060 Ti · 16 GB | resident + MTP | **2m15s** |
| **gpt-oss-20b** MXFP4 | 5060 Ti · 16 GB | whole (FP4) | **1m13s** 🥇 |

Two things the table makes plain: a **fast small model can beat a slow big one across
tiers** — Ornith-9B+MTP on the *lowest* box (4m29s) edges Ornith-9B on the 5060 and
nearly matches gpt-oss on the M1 Max — and on the **12 GB 3060 the spread is 3×**
(4m29s → 13m40s), driven by **MTP + how many tokens the model thinks**, not raw size.

**Two gates a model must clear** (hardware-independent): **native tool-calling
llama.cpp parses** + **strict-JSON output**. Larger MoE/dense coders (≥26B) clear
both; smaller *general* models often fail one — Qwen2.5-Coder (tool-call
delimiter), Qwen3-8B / Qwen3-14B (JSON), Phi-4 (no tool calls) — while a
*purpose-trained* coder like **Ornith-1.0-9B clears both at 9B**. gpt-oss-20b
fails the JSON gate unaided but the harness's `emit` self-heal carries it through.
See the [3090 POC](docs/nvidia-3090-poc-summary.md) for the full model-search
matrix and [performance-analysis.md](docs/performance-analysis.md) for throughput.

> The binary and crate are named `bootstrap`; the git repository is
> `local-llm-loop`.

## How long does the loop take? — where the time goes

Wall-clock for the same greeter spec (plan → code → review → repeat). The
**per-phase numbers are measured on the M1 Max** (recovered from `.bootstrap/`
artifact timestamps); the orchestrator doesn't log durations and the GPU POCs
recorded **throughput, not phase timing**, so the cross-system table ranks by the
measured throughput that *drives* wall-clock (see
[performance-analysis.md](docs/performance-analysis.md)).

**Per-phase wall-clock — M1 Max (measured):**

| Phase | Qwen3.6-35B-A3B-MTP | gpt-oss-20b |
|-------|--------------------:|------------:|
| Model load → ready | ~4 s | ~4 s |
| **Plan** (1 call) | 1m32s | 1m35s |
| **Code** (per step, avg) | ~78 s | ~36 s |
| **Review** (per step, avg) | ~51 s | ~24 s |
| Retries / self-heals | **0** | 6 retries + 1 `emit` fix |
| **Total loop** | **8m01s** (3 steps) | **3m34s** (2 steps) |

**Cross-system ranking (by measured throughput — the wall-clock driver):**

| Target system | gpt-oss-20b decode / prefill | Qwen3.6-35B decode / prefill | Loop speed rank |
|---------------|------------------------------:|------------------------------:|:---------------:|
| **RTX 3090** (24 GB) | ~205 / ~5,652 t/s | ~143 / ~3,365 t/s (MTP) | 🥇 fastest |
| **RTX 5060 Ti** (16 GB) | 138 / 5,920 t/s | — | 🥈 |
| **M1 Max** (64 GB) | 76 / 998 t/s | 57 / 827 t/s (MTP) | 🥉 (but holds any model whole) |

**Where the time actually goes (per model / hardware):**

- **Model load is not the bottleneck** — on the M1 Max both the 21 GB MoE and the
  11 GB MoE reach *ready* in ~4 s (mmap + unified memory); model size barely moves
  it. On the GPUs, load is dominated by VRAM transfer but still seconds, not the
  loop cost.
- **Plan is the single most expensive call** (~1.5 min here) — it's a cold prompt
  cache and the longest single generation (full design + step list). Same on every
  box; it just scales with decode speed.
- **"Thinking" vs. doing splits the models.** Qwen3.6-MTP writes more per step
  (slower decode + longer code → ~78 s/step) but lands it in **one attempt, every
  time (0 retries)** — the clean, deliberate run. gpt-oss-20b is faster per call
  (~36 s) but **wobbles on the JSON envelope: 6 retries + 1 `emit` self-correction**
  — the same gate-2 behavior the 3090/5060 POCs saw, here absorbed by the
  self-healing `emit` channel rather than failing the loop.
- **Tool-use reliability is the real time sink, not raw tok/s.** The 3090/5060
  POCs show the same split: **Gemma-4-26B = 0 retries (cleanest)**, gpt-oss-20b
  needs `emit` rescues, qwen3-8b needed **12 in-context-RL re-prompts**, and dense
  coders (Qwen3.6-27B, Devstral-24B) burn the clock on *generation* — Devstral
  spent **>39 min on the plan alone**. A model that one-shots each envelope beats a
  faster model that retries.

**Net:** raw speed ranks **3090 > 5060 Ti > M1 Max**, but per-model the deciding
factor is **attempts-per-phase**: the fewer retries and self-heals a model needs,
the shorter the loop — independent of the box.

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
with the needed context passed in the prompt. Each role writes its JSON envelope
to a file and validates it with the **`bootstrap emit`** helper (a CLI on the
model's `$PATH`), which checks it against the harness schema and returns a
correctable error — making the envelope channel self-healing (see
[`emit`](#emit--validate-a-json-envelope-the-model-facing-helper) below). The
harness also extracts JSON robustly from whatever the model produces. See
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

The binary has three subcommands.

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

### `emit` — validate a JSON envelope (the model-facing helper)

```bash
bootstrap emit <plan|step|review> --file path/to/envelope.json
```

A plain CLI helper (on `$PATH`, **not** MCP) that the **model** calls from
opencode's shell tool. It validates the JSON the model just wrote against the
harness's own schema, rewrites it canonically on success, and prints `OK:` or a
precise, correctable `Error:` with the expected shape:

```text
$ bootstrap emit review --file step-02-review.llm.json
Error: review envelope invalid: missing field `action` …
expected shape:
{"action":"continue|insert|skip|stop","reason":"...","new_steps":[]}
```

The role prompts tell each LLM to write its envelope to a file, run `emit`, fix
on any `Error:`, and only reply `DONE` after `OK:`. This makes the envelope
channel **self-healing** — a model that first emits a malformed or wrong-schema
envelope (e.g. a `todowrite` list with no `action`) sees the error and corrects
it instead of silently breaking the loop. It is what lets gpt-oss-20b (which
failed the JSON gate on the 3090) complete the loop on the 5060. The worked
shell invocation in the prompt also generalizes the model to calling other CLI
tools. `emit` exits non-zero on failure, so the model's shell sees the error.

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
- [docs/nvidia-3060-12-results.md](docs/nvidia-3060-12-results.md) — **measured** 12 GB RTX 3060 loop wall-clocks: 4/4 models green — Ornith-9B+MTP (**4m29s**), Qwen3-Coder-30B (7m20s), Ornith-9B (10m58s), Qwen3.6-35B (13m40s); MTP head ~1.3–1.7× decode; offloaded A3B MoE beats the resident dense 9B
- [docs/nvidia-3090-poc-summary.md](docs/nvidia-3090-poc-summary.md) — RTX 3090 fast-GPU-resident coder run (Qwen3-Coder-30B-A3B MoE); the model-search matrix + speculative-decoding findings
- [docs/nvidia-5060-poc-summary.md](docs/nvidia-5060-poc-summary.md) — RTX 5060 Ti 16 GB (Blackwell/FP4) run: **4 models produce working code**, by loop wall-clock — gpt-oss-20b (**1m13s**), Ornith-1.0-9B (4m42s, a purpose-trained 9B coder), Qwen3-Coder-30B (6m38s), Gemma-4-26B-A4B (9m02s); dense ≥27B build but are too slow (Qwen3.6-27B ~75m, Devstral-24B); general ≤14B can't code. Native MXFP4 prefill **beats the 3090** on gpt-oss-20b
- [docs/plan-rtx5060-16.md](docs/plan-rtx5060-16.md) — plan: testing small coders (±spec-decode) on a 16 GB RTX 5060 (Blackwell / FP4)
- [docs/plan-rtx3060-12.md](docs/plan-rtx3060-12.md) — plan: testing small coders (±spec-decode) on a 12 GB RTX 3060 (Ampere + RAM offload)
- [docs/performance-analysis.md](docs/performance-analysis.md) — throughput benchmarks: RTX 3090 vs RTX 3060 (CPU-offload) vs M1 Max
- [docs/older-hardware.md](docs/older-hardware.md) — older 24 GB GPUs (Tesla M40 / K80) for an overnight batch fleet
- [docs/fleet-strategy.md](docs/fleet-strategy.md) — deploying across heterogeneous nodes (GPU/CPU tiers, scheduling, power/$) for overnight batch runs
- [docs/glm-models.md](docs/glm-models.md) — pointing the harness at GLM-5.2 (3-bit, big-RAM CPU + A2 offload)
- [docs/plan-ornith-models.md](docs/plan-ornith-models.md) — Ornith-1.0 family (Qwen-3.5/Gemma-4 agentic coders): GGUF/MLX/MTP ecosystem, per-system fit, and measured M1 Max MLX-vs-GGUF results
- [docs/future-research.md](docs/future-research.md) — deferred investigations: DeepSeek-V4/DSpark self-speculation, and speculative decoding with an offloaded target on the RTX 3090 + Xeon box

## License

Released under the [MIT License](LICENSE).

Copyright (c) 2026 Michael A. Wright
