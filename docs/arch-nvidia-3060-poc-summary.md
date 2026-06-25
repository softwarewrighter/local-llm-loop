# Proof-of-Concept Summary — Arch Linux / NVIDIA RTX 3060 12 GB (CPU-offload)

A full end-to-end run of the planner → executor → reviewer orchestrator
(`bootstrap orchestrate`), building a small Rust CLI from a spec file with no
human authoring of code — reproduced on a **12 GB** consumer GPU that **cannot
hold the model**, with the MoE experts offloaded to system RAM. This is the same
hybrid GPU + CPU/RAM regime [GLM-5.2](glm-models.md) will use, validated small.
The 24 GB / full-GPU counterpart is [arch-poc-summary.md](arch-poc-summary.md)
(RTX 3090) and the Apple-Silicon one is [mac-poc-summary.md](mac-poc-summary.md).

- **Date:** 2026-06-24
- **Model (all three roles):** `llamacpp/qwable` (Qwable-v1 IQ4_XS) on local `llama-server`
  — `qwen35moe`, **34.66B total / ~3B active** (256 experts, 8 used per token, 40 layers)
- **Host:** Arch Linux · **RTX 3060 12 GB** (CUDA 13.0, driver 580.95.05) ·
  Xeon E5-2697 v4 (2×18c / 72 threads, AVX2) · **503 GB RAM** ·
  llama.cpp CUDA build **b9784** (`8be759e`)
- **Server settings:** `-ngl 99 --n-cpu-moe 18 --reasoning-budget 0 --ctx-size 32768
  --parallel 1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --jinja
  --n-predict 4096` (see [config/arch-nvidia-3060](config/arch-nvidia-3060/start-qwable.sh))
- **VRAM used:** **~11.2 GB / 12 GB** — the GPU holds the dense/attention weights,
  the KV cache, and the experts of **22 of 40** layers; the other 18 layers'
  experts (and the rest of the 17.6 GiB model) live in the 503 GB system RAM.
- **Throughput (`llama-bench`, this config):** **prefill `pp512` ≈ 413 tok/s,
  decode `tg128` ≈ 49 tok/s** (speculative decoding off).
- **opencode:** **1.17.10** — the harness was **adapted to opencode 1.x** for this
  run (see "Harness adaptation" below).
- **Command:** `./scripts/demo-orchestrate.sh` (builds in an isolated temp workspace)
- **Outcome:** working, tested Rust CLI produced; loop halted with a **legitimate
  completion** (plan exhausted after a Continue; verification confirms it works).

## Input

Identical spec to the other runs — `examples/spec-greeter.txt` (a `greet` CLI:
required `NAME`, optional `--times N` default 1, reject `--times 0`, pure
unit-tested `greet()`, clap-only). See
[mac-poc-summary.md#input](mac-poc-summary.md#input) for the full text.

## Harness adaptation (what's different here besides the GPU)

The earlier POCs ran against **opencode 0.2.x**, whose `run` behaved like a plain
completion: the model's JSON came back on stdout and the harness scraped it. This
box was forced onto **opencode 1.17.10** (0.2.x throws a `DecimalError` against the
newer llama.cpp server), and 1.x is a **full agentic harness** — it prefers tool
calls over inline prose. Two changes made the loop work:

1. **File-based JSON hand-off.** Each role is now told to **write its JSON to a
   known path** (e.g. `.bootstrap/plan.llm.json`) instead of printing it; the
   harness reads that file (falling back to scraping stdout). This leans into
   opencode 1.x's natural file-writing rather than fighting it, and yields clean
   JSON with no reasoning preamble. All three roles use the default `build` agent
   (the read-only/custom agents hung on this box).
2. **`--reasoning-budget 0`.** This Qwable build is *reasoning-distilled*; left
   unbounded it spent the entire `--n-predict` budget in a `<think>` stream before
   answering. Budgeting it to 0 lets the answer (and JSON) through.

These changes are model/opencode-version specific, not GPU-specific — but they're
what it took to reproduce the loop on a current software stack.

## Stage 1 — Plan (one LLM call)

The planner produced a design + 5-step plan (`.bootstrap/plan.json`):

> **Design:** A single Rust 2024 binary crate with clap (derive) for argument
> parsing. Core logic is a pure function `greet(name: &str, times: usize) ->
> String` that returns the repeated greeting. `main()` parses CLI args via clap,
> validates `times >= 1`, calls `greet()`, and prints the result. Unit tests
> exercise `greet()` for times=1 and times=3. Minimal dependencies: only clap.

| # | Planned step |
|---|---|
| 1 | Create project scaffold and Cargo.toml |
| 2 | Implement argument parsing |
| 3 | Implement pure greet function |
| 4 | Add unit tests |
| 5 | Validate CLI behavior end-to-end |

## Stage 2 — Execute + Review loop

The executor folded the unit tests into step 3, so the reviewer **contracted** the
plan — recognizing already-satisfied work and skipping the redundant step. 4 of
the 5 steps executed; step 4 was skipped.

| Step | Title | Executor result | Review decision |
|------|-------|-----------------|-----------------|
| 1 | Scaffold + Cargo.toml | completed (`cargo new greet`, added clap derive, `cargo check`) | Continue |
| 2 | Argument parsing | completed (`GreetArgs` clap derive: positional `name`, `--times`, reject 0) | Continue |
| 3 | Pure `greet()` | completed (added `greet()` + **unit tests for times 1 & 3**; `cargo test`) | **Skip** — "tests already added in step 3; step 4 redundant" |
| 4 | Add unit tests | *(skipped — no execution)* | — |
| 5 | Validate end-to-end | completed (ran binary for valid name, `--times 0`, `--times 2`; checked exit codes) | **Continue** — last step, plan exhausted |

**Review action tally:** 3 Continue · 1 Skip · 0 Insert · 0 Stop.

Every executed step emitted a well-formed result envelope (via its
`.bootstrap/step-NN-result.llm.json` file) — **0 steps reported `unknown`**. Each
result/review is persisted as `.bootstrap/step-NN-result.json` /
`.bootstrap/step-NN-review.json`.

> Minor note: the executor ran `cargo new greet`, so the crate landed in a
> `greet/` subdirectory of the workspace (a few `Read .` probes of the workspace
> root failed before the agent looked in `greet/`). It self-corrected each time;
> the result is correct, just nested one level down.

## Output — the generated CLI

`greet/src/main.rs` (verbatim, model-authored):

```rust
use clap::Parser;

/// Simple greeting CLI
#[derive(Parser, Debug)]
#[command(name = "greet")]
struct GreetArgs {
    /// The name to greet
    name: String,

    /// Number of times to greet
    #[arg(long, default_value = "1")]
    times: usize,
}

/// Generate a greeting repeated `times` times.
pub fn greet(name: &str, times: usize) -> String {
    let line = format!("Hello, {}!", name);
    (0..times).map(|_| line.clone()).collect::<Vec<_>>().join("\n")
}

fn main() {
    let args = GreetArgs::parse();
    if args.times == 0 {
        eprintln!("Error: times must be at least 1");
        std::process::exit(1);
    }
    println!("{}", greet(&args.name, args.times));
}

#[cfg(test)]
mod tests {
    use super::greet;

    #[test]
    fn greet_once() {
        assert_eq!(greet("World", 1), "Hello, World!");
    }

    #[test]
    fn greet_multiple() {
        assert_eq!(
            greet("World", 3),
            "Hello, World!\nHello, World!\nHello, World!",
        );
    }
}
```

`greet/Cargo.toml`: edition 2024, single dependency `clap = { features = ["derive"] }`.

### Independent verification (performed by a human, not the model's claim)

Run against the generated crate after the loop halted:

- `cargo test` → **2 passed** (`greet_once`, `greet_multiple`)
- `greet Alice` → `Hello, Alice!`, exit **0**
- `greet Bob --times 2` → two greeting lines, exit **0**
- `greet X --times 0` → `Error: times must be at least 1`, exit code **1**

The project met the spec: pure `greet()` with I/O isolated in `main()`, both
required tests, clap-only, required positional `name`.

## How it compares (3060 12 GB vs 3090 24 GB vs M1 Max)

Same model, same spec, same loop — but this is the **only** run where the model
does **not** fit in the accelerator's memory, so it's a different performance
regime (see [performance-analysis.md](performance-analysis.md) for the full
breakdown and the apples-to-apples `llama-bench` rows).

| | RTX 3090 (24 GB) | **RTX 3060 (12 GB)** | M1 Max |
|---|---|---|---|
| Model placement | **whole model in VRAM** | **experts in RAM**, 22/40 expert layers + dense/KV on GPU | whole model in unified memory |
| Prefill (`pp512`) | ~3,605 tok/s | **~413 tok/s** | ~250–400 tok/s (pred.) |
| Decode (`tg128`) | ~158 tok/s | **~49 tok/s** | ~65–70 tok/s (pred.) |
| Loop outcome | works (Continue/Skip/Stop) | works (Continue/Skip) | works (Continue/Insert/Stop) |

- **Decode (~49 tok/s)** is RAM-bandwidth-bound for the CPU-resident experts —
  ~3× slower than the 3090 and a bit under the M1 Max, whose unified memory holds
  the whole model. Pushing more expert layers onto the GPU (lower `--n-cpu-moe`)
  recovers decode toward the GPU ceiling: all-experts-on-CPU measures **~33
  tok/s**, the tuned 22-on-GPU config **~49 tok/s**.
- **Prefill (~413 tok/s)** takes the larger hit — ~9× slower than the 3090.
  Prefill is compute-bound and the experts that live in RAM are computed on the
  CPU, so the 3060's tensor cores only accelerate the layers they hold. This is
  the cost of not fitting in VRAM.
- **Net:** a 12 GB consumer card runs this 35B-MoE perfectly well for
  *correctness* (identical loop behavior and output quality), at a **decode** rate
  in the M1 Max's league and a **prefill** rate well below the 3090's. Good enough
  for interactive small tasks; for big-prompt agentic turns the prefill gap shows.

This is exactly the behavior the GLM-5.2 plan predicts at larger scale: the GPU
accelerates only what it holds, the experts live in RAM, and decode is bounded by
RAM bandwidth.

## Reproduce

```bash
# 1. Build llama.cpp with CUDA (sm_86), then start the server:
docs/config/arch-nvidia-3060/start-qwable.sh        # -ngl 99 --n-cpu-moe 18 --reasoning-budget 0, port 8081

# 2. opencode.json: llamacpp/qwable provider at http://localhost:8081/v1,
#    apiKey "local", and a zero "cost" block (opencode 1.x cost-calc).

# plan only (one cheap call):
PLAN_ONLY=1 ./scripts/demo-orchestrate.sh

# full loop in an isolated temp workspace:
./scripts/demo-orchestrate.sh
```
