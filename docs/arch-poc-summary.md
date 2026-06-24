# Proof-of-Concept Summary — Arch Linux / NVIDIA

A full end-to-end run of the planner → executor → reviewer orchestrator
(`bootstrap orchestrate`) against the local LLM, building a small Rust CLI from a
spec file with no human authoring of code — reproduced on Arch Linux with an
NVIDIA GPU. The macOS / Apple-Silicon counterpart is
[mac-poc-summary.md](mac-poc-summary.md).

- **Date:** 2026-06-24
- **Model (all three roles):** `llamacpp/qwable` (Qwable-v1 IQ4_XS) on local `llama-server`
- **Host:** Arch Linux, NVIDIA RTX 3090 (24 GB), llama.cpp CUDA build (llama-server 9728)
- **Server settings:** `--ctx-size 32768 --parallel 1 --gpu-layers 99 --flash-attn on
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja --n-predict 4096`
  (the VRAM-fitted Arch profile — see [config/arch-nvidia](config/arch-nvidia/start-qwable.sh))
- **Command:** `./scripts/demo-orchestrate.sh` (builds in an isolated temp workspace)
- **Generation rate observed:** ~132 tok/s sustained (peaks ~144); prompt eval
  ~2000–2600 tok/s. Speculative decoding was **not** active this run.
- **Outcome:** working, tested Rust CLI produced; loop halted with a **legitimate
  Stop** (verification confirmed completion — not a false alarm)

## Input

Identical spec to the macOS run — `examples/spec-greeter.txt` (a `greet` CLI:
required `NAME`, optional `--times N` default 1, reject `--times 0`, pure
unit-tested `greet()`, clap-only). See
[mac-poc-summary.md#input](mac-poc-summary.md#input) for the full text.

## Stage 1 — Plan (one LLM call)

The planner produced a design + 5-step plan (`.bootstrap/plan.json`):

> **Design:** A single binary crate (`greet`) with a pure
> `greet(name: &str, times: usize) -> String` that formats N copies of
> "Hello, {name}!" separated by newlines. Clap (derive) defines the CLI:
> positional `name` (required) and optional `--times` (default 1). `main` parses
> args, calls `greet`, prints the result, and exits with an error on invalid
> input (e.g. `times == 0`). A `greet_tests` module unit-tests times 1 and 3.

| # | Planned step |
|---|---|
| 1 | Initialize Rust project and add clap dependency |
| 2 | Define CLI argument struct and main entry point |
| 3 | Implement pure `greet` function |
| 4 | Add unit tests for `greet` |
| 5 | Wire `greet` into main and ensure error handling |

## Stage 2 — Execute + Review loop

The executor implemented the whole crate in step 1, so the reviewer **contracted**
the plan — recognizing already-satisfied work and skipping the redundant step
rather than inserting more. 4 of the 5 steps executed; step 4 was skipped.

| Step | Title | Executor result | Review decision |
|------|-------|-----------------|-----------------|
| 1 | Init crate + clap | completed (wrote `Cargo.toml`, `src/main.rs` with greet + tests + main) | Continue |
| 2 | CLI struct + main wiring | completed (renamed `Args`→`Cli`, `times: Option<usize>`) | Continue |
| 3 | Pure `greet()` | completed (already present; no changes needed) | **Skip** — "step 4 redundant; greet + tests already in place" |
| 4 | Add unit tests | *(skipped — no execution)* | — |
| 5 | Wire greet + error handling | completed (verified wiring, `--times 0` error, ran tests) | **Stop** — "all steps finished; verification confirms it works" |

**Review action tally:** 2 Continue · 1 Skip · 1 Stop · 0 Insert.

Every executed step emitted a well-formed `<<<STEP_RESULT_JSON>>>` envelope —
**0 steps reported `unknown`**. Each result/review is persisted as
`.bootstrap/step-NN-result.json` / `.bootstrap/step-NN-review.json`.

## Output — the generated CLI

`greet/src/main.rs`:

```rust
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "greet", version, about = "A friendly greeting utility")]
struct Cli {
    /// The name to greet
    name: String,

    /// Number of times to repeat the greeting (default: 1)
    #[arg(short, long, default_value = "1")]
    times: Option<usize>,
}

fn greet(name: &str, times: usize) -> String {
    let greeting = format!("Hello, {}!", name);
    (0..times)
        .map(|_| greeting.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod greet_tests {
    use super::greet;

    #[test]
    fn greet_once() {
        assert_eq!(greet("Alice", 1), "Hello, Alice!");
    }

    #[test]
    fn greet_thrice() {
        assert_eq!(greet("Bob", 3), "Hello, Bob!\nHello, Bob!\nHello, Bob!");
    }
}

fn main() {
    let args = Cli::parse();
    let times = args.times.unwrap_or(1);
    if times == 0 {
        eprintln!("Error: --times must be at least 1");
        std::process::exit(1);
    }
    println!("{}", greet(&args.name, times));
}
```

`greet/Cargo.toml`: edition 2024, single dependency `clap = { version = "4", features = ["derive"] }`.

### Independent verification (performed by a human, not the model's claim)

Run against the generated crate after the loop halted:

- `cargo test` → **2 passed** (`greet_once`, `greet_thrice`)
- `cargo build` → clean
- `greet Alice` → `Hello, Alice!`
- `greet Bob --times 3` → three greeting lines
- `greet X --times 0` → `Error: --times must be at least 1`, exit code **1**

The project met the spec, including the pure `greet()` with I/O isolated in
`main()`, both required tests, and clap-only dependencies. Unlike the macOS run,
`name` was implemented as the **required positional argument** the spec asked for
(the macOS run used a `-n` flag).

## What this run shows (and how it differs from the macOS run)

1. **Reliable structured output.** Every step emitted its result envelope — 0
   `unknown`. The macOS POC's main weakness (5/12 steps `unknown` because the
   executor skipped the envelope) **did not manifest here**, so the reviewer
   never had to insert recovery/verification steps.
2. **The reviewer contracted instead of expanding.** With trustworthy results it
   recognized completed work, **Skipped** the redundant test step, and **Stopped
   on genuine completion** — a legitimate Stop, not the macOS run's false alarm.
3. **The two runs together exercise all four review actions.** macOS hit
   Continue / Insert / Stop (plan 6 → 12 steps); Arch hit Continue / Skip / Stop
   (plan 5 → 4 steps). Skip and Insert each appeared in exactly one of the runs.
4. **Performance.** ~132 tok/s generation on the 3090 vs ~31 tok/s reported on the
   Mac — roughly 4× — and the Arch run reached this **without** speculative
   decoding. (Note the two figures aren't a controlled benchmark; see the caveats
   below.)

> These two runs are independent single samples on different hardware with
> different decoding settings and non-deterministic generation — the divergence
> (12 steps vs 4, Insert vs Skip) reflects run-to-run variance as much as
> platform. Both reached a correct, tested result; neither is "the" canonical
> trace.

## Reproduce

```bash
# start the server (foreground, this box): docs/config/arch-nvidia/start-qwable.sh
#   (or the pinned launcher ~/tools/qwable/run-qwable.sh)

# plan only (one cheap call):
PLAN_ONLY=1 ./scripts/demo-orchestrate.sh

# full loop in an isolated temp workspace:
./scripts/demo-orchestrate.sh
```
