# Proof-of-Concept Summary — macOS / Apple Silicon

A full end-to-end run of the planner → executor → reviewer orchestrator
(`bootstrap orchestrate`) against a local LLM, building a small Rust CLI from a
spec file with no human authoring of code — on macOS / Apple Silicon. The Arch
Linux / NVIDIA counterpart is [arch-poc-summary.md](arch-poc-summary.md).

- **Date:** 2026-06-24
- **Model (all three roles):** `llamacpp/qwable` (Qwable-v1 IQ4_XS) on local llama-server
- **Host:** macOS / Apple Silicon (Metal), unified memory (full 128k-token context)
- **Command:** `./scripts/demo-orchestrate.sh` (builds in an isolated temp workspace)
- **Generation rate observed:** ~31 tok/s (speculative decoding active)
- **Outcome:** working, tested Rust CLI produced; loop halted with a (false-alarm) Stop

## Input

The orchestrator's only input was this spec file (`examples/spec-greeter.txt`):

```
GOAL
Build a small, well-tested Rust command-line "greeter" utility.

DESCRIPTION
A CLI named `greet` that prints a friendly greeting. It takes a required NAME
argument and an optional --times N flag (default 1) that repeats the greeting N
times. Invalid input (e.g. --times 0) should produce a clear error.

ARCHITECTURAL DECISIONS
- Language: Rust, edition 2024. Single binary crate.
- Argument parsing: clap with the derive feature.
- Core logic lives in a pure function `greet(name: &str, times: usize) -> String`
  that is unit-tested; main() only wires up I/O.
- Include at least one #[test] verifying greet() output for times = 1 and times = 3.
- Keep dependencies minimal (clap only).
```

## Stage 1 — Plan (one LLM call)

The planner turned the spec into a design + 6-step plan (persisted to
`.bootstrap/plan.json`):

> **Design:** A single binary crate built with Rust 2024 edition. The CLI is
> defined by a clap `derive` struct with a positional `name` argument and an
> optional `--times` argument defaulting to 1. `main()` only parses arguments,
> validates `times > 0`, calls the pure `greet(name, times)` function, and
> prints the result…
>
> **Notes:** Rust 2024 edition requires `rustc` 1.85+… clap 4.x with the
> `derive` feature. No additional crates.

| # | Planned step |
|---|---|
| 1 | Initialize the Rust project and add clap |
| 2 | Define the CLI argument struct and main wiring |
| 3 | Implement the pure `greet` function |
| 4 | Add unit tests for `greet` |
| 5 | Validate `--times` input and produce errors |
| 6 | Final verification and clean build |

The planner even surfaced a real toolchain risk in `notes` (edition 2024 needs
rustc ≥ 1.85) — unprompted foresight.

## Stage 2 — Execute + Review loop

The 6-step plan grew to **12 executed steps** as the reviewer dynamically
replanned. Full trace (executor result → reviewer decision):

| Step | Title | Executor result | Review decision |
|------|-------|-----------------|-----------------|
| 1 | Init crate + clap | completed | Continue |
| 2 | CLI struct + main wiring | completed | Continue |
| 3 | Pure `greet()` | completed | Continue |
| 4 | Unit tests | completed | **Insert** — "CLI accepts `--times 0`; should be rejected" |
| 5 (id 500) | Validate `--times` (inserted) | *unknown* | Continue |
| 6 (id 5) | Validate `--times` | completed | Continue |
| 7 (id 6) | Final verification | *unknown* | **Insert** — "no results; must build+test" |
| 8 (id 100) | Build with tests | completed | Continue |
| 9 (id 101) | Run binary | *unknown* | **Insert** — "no JSON result; insert build/run/test" |
| 10 (id 102) | Build release | completed | Continue |
| 11 (id 103) | Run binary | *unknown* | Continue |
| 12 (id 104) | Run unit tests | *unknown* | **Stop** — "test step returned no result; halt for human" |

**Review action tally:** 8 Continue · 3 Insert · 1 Stop · 0 Skip.

Each step's result and review decision is persisted as
`.bootstrap/step-NN-result.json` / `.bootstrap/step-NN-review.json`.

## Output — the generated CLI

The executor scaffolded a `greet/` crate. Final `greet/src/main.rs`:

```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "greet", about = "A friendly greeting CLI")]
struct GreetArgs {
    #[arg(short = 'n', required = true)]
    name: String,

    #[arg(short = 't', long = "times", default_value_t = 1)]
    times: usize,
}

fn greet(name: &str, times: usize) -> String {
    let greeting = format!("Hello, {name}!");
    (0..times).map(|_| greeting.clone()).collect::<Vec<_>>().join("\n")
}

fn main() -> std::process::ExitCode {
    let args = GreetArgs::parse();
    let name = args.name;
    let times = args.times;

    if times == 0 {
        eprintln!("Error: --times must be >= 1");
        return std::process::ExitCode::from(1);
    }

    let greeting = greet(&name, times);
    println!("{}", greeting);
    std::process::ExitCode::from(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greet_once() {
        assert_eq!(greet("Bob", 1), "Hello, Bob!");
    }

    #[test]
    fn greet_thrice() {
        assert_eq!(greet("Bob", 3), "Hello, Bob!\nHello, Bob!\nHello, Bob!");
    }
}
```

`greet/Cargo.toml`: edition 2024, single dependency `clap = { version = "4", features = ["derive"] }`.

### Independent verification (performed by a human, not the model's claim)

When the loop halted requesting human intervention, the verification it asked
for was performed manually:

- `cargo test` → **2 passed** (`greet_once`, `greet_thrice`)
- `greet -n Alice` → `Hello, Alice!`
- `greet -n Bob --times 3` → three greeting lines
- `greet -n X --times 0` → `Error: --times must be >= 1`, exit code **1**

The project met the spec, including the pure `greet()` function with I/O
isolated in `main()`, both required tests, and clap-only dependencies. The only
deviation: `name` was implemented as a `-n` flag rather than a positional
argument.

## What the POC proved

1. **The full pipeline works against a real local model.** A plain-text spec
   became a designed, planned, implemented, and (independently) verified Rust
   CLI with zero human-authored code.
2. **The control loop is sound.** The reviewer genuinely *supervised*: it caught
   a spec gap the planner missed (`--times 0` rejection) and inserted a fix, it
   inserted recovery steps when execution stalled, and — crucially — it chose to
   **Stop and request a human rather than declare false success.** Three of the
   four review actions (Continue / Insert / Stop) fired naturally; Skip did not
   occur in this run.
3. **Rust-owned state held up.** Plan, step cursor, mid-run inserts, and history
   were all managed in Rust; every artifact was persisted to `.bootstrap/`.

## The key weakness — executor result reliability

5 of the 12 steps reported `status: unknown` because the **executor did the work
but did not emit the `<<<STEP_RESULT_JSON>>>` envelope**. The tolerant fallback
prevented crashes, and the reviewer compensated by inserting verification steps
— which is *why the plan ballooned from 6 to 12 steps*.

The final **Stop was a false alarm**: the unit-test step actually passed, but
because the executor returned no envelope, the reviewer couldn't confirm it and
conservatively halted. The reviewer is only as good as the structured result it
receives.

### Highest-value next step

Make the executor's structured output reliable, so `unknown` becomes rare and
`Stop` means a real problem:

- Use `opencode run --format json` and parse the assistant message
  deterministically, **or**
- Add a dedicated "report" sub-call after each step whose *only* job is to emit
  the result envelope (separating "do the work" from "report the work").

## Other follow-ups

- Distinct top (planner/reviewer) vs executor models.
- Feed cumulative review history back into planning for larger mid-build replans.
- Detect and collapse redundant reviewer inserts (the loop re-derived
  build/test several times).

## Reproduce

```bash
# plan only (one cheap call):
PLAN_ONLY=1 ./scripts/demo-orchestrate.sh

# full loop in an isolated temp workspace:
./scripts/demo-orchestrate.sh
```
