# Plan / Roadmap

Status of the project and what's next. Reflects the current orchestrator design,
not the original loop prototype.

## Done
- [x] CLI with `orchestrate` and `exec` subcommands (clap).
- [x] `opencode run` adapter with ANSI stripping and defensive JSON extraction.
- [x] serde contracts: `Plan`, `Step`, `StepResult`, `ReviewDecision`.
- [x] Planner: spec → design + serialized step list.
- [x] Executor: tool-using step execution with a structured result envelope.
- [x] Reviewer: continue / insert / skip / stop, with mid-run plan mutation.
- [x] Artifact persistence under `.bootstrap/`.
- [x] Safety: `--max-steps`, `--plan-only`, fail-safe `stop` on unparseable review.
- [x] Demo scripts and a sample spec (`examples/spec-greeter.txt`).
- [x] End-to-end proof of concept against a local model — see
      [poc-summary.md](poc-summary.md).

## Next (highest value first)
1. **Reliable executor reporting.** The PoC's main weakness: steps did the work
   but sometimes omitted the JSON envelope, yielding `unknown` results and a
   false-alarm `stop`. Fix via `opencode run --format json` parsing, or a
   dedicated "report" sub-call that only emits the envelope.
2. **Distinct planner/reviewer vs executor models** (`--top-model` /
   `--exec-model`).
3. **Collapse redundant inserts** — the reviewer re-derived build/test steps
   several times; detect and dedupe.
4. **Resume after stop** — let a human edit and resume a halted run instead of
   restarting.

## Later
- Feed cumulative review history back into planning for larger mid-build replans.
- Per-step ret/timeouts and richer failure classification.
- Optional parallel execution for independent steps.

## Risks & Mitigations
- **LLM output variance** (local quantized models) → defensive extraction +
  fail-safe stop; tighten via structured-output mode (item 1).
- **Runaway inserts** → `--max-steps` cap.
- **opencode CLI drift** → all invocation isolated in `src/llm.rs`.
