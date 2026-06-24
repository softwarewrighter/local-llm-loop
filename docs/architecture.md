# Architecture

High-level system structure. For the control-flow narrative and rationale see
[orchestrator.md](orchestrator.md); for the concrete contracts (CLI, JSON
schemas, artifacts) see [design.md](design.md).

## System Overview
A Rust binary (`bootstrap`) orchestrates a local LLM via the `opencode` CLI. In
its primary `orchestrate` mode it turns a spec file into a plan, then iterates
the steps — executing each with a tool-using model and reviewing the result with
a supervising model that decides how to proceed. A secondary `exec` mode runs an
explicit task list.

## Core Components
| Component | File | Responsibility |
|-----------|------|----------------|
| CLI / dispatch | `src/main.rs` | Parse args (clap subcommands), dispatch `orchestrate` / `exec`; `exec` task-runner lives here |
| Orchestrator | `src/orchestrate.rs` | The plan → execute → review loop; owns plan, step cursor, history; persists artifacts |
| Prompts | `src/prompts.rs` | Build planner/executor/reviewer prompts; JSON sentinels and schemas |
| LLM adapter | `src/llm.rs` | `opencode run` wrapper; ANSI stripping; defensive JSON extraction |
| Types | `src/types.rs` | serde types: `Plan`, `Step`, `StepResult`, `ReviewDecision`, `ReviewAction` |

## Data Flow
```
spec.txt
  │
  ▼
LLM(plan) ──► Plan{design, steps[]} ──► persist plan.json
  │
  ▼   (Rust owns the cursor + history)
loop over steps:
  LLM(execute, tools) ──► StepResult ──► persist step-NN-result.json
  LLM(review)        ──► ReviewDecision ──► persist step-NN-review.json
       │
       ├─ continue → advance cursor
       ├─ insert   → splice new steps after cursor
       ├─ skip     → drop next step
       └─ stop     → halt for human
```

## Key Design Decisions
- **Rust owns orchestration state.** Every LLM call is a stateless
  `opencode run`; all needed context (goal, design, prior summaries, the step) is
  passed in the prompt. This is what makes insert/skip/replan deterministic.
- **Structured hand-offs via sentinel-delimited JSON.** Each role emits JSON
  between explicit markers; the adapter strips ANSI and prefers the
  sentinel span, falling back to the first `{` … last `}`.
- **Fail safe.** A missing executor envelope is recorded as `status: unknown`
  (the reviewer still judges); an unparseable review becomes `stop`.
- **Synchronous, one slot.** Steps run sequentially via blocking
  `Command::output()`; sufficient and keeps state simple.
- **Local-first.** Targets a local `llama-server` (provider/model
  `llamacpp/qwable`); no cloud dependency.

## Future Extensions
- Distinct top (planner/reviewer) vs executor models.
- More reliable executor reporting (`opencode run --format json` or a dedicated
  report sub-call).
- Resume a stopped loop after human edits.

See [plan.md](plan.md) for the roadmap and current status.
