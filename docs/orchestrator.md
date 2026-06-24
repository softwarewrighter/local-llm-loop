# Orchestrator

> Supersedes the original "run opencode in a loop" design in `prd.md` /
> `architecture.md` / `design.md` / `plan.md`. Those describe the earlier
> task-runner; the project is now a planner → executor → reviewer orchestrator.

## Goal

Take a small spec file (goal + description + architectural decisions), have an
LLM turn it into a design and a serialized list of implementation steps, then
iterate the steps — executing each with a tool-using LLM and feeding the
structured result back to a supervising LLM that decides how to proceed.

## Control flow

```
spec.txt
   │
   ▼
[PLAN]   planner LLM → { design, notes, steps[] }            (persisted: plan.json)
   │
   ▼
for each step (Rust owns the cursor):
   │
   ├─ [EXECUTE]  tool-using LLM performs the step, then emits a JSON
   │             result envelope: { summary, new_or_changed_files[],
   │             changes, features[], bugs[], issues[], status }   (step-NN-result.json)
   │
   └─ [REVIEW]   supervising LLM sees the step + its result + remaining steps,
                 and returns a decision:                           (step-NN-review.json)
                   continue → next planned step, unchanged
                   insert   → splice new_steps in to run next
                   skip     → drop the next planned step
                   stop     → request human intervention, halt
```

The same model fills all three roles today (`--model`, default
`llamacpp/qwable`). The roles are kept separate in the code so a different
"top" model can be wired in later.

## Why Rust owns the state

Every LLM call is a stateless `opencode run`; all needed context (goal, design,
prior step summaries, the step itself) is passed explicitly in the prompt. The
orchestrator — not opencode's session memory — holds the plan, the step cursor,
and the running history. That's what makes `insert` / `skip` / replanning
deterministic and controllable.

## Structured hand-offs

Each role is asked to emit JSON between explicit sentinels
(`<<<PLAN_JSON>>>`, `<<<STEP_RESULT_JSON>>>`, `<<<REVIEW_JSON>>>`). The Rust
side strips ANSI, prefers the sentinel-delimited span, and falls back to the
first `{` … last `}`. Tolerant by design:

- Executor emits no/invalid envelope → recorded as `status: unknown` with the
  raw text as summary; the reviewer still gets to judge.
- Reviewer verdict unparseable → fail safe to `stop` (human intervention).

## Safety

- `--max-steps` (default 20) caps total executed steps so `insert` loops can't
  run away.
- llama-server should also keep `--n-predict` bounded (see `start-qwable.sh`).

## Code map

| File | Role |
|------|------|
| `src/main.rs` | CLI; dispatches `orchestrate` and `exec` subcommands |
| `src/orchestrate.rs` | the plan → execute → review loop |
| `src/prompts.rs` | prompt builders + JSON sentinels/schemas |
| `src/llm.rs` | `opencode run` wrapper, ANSI strip, JSON extraction |
| `src/types.rs` | serde types: `Plan`, `Step`, `StepResult`, `ReviewDecision` |

## Usage

```bash
# Plan only (cheap — one LLM call — to sanity-check the planner):
PLAN_ONLY=1 ./scripts/demo-orchestrate.sh

# Full loop in an isolated temp workspace:
./scripts/demo-orchestrate.sh

# Direct:
bootstrap orchestrate path/to/spec.txt --model llamacpp/qwable \
  --work-dir /path/to/project --max-steps 20 --verbose
```

Artifacts land in `<work-dir>/.bootstrap/` (`plan.json`, `step-NN-result.json`,
`step-NN-review.json`).

## Future

- Distinct top (planner/reviewer) vs executor models.
- Feed cumulative review history back into planning for mid-build replans.
- Switch hand-offs to `opencode run --format json` if sentinel parsing proves
  brittle on a given model.
```
