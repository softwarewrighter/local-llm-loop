# Product Requirements Document (PRD)

## Problem Statement
Turning a high-level idea into working code with a local LLM is tedious and
error-prone: a single long generation drifts, lacks verification, and gives the
operator no control points. Users need a harness that decomposes the work,
implements it step by step, and *reviews* each step — pausing for a human only
when something genuinely needs attention.

## Goals
- Take a short plain-text spec (goal + description + architectural decisions) and
  drive it to a working, verified implementation with a local LLM.
- Keep a human in control of the loop's *direction* without authoring code:
  the supervising model can continue, insert, skip, or stop for intervention.
- Make every decision inspectable and reproducible (persisted artifacts).
- Run fully against a local model (no cloud dependency).

## User Stories
1. As a user, I want to hand the tool a spec file and get a designed, planned,
   implemented project back.
2. As a user, I want the tool to verify each step and adapt the plan when a step
   reveals a gap, instead of blindly running a fixed list.
3. As a user, I want it to stop and ask me when it can't confirm success, rather
   than declare false success.
4. As a user, I want to inspect the plan, each step's result, and each review
   decision after the fact.
5. As a power user, I want a lower-level mode to run an explicit list of tasks.

## Functional Requirements
- `orchestrate <spec>`: plan → execute → review loop (see
  [orchestrator.md](orchestrator.md)).
- `exec <tasks...>`: run an explicit task list, one `opencode run` per task,
  optionally chained into one session.
- Planner produces a serialized design + ordered step list.
- Executor performs each step with tools and reports a structured result
  (summary, new/changed files, changes, features, bugs, issues, status).
- Reviewer returns one of: continue | insert | skip | stop.
- Persist plan, per-step results, and per-step review decisions under
  `<work-dir>/.bootstrap/`.
- A `--max-steps` safety cap and a `--plan-only` dry-run mode.

## Non-Functional Requirements
- **Controllability**: Rust owns all loop state; LLM calls are stateless.
- **Robustness**: tolerate malformed LLM output (extract JSON defensively; fail
  safe to `stop` on an unparseable review).
- **Reproducibility**: artifacts on disk for every decision.
- **Simplicity / locality**: minimal dependencies; runs against a local model.

## Out of Scope (for now)
- A GUI.
- Distinct planner vs executor models (planned — see [plan.md](plan.md)).
- Parallel/concurrent step execution.
- Feeding human edits back into a paused loop to resume it.

## Success Metrics
- From a small spec, the loop produces a project that builds and passes its own
  tests, verified independently.
- The reviewer demonstrably adapts the plan (insert/skip) and halts (stop) rather
  than emitting false success. (Demonstrated — see
  [mac-poc-summary.md](mac-poc-summary.md) and [arch-poc-summary.md](arch-poc-summary.md).)
- `unknown`-status steps (executor failed to report a structured result) are rare.
