# Design

The concrete contracts: CLI surface, the JSON data structures exchanged with the
LLM, the sentinel protocol, and the on-disk artifacts. For the loop's rationale
see [orchestrator.md](orchestrator.md).

## CLI Surface

```
bootstrap orchestrate <SPEC_FILE> [--model <M>] [--work-dir <DIR>]
                                  [--max-steps <N>] [--plan-only] [--verbose]
bootstrap exec <TASKS>...          [--model <M>] [--no-chain] [--verbose]
```

| Option | Applies to | Default |
|--------|-----------|---------|
| `--model` | both | `llamacpp/qwable` |
| `--work-dir` | orchestrate | `.` |
| `--max-steps` | orchestrate | `20` |
| `--plan-only` | orchestrate | off |
| `--no-chain` | exec | off (tasks chain into one session) |
| `--verbose` | both | off |

## Data Structures (`src/types.rs`)

```rust
struct Step {                 // one unit of work
    id: u32,
    title: String,
    instructions: String,     // what the executor must do
    context: String,          // specifics: paths, signatures, decisions
    acceptance: String,       // how to know it's done
}

struct Plan {                 // planner output
    design: String,
    notes: String,
    steps: Vec<Step>,
}

struct StepResult {           // executor output (the "report")
    summary: String,
    new_or_changed_files: Vec<String>,
    changes: String,
    features: Vec<String>,
    bugs: Vec<String>,
    issues: Vec<String>,
    status: String,           // completed | partial | blocked | unknown
}

enum ReviewAction { Continue, Insert, Skip, Stop }

struct ReviewDecision {       // reviewer output
    action: String,           // continue | insert | skip | stop
    reason: String,
    new_steps: Vec<Step>,     // required only for "insert"
}
```

All fields use serde defaults so partial/loose LLM JSON still deserializes.

## Sentinel Protocol

Each role is asked to emit its JSON between explicit markers so the Rust side can
extract it regardless of surrounding prose or ANSI:

| Role | Start | End |
|------|-------|-----|
| Planner | `<<<PLAN_JSON>>>` | `<<<END_PLAN_JSON>>>` |
| Executor | `<<<STEP_RESULT_JSON>>>` | `<<<END_STEP_RESULT_JSON>>>` |
| Reviewer | `<<<REVIEW_JSON>>>` | `<<<END_REVIEW_JSON>>>` |

Extraction (`src/llm.rs`): strip ANSI → prefer the sentinel-delimited span →
fall back to the first `{` … last `}` → `serde_json` parse.

## Loop Algorithm

1. Read + validate the spec; create `<work-dir>/.bootstrap/`.
2. **Plan:** one LLM call → `Plan`; persist `plan.json`.
   Stop here if `--plan-only`.
3. Maintain `steps: Vec<Step>`, a cursor `i`, and `history`.
4. While `i < steps.len()` and `executed < max_steps`:
   a. **Execute** `steps[i]` (tool-using) → `StepResult`; persist
      `step-NN-result.json`; push to history.
   b. **Review** with the result + remaining steps → `ReviewDecision`; persist
      `step-NN-review.json`.
   c. Apply: `continue` → `i+=1`; `insert` → splice `new_steps` after `i`,
      `i+=1`; `skip` → remove `steps[i+1]`, `i+=1`; `stop` → halt for human.

## Error Handling

- Executor emits no/invalid envelope → `StepResult { status: "unknown", … }`
  with the raw text as summary; the reviewer still decides.
- Reviewer output unparseable → synthesized `stop` (fail safe to human).
- `opencode` missing or a call exits non-zero → propagated as an error.
- `--max-steps` bounds runaway inserts.

## Artifacts (`<work-dir>/.bootstrap/`)

```
plan.json
step-01-result.json   step-01-review.json
step-02-result.json   step-02-review.json
...
```

Files are named by each step's `id`, so inserted steps (which carry their own
ids) are traceable.
