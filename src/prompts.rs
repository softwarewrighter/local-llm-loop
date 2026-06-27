//! Prompt construction for the three LLM roles: planner, executor, reviewer.
//!
//! Each prompt asks the model to emit a JSON payload between explicit sentinels
//! so the Rust side can extract it reliably regardless of surrounding prose.

use crate::types::{HistoryEntry, Step};
use std::fmt::Write as _;

pub const PLAN_START: &str = "<<<PLAN_JSON>>>";
pub const PLAN_END: &str = "<<<END_PLAN_JSON>>>";
pub const STEP_START: &str = "<<<STEP_RESULT_JSON>>>";
pub const STEP_END: &str = "<<<END_STEP_RESULT_JSON>>>";
pub const REVIEW_START: &str = "<<<REVIEW_JSON>>>";
pub const REVIEW_END: &str = "<<<END_REVIEW_JSON>>>";

const PLAN_SCHEMA: &str = r#"{
  "design": "string: a few sentences describing the architecture and how it honors the spec's decisions",
  "notes": "string: assumptions or risks (optional, may be empty)",
  "steps": [
    {
      "id": 1,
      "title": "short imperative title",
      "instructions": "precise instructions for a tool-using coding agent: exactly what to create or change",
      "context": "specifics the agent needs: file paths, function signatures, decisions to honor",
      "acceptance": "how to verify this step is complete"
    }
  ]
}"#;

const STEP_RESULT_SCHEMA: &str = r#"{
  "summary": "1-3 sentence summary of what you did",
  "new_or_changed_files": ["relative/path/one", "relative/path/two"],
  "changes": "concise description of the concrete changes made",
  "features": ["feature or capability added"],
  "bugs": ["bug you hit or introduced, if any"],
  "issues": ["open issue / TODO / thing needing attention"],
  "status": "completed | partial | blocked"
}"#;

const REVIEW_SCHEMA: &str = r#"{
  "action": "continue | insert | skip | stop",
  "reason": "one or two sentences explaining the decision",
  "new_steps": [
    {
      "id": 100,
      "title": "...",
      "instructions": "...",
      "context": "...",
      "acceptance": "..."
    }
  ]
}"#;

/// Append the "write the JSON to a file, then validate it with our helper"
/// instruction shared by every role.
///
/// We hand the JSON off via a file (opencode is agentic and prefers tool calls
/// over inline prose, so a known output path is the reliable channel) AND have
/// the model validate it with the `bootstrap emit` CLI helper. The helper checks
/// the JSON against the harness's own schema and prints `OK:` or a correctable
/// `Error:`, giving the model a self-healing loop — and the call doubles as a
/// worked example of invoking a shell tool, which generalizes to others.
fn append_write_json(p: &mut String, role: &str, out_path: &str, schema: &str) {
    let _ = write!(
        p,
        "OUTPUT — produce a single JSON object in EXACTLY this shape (no markdown \
         fences, no prose, nothing but the JSON):\n"
    );
    p.push_str(schema);
    p.push('\n');
    let _ = write!(
        p,
        "Deliver it with our command-line helper `bootstrap` (a tool on your PATH, run \
         from the shell like `cargo` or `ls`).\n\
         \n\
         WORKED EXAMPLE — copy this exact sequence of ACTIONS (not the content):\n\
         \x20  • write the JSON object to the file `{out_path}` (relative to the project dir)\n\
         \x20  • run the shell command:   bootstrap emit {role} --file {out_path}\n\
         \x20  • it replies:              OK: {role} envelope valid — …\n\
         \x20  • you reply with just:     DONE\n\
         If instead it replies `Error: …`, it names the one thing that is wrong: edit \
         `{out_path}` to fix exactly that and run the SAME command again until it says OK.\n\
         \n\
         RULES (this is where weaker models fail — do not repeat these mistakes):\n\
         \x20  • the file is the BARE object — do NOT wrap it, e.g. NOT {{\"content\": \"{{…}}\"}}\n\
         \x20  • write EXACTLY ONE object — no second JSON object, no prose around it\n\
         \x20  • do NOT use task / sub-agent / todo tools, and do NOT inspect the helper\n\
         \x20    (no `bootstrap --version`) — just write the file and run the command above\n\
         Your ONLY deliverable is `{out_path}` passing `bootstrap emit {role}`. Reply DONE only after OK.\n"
    );
}

/// Planner: turn the spec into a design + ordered, serialized step list.
pub fn plan_prompt(spec: &str, out_path: &str) -> String {
    let mut p = String::new();
    p.push_str(
        "You are a senior software architect and planner. You produce concrete, \
         executable plans for a single tool-using coding agent.\n\n",
    );
    p.push_str("PROJECT SPECIFICATION (goal, description, architectural decisions):\n");
    p.push_str("----------------------------------------------------------------\n");
    p.push_str(spec.trim());
    p.push_str("\n----------------------------------------------------------------\n\n");
    p.push_str(
        "Produce (1) a concise DESIGN and (2) an ordered list of small, concrete, \
         independently-verifiable implementation STEPS. Each step must be small enough \
         for one coding agent to complete in a single pass. Honor every architectural \
         decision in the spec. Prefer 3-8 steps for a small project. Number steps \
         sequentially starting at 1.\n\n",
    );
    append_write_json(&mut p, "plan", out_path, PLAN_SCHEMA);
    p
}

/// Executor: perform one step (tool-using) and report a structured result.
pub fn step_prompt(spec: &str, design: &str, history: &[HistoryEntry], step: &Step, out_path: &str) -> String {
    let mut p = String::new();
    p.push_str(
        "You are an implementation agent with tools (read/write files, run shell \
         commands) operating in the project working directory. You make real changes; \
         you do not ask for confirmation.\n\n",
    );
    p.push_str("PROJECT GOAL & ARCHITECTURE:\n");
    p.push_str(spec.trim());
    p.push_str("\n\nDESIGN:\n");
    p.push_str(design.trim());
    p.push_str("\n\nPROGRESS SO FAR:\n");
    if history.is_empty() {
        p.push_str("(none yet — this is the first step)\n");
    } else {
        for h in history {
            let (s, r) = (&h.step, &h.result);
            let _ = writeln!(
                p,
                "- step {} \"{}\": {} [status: {}]",
                s.id,
                s.title,
                if r.summary.is_empty() { "(no summary)" } else { &r.summary },
                if r.status.is_empty() { "unknown" } else { &r.status }
            );
        }
    }
    let _ = write!(p, "\nYOUR CURRENT STEP ({}): {}\n", step.id, step.title);
    let _ = write!(p, "Instructions: {}\n", step.instructions);
    if !step.context.trim().is_empty() {
        let _ = write!(p, "Context: {}\n", step.context);
    }
    if !step.acceptance.trim().is_empty() {
        let _ = write!(p, "Acceptance: {}\n", step.acceptance);
    }
    p.push_str(
        "\nDo the work now using your tools (read/write project files, run commands). \
         When the work is done, report what you did: ",
    );
    append_write_json(&mut p, "step", out_path, STEP_RESULT_SCHEMA);
    p
}

/// Reviewer (top LLM): decide how the build proceeds after a step.
pub fn review_prompt(
    spec: &str,
    design: &str,
    step: &Step,
    result_json: &str,
    remaining: &[Step],
    out_path: &str,
) -> String {
    let mut p = String::new();
    p.push_str(
        "You are the supervising planner. After each implementation step you decide \
         how the build proceeds. Be pragmatic: only intervene when warranted.\n\n",
    );
    p.push_str("PROJECT GOAL & ARCHITECTURE:\n");
    p.push_str(spec.trim());
    p.push_str("\n\nDESIGN:\n");
    p.push_str(design.trim());
    let _ = write!(p, "\n\nSTEP JUST EXECUTED ({}): {}\n", step.id, step.title);
    p.push_str("Its reported result (JSON):\n");
    p.push_str(result_json.trim());
    p.push_str("\n\nREMAINING PLANNED STEPS:\n");
    if remaining.is_empty() {
        p.push_str("(none — this was the last planned step)\n");
    } else {
        for s in remaining {
            let _ = writeln!(p, "- {}: {}", s.id, s.title);
        }
    }
    p.push_str(
        "\nChoose the next action:\n\
         - \"continue\": the step is acceptable; proceed to the next planned step unchanged.\n\
         - \"insert\": add one or more new steps to run NEXT (put them in \"new_steps\"), then continue.\n\
         - \"skip\": the next planned step is no longer needed; skip it.\n\
         - \"stop\": something needs human intervention; halt the build.\n\n",
    );
    p.push_str("Use an empty \"new_steps\" list unless the action is \"insert\".\n");
    append_write_json(&mut p, "review", out_path, REVIEW_SCHEMA);
    p
}
