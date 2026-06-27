//! `bootstrap emit <role> --file <path>` — the model-facing JSON helper.
//!
//! Prompts tell each LLM role to write its JSON envelope to a file and then run
//! this command. It validates the file against the *same* schema the harness
//! parses (so "it passed `emit`" ⇒ "the harness will accept it"), rewrites it
//! cleanly on success, and prints a precise, correctable `Error:` on failure.
//! That gives the model a self-correcting feedback loop instead of silently
//! emitting a malformed or wrong-schema envelope (e.g. a `todowrite` list with
//! no `action` field — the exact gate-2 break we saw with Qwen3-Coder).
//!
//! It is a plain binary on `$PATH` (not MCP): the model invokes it through
//! opencode's shell tool, which also serves as a worked example that generalizes
//! to calling any other CLI tool.

use crate::types::{Plan, ReviewDecision, StepResult};
use crate::{llm, prompts};
use anyhow::{Context, Result, anyhow, bail};
use std::path::Path;

/// Validate (and canonicalize in place) the envelope `role` stored at `file`.
pub fn run(role: &str, file: &Path) -> Result<()> {
    let raw = std::fs::read_to_string(file)
        .with_context(|| format!("cannot read {} (write the JSON there first)", file.display()))?;

    let (canonical, summary) = match role {
        "plan" => {
            let v = parse::<Plan>(&raw, prompts::PLAN_START, prompts::PLAN_END, role, PLAN_HINT)?;
            if v.steps.is_empty() {
                bail!("plan envelope invalid: `steps` is empty — a plan needs at least one step.\nexpected shape:\n{PLAN_HINT}");
            }
            (serde_json::to_string_pretty(&v)?, format!("{} step(s)", v.steps.len()))
        }
        "step" => {
            let v = parse::<StepResult>(&raw, prompts::STEP_START, prompts::STEP_END, role, STEP_HINT)?;
            let s = if v.status.is_empty() { "unknown".into() } else { v.status.clone() };
            (serde_json::to_string_pretty(&v)?, format!("status={s}"))
        }
        "review" => {
            let v = parse::<ReviewDecision>(&raw, prompts::REVIEW_START, prompts::REVIEW_END, role, REVIEW_HINT)?;
            let a = v.action.trim().to_lowercase();
            if !matches!(a.as_str(), "continue" | "insert" | "skip" | "stop") {
                bail!("review envelope invalid: `action` is \"{}\", must be one of continue|insert|skip|stop.\nexpected shape:\n{REVIEW_HINT}", v.action);
            }
            (serde_json::to_string_pretty(&v)?, format!("action={a}"))
        }
        other => bail!("unknown role `{other}` — expected: plan | step | review"),
    };

    std::fs::write(file, format!("{canonical}\n"))
        .with_context(|| format!("cannot write {}", file.display()))?;
    println!("OK: {role} envelope valid — {summary}");
    Ok(())
}

fn parse<T: serde::de::DeserializeOwned>(
    raw: &str,
    start: &str,
    end: &str,
    role: &str,
    hint: &str,
) -> Result<T> {
    llm::parse_json::<T>(raw, start, end)
        .map_err(|e| anyhow!("{role} envelope invalid: {e}\nexpected shape:\n{hint}"))
}

const PLAN_HINT: &str = r#"{"design":"...","notes":"...","steps":[{"id":1,"title":"...","instructions":"...","context":"...","acceptance":"..."}]}"#;
const STEP_HINT: &str = r#"{"summary":"...","new_or_changed_files":["..."],"changes":"...","features":["..."],"bugs":[],"issues":[],"status":"completed|partial|blocked"}"#;
const REVIEW_HINT: &str = r#"{"action":"continue|insert|skip|stop","reason":"...","new_steps":[]}"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("emit-{}-{}", std::process::id(), name));
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn accepts_valid_plan_and_canonicalizes() {
        // note the trailing comma — emit should relax + accept it.
        let f = tmp("plan.json", r#"{"design":"d","steps":[{"id":1,"title":"t"},]}"#);
        assert!(run("plan", &f).is_ok());
        let rewritten = std::fs::read_to_string(&f).unwrap();
        assert!(rewritten.contains("\"design\""));
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn rejects_review_without_action() {
        // the exact Qwen3-Coder break: a todowrite-shaped object, no `action`.
        let f = tmp("rev.json", r#"{"content":"do x","status":"pending","priority":"high"}"#);
        let err = run("review", &f).unwrap_err().to_string();
        assert!(err.contains("action"), "error should mention the missing field: {err}");
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn rejects_bad_action_verb() {
        let f = tmp("rev2.json", r#"{"action":"proceed","reason":"r"}"#);
        let err = run("review", &f).unwrap_err().to_string();
        assert!(err.contains("continue|insert|skip|stop"), "{err}");
        let _ = std::fs::remove_file(&f);
    }
}
