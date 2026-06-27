//! The planner → executor → reviewer control loop.
//!
//! Rust owns the plan, the step cursor, and the running history. Each LLM call
//! is a stateless `opencode run`; all needed context is passed in the prompt.

use crate::llm::{self, Llm};
use crate::prompts;
use crate::types::{HistoryEntry, LoopState, Plan, ReviewAction, ReviewDecision, Step, StepResult};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub struct Options {
    pub spec_file: PathBuf,
    pub model: String,
    pub work_dir: PathBuf,
    /// Max steps to execute in THIS invocation (a chunk). With `--resume` you can
    /// run one step at a time across many invocations.
    pub max_steps: usize,
    pub plan_only: bool,
    /// Continue from `.bootstrap/state.json` instead of planning afresh.
    pub resume: bool,
    pub verbose: bool,
}

/// Absolute safety cap on total steps executed across all resumes (guards
/// against an unbounded insert→insert→… loop on a long-running build).
const MAX_TOTAL_STEPS: usize = 50;

pub fn run(opts: &Options) -> Result<()> {
    let spec = std::fs::read_to_string(&opts.spec_file)
        .with_context(|| format!("failed to read spec file {}", opts.spec_file.display()))?;
    if spec.trim().is_empty() {
        anyhow::bail!("spec file {} is empty", opts.spec_file.display());
    }

    let art_dir = opts.work_dir.join(".bootstrap");
    std::fs::create_dir_all(&art_dir)
        .with_context(|| format!("failed to create artifact dir {}", art_dir.display()))?;

    let llm = Llm::new(
        opts.model.clone(),
        Some(opts.work_dir.to_string_lossy().to_string()),
        opts.verbose,
    );

    let state_path = art_dir.join("state.json");

    // ---- PLAN (or load saved state with --resume) ---------------------------
    let mut state: LoopState = if opts.resume {
        let s = load_state(&state_path)?;
        println!("== RESUME == loaded saved state: {} step(s) executed, cursor at {}/{}",
            s.executed, s.cursor, s.steps.len());
        s
    } else {
        println!("== PLAN == asking {} to design a plan from the spec…", opts.model);
        let plan = plan(&llm, &opts.work_dir, &spec)?;
        persist_json(&art_dir.join("plan.json"), &plan)?;
        print_plan(&plan);
        let state = LoopState {
            design: plan.design,
            notes: plan.notes,
            steps: plan.steps,
            cursor: 0,
            executed: 0,
            history: Vec::new(),
            finished: false,
            stopped_reason: None,
        };
        persist_json(&state_path, &state)?; // so --resume can pick up even after --plan-only
        state
    };

    if opts.plan_only {
        println!("\n[--plan-only] plan + state saved to {}.\nRun the steps with: --resume --max-steps 1 (one step per invocation).", art_dir.display());
        return Ok(());
    }

    if state.finished {
        println!("\n== ALREADY FINISHED == nothing to do (executed {} step(s)).", state.executed);
        if let Some(r) = &state.stopped_reason {
            println!("   (had stopped for human review: {r})");
        }
        summarize(&state.history);
        return Ok(());
    }

    // ---- EXECUTE + REVIEW LOOP (up to opts.max_steps THIS invocation) --------
    let mut ran = 0usize;
    while state.cursor < state.steps.len() {
        if ran >= opts.max_steps {
            println!("\n== PAUSED == ran {ran} step(s) this invocation (--max-steps {}).", opts.max_steps);
            println!("   {} of {} planned step(s) done. Resume the next with: --resume --max-steps 1",
                state.cursor, state.steps.len());
            break;
        }
        if state.executed >= MAX_TOTAL_STEPS {
            println!("\n!! reached absolute safety cap ({MAX_TOTAL_STEPS} total steps). Stopping.");
            state.finished = true;
            break;
        }

        let i = state.cursor;
        let step = state.steps[i].clone();
        println!("\n== STEP {}/{} == [{}] {}", i + 1, state.steps.len(), step.id, step.title);

        // EXECUTE
        let (result, raw_result_json) =
            execute(&llm, &opts.work_dir, &spec, &state.design, &state.history, &step)?;
        persist_json(&art_dir.join(format!("step-{:02}-result.json", step.id)), &result)?;
        print_result(&result);
        state.history.push(HistoryEntry { step: step.clone(), result });
        state.executed += 1;
        ran += 1;

        // REVIEW
        let remaining: Vec<Step> = state.steps.get(i + 1..).map(|s| s.to_vec()).unwrap_or_default();
        let decision = review(&llm, &opts.work_dir, &spec, &state.design, &step, &raw_result_json, &remaining)?;
        persist_json(&art_dir.join(format!("step-{:02}-review.json", step.id)), &decision)?;

        let action = ReviewAction::parse(&decision.action);
        println!("   review → {:?}: {}", action, decision.reason.trim());

        match action {
            ReviewAction::Continue => state.cursor += 1,
            ReviewAction::Insert => {
                let inserted = decision.new_steps.clone();
                if inserted.is_empty() {
                    println!("   (insert requested but no new_steps provided — continuing)");
                } else {
                    println!("   inserting {} new step(s) to run next", inserted.len());
                    let at = i + 1;
                    for (k, s) in inserted.into_iter().enumerate() {
                        state.steps.insert(at + k, s);
                    }
                }
                state.cursor += 1; // advance into the inserted steps
            }
            ReviewAction::Skip => {
                if i + 1 < state.steps.len() {
                    let skipped = state.steps.remove(i + 1);
                    println!("   skipping next step [{}] {}", skipped.id, skipped.title);
                }
                state.cursor += 1;
            }
            ReviewAction::Stop => {
                state.finished = true;
                state.stopped_reason = Some(decision.reason.trim().to_string());
                persist_json(&state_path, &state)?;
                println!("\n>> HUMAN INTERVENTION REQUESTED — halting build.");
                println!("   reason: {}", decision.reason.trim());
                println!("   completed {} step(s). State saved at {}", state.executed, state_path.display());
                return Ok(());
            }
        }

        // Persist after each fully-processed step so we can resume exactly here.
        persist_json(&state_path, &state)?;
    }

    if state.cursor >= state.steps.len() {
        state.finished = true;
        persist_json(&state_path, &state)?;
        println!("\n== DONE == all {} planned step(s) processed ({} executed). Artifacts in {}",
            state.steps.len(), state.executed, art_dir.display());
        summarize(&state.history);
    }
    Ok(())
}

fn load_state(path: &Path) -> Result<LoopState> {
    let s = std::fs::read_to_string(path).with_context(|| {
        format!("no saved state at {} — run without --resume first (e.g. --plan-only)", path.display())
    })?;
    serde_json::from_str(&s).with_context(|| format!("failed to parse saved state {}", path.display()))
}

/// Bare in-dir filename each role is told to write its JSON to. A `.dotdir/`
/// prefix made some models (Qwen3-Coder) rewrite it to an un-writable absolute
/// path (`/bootstrap/…`), which opencode then rejected; a bare name avoids that.
const PLAN_OUT: &str = "plan.llm.json";

/// Run one stateless `opencode` call whose JSON answer is delivered via a file.
///
/// Modern opencode is agentic and reliably *writes a file* rather than printing
/// JSON, so we point it at `rel_out` and read it back. Resolution is robust, in
/// priority order:
///   1. the file at the path we asked for;
///   2. a file the reply *points to or implies* — a path it named (even an
///      absolute one it invented), or the same basename written elsewhere in the
///      work dir;
///   3. the reply text itself (the model answered inline instead of writing).
/// Stale files are removed first so a failed call can't read an old answer.
/// Max envelope attempts per role: the first try plus in-context-RL retries.
const MAX_ENVELOPE_TRIES: usize = 3;

/// Run one opencode call, resolve the envelope text, and validate it with
/// `check` (the role's parser = the reward signal). On failure, **re-prompt with
/// the precise error fed back** so the model corrects its next attempt — in-
/// context reinforcement learning, no weight updates (see
/// docs/nvidia-5060-poc-summary.md / the ICL+ICRL rationale). Always returns the
/// best text after the attempts; the caller still parses it, so a final failure
/// degrades exactly as before.
fn capture_json(
    llm: &Llm,
    work_dir: &Path,
    rel_out: &str,
    role: &str,
    base_prompt: &str,
    label: &str,
    check: impl Fn(&str) -> Result<()>,
) -> Result<String> {
    let abs = work_dir.join(rel_out);
    let mut prompt = std::borrow::Cow::Borrowed(base_prompt);
    let mut text = String::new();
    for attempt in 1..=MAX_ENVELOPE_TRIES {
        let _ = std::fs::remove_file(&abs);
        let stdout = llm.run(&prompt, &format!("{label} try {attempt}/{MAX_ENVELOPE_TRIES}"))?;
        text = read_nonempty(&abs)
            .or_else(|| recover_written_file(work_dir, rel_out, &stdout))
            .unwrap_or(stdout);
        match check(&text) {
            Ok(()) => break,
            Err(e) if attempt < MAX_ENVELOPE_TRIES => {
                prompt = std::borrow::Cow::Owned(format!(
                    "{base_prompt}\n\n--- YOUR PREVIOUS ATTEMPT WAS REJECTED ---\n{}\n\
                     Fix exactly that problem. Write the corrected single JSON object to \
                     `{rel_out}` (overwrite it), then run `bootstrap emit {role} --file {rel_out}` \
                     again. One bare object only — no wrapper, no second object, no task/todo tools.",
                    llm::truncate(&e.to_string(), 400),
                ));
            }
            Err(_) => break, // retries exhausted: return best effort, caller degrades
        }
    }
    Ok(text)
}

fn read_nonempty(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok().filter(|s| !s.trim().is_empty())
}

/// Recover JSON the model wrote to a file *other* than the one we asked for — it
/// may have named a different path in its reply, or written the right basename
/// to a different in-dir spot. Tries every path the reply references that ends
/// in our basename (as-is, and re-rooted under the work dir to defang an
/// absolute-path rewrite), then the basename at the work-dir root and `.bootstrap/`.
fn recover_written_file(work_dir: &Path, rel_out: &str, stdout: &str) -> Option<String> {
    let basename = Path::new(rel_out).file_name()?.to_str()?;
    for path in referenced_paths(stdout, basename) {
        if let Some(s) = read_nonempty(Path::new(&path)) {
            return Some(s);
        }
        if let Some(s) = read_nonempty(&work_dir.join(path.trim_start_matches('/'))) {
            return Some(s);
        }
    }
    for dir in [work_dir.to_path_buf(), work_dir.join(".bootstrap")] {
        if let Some(s) = read_nonempty(&dir.join(basename)) {
            return Some(s);
        }
    }
    None
}

/// Tokens in `text` that name a file ending in `basename` — either the bare
/// basename, or a `/`-or-`\`-separated path ending in it (so a `Write
/// /bootstrap/plan.llm.json` line yields `/bootstrap/plan.llm.json`, but
/// `myplan.llm.json` is not mistaken for `plan.llm.json`).
fn referenced_paths(text: &str, basename: &str) -> Vec<String> {
    let mut out = Vec::new();
    for tok in text.split(|c: char| c.is_whitespace() || matches!(c, '`' | '"' | '\'' | '(' | ')')) {
        let tok = tok.trim_end_matches(|c| matches!(c, '.' | ',' | ':' | ';'));
        let is_path = tok.len() > basename.len()
            && tok.ends_with(basename)
            && matches!(tok.as_bytes()[tok.len() - basename.len() - 1], b'/' | b'\\');
        if tok == basename || is_path {
            out.push(tok.to_string());
        }
    }
    out.sort();
    out.dedup();
    out
}

fn plan(llm: &Llm, work_dir: &Path, spec: &str) -> Result<Plan> {
    let out = capture_json(
        llm,
        work_dir,
        PLAN_OUT,
        "plan",
        &prompts::plan_prompt(spec, PLAN_OUT),
        "plan",
        |t| llm::parse_json::<Plan>(t, prompts::PLAN_START, prompts::PLAN_END).map(|_: Plan| ()),
    )?;
    llm::parse_json::<Plan>(&out, prompts::PLAN_START, prompts::PLAN_END)
        .context("planner did not return a valid plan")
}

/// Returns the parsed result plus the raw extracted JSON (handed to the reviewer
/// verbatim so it sees exactly what the executor reported).
fn execute(
    llm: &Llm,
    work_dir: &Path,
    spec: &str,
    design: &str,
    history: &[HistoryEntry],
    step: &Step,
) -> Result<(StepResult, String)> {
    let rel_out = format!("step-{:02}-result.llm.json", step.id);
    let out = capture_json(
        llm,
        work_dir,
        &rel_out,
        "step",
        &prompts::step_prompt(spec, design, history, step, &rel_out),
        &format!("step {}", step.id),
        |t| {
            llm::extract_json(t, prompts::STEP_START, prompts::STEP_END)
                .and_then(|j| serde_json::from_str::<StepResult>(&j).map_err(anyhow::Error::from))
                .map(|_| ())
        },
    )?;
    match llm::extract_json(&out, prompts::STEP_START, prompts::STEP_END) {
        Ok(json) => {
            let result: StepResult = serde_json::from_str(&json).unwrap_or_else(|_| StepResult {
                summary: llm::truncate(out.trim(), 300),
                status: "unknown".to_string(),
                ..Default::default()
            });
            Ok((result, json))
        }
        Err(_) => {
            // No envelope found — record what we can and let the reviewer judge.
            let result = StepResult {
                summary: llm::truncate(out.trim(), 300),
                status: "unknown".to_string(),
                issues: vec!["executor did not emit a JSON result envelope".to_string()],
                ..Default::default()
            };
            let json = serde_json::to_string_pretty(&result)?;
            Ok((result, json))
        }
    }
}

fn review(
    llm: &Llm,
    work_dir: &Path,
    spec: &str,
    design: &str,
    step: &Step,
    result_json: &str,
    remaining: &[Step],
) -> Result<ReviewDecision> {
    let rel_out = format!("step-{:02}-review.llm.json", step.id);
    let out = capture_json(
        llm,
        work_dir,
        &rel_out,
        "review",
        &prompts::review_prompt(spec, design, step, result_json, remaining, &rel_out),
        &format!("review {}", step.id),
        |t| llm::parse_json::<ReviewDecision>(t, prompts::REVIEW_START, prompts::REVIEW_END).map(|_: ReviewDecision| ()),
    )?;
    match llm::parse_json::<ReviewDecision>(&out, prompts::REVIEW_START, prompts::REVIEW_END) {
        Ok(d) => Ok(d),
        Err(e) => {
            // If the supervisor's verdict is unparseable, fail safe: stop.
            Ok(ReviewDecision {
                action: "stop".to_string(),
                reason: format!("could not parse review decision ({e}); stopping for human review"),
                new_steps: Vec::new(),
            })
        }
    }
}

fn persist_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(value)?;
    std::fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn print_plan(plan: &Plan) {
    println!("\n  design: {}", plan.design.trim());
    if !plan.notes.trim().is_empty() {
        println!("  notes:  {}", plan.notes.trim());
    }
    println!("  steps ({}):", plan.steps.len());
    for s in &plan.steps {
        println!("    {}. {}", s.id, s.title);
    }
}

fn print_result(r: &StepResult) {
    println!("   result [{}]: {}", if r.status.is_empty() { "unknown" } else { &r.status }, r.summary.trim());
    if !r.new_or_changed_files.is_empty() {
        println!("   files: {}", r.new_or_changed_files.join(", "));
    }
    if !r.issues.is_empty() {
        println!("   issues: {}", r.issues.join("; "));
    }
    if !r.bugs.is_empty() {
        println!("   bugs: {}", r.bugs.join("; "));
    }
}

fn summarize(history: &[HistoryEntry]) {
    for h in history {
        let (s, r) = (&h.step, &h.result);
        println!("  [{}] {} → {}", s.id, s.title, if r.status.is_empty() { "unknown" } else { &r.status });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn referenced_paths_finds_absolute_and_bare() {
        // The exact opencode failure line names an absolute path.
        let s = "! permission requested: external_directory (/bootstrap/*)\n\u{2717} Write /bootstrap/plan.llm.json failed";
        assert_eq!(referenced_paths(s, "plan.llm.json"), vec!["/bootstrap/plan.llm.json"]);
        // A bare mention is captured too.
        assert_eq!(referenced_paths("wrote plan.llm.json done", "plan.llm.json"), vec!["plan.llm.json"]);
    }

    #[test]
    fn referenced_paths_ignores_basename_suffix_of_another_name() {
        // `myplan.llm.json` must NOT match basename `plan.llm.json`.
        assert!(referenced_paths("saved to myplan.llm.json", "plan.llm.json").is_empty());
    }

    #[test]
    fn recover_reads_file_at_workdir_root() {
        let dir = std::env::temp_dir().join(format!("orch-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("plan.llm.json"), r#"{"design":"d","steps":[]}"#).unwrap();
        let got = recover_written_file(&dir, "plan.llm.json", "(no path mentioned)");
        assert_eq!(got.as_deref(), Some(r#"{"design":"d","steps":[]}"#));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
