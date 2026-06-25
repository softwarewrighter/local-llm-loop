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

/// Relative path (under the work dir) where each role is told to write its JSON.
const PLAN_OUT: &str = ".bootstrap/plan.llm.json";

/// Run one stateless `opencode` call whose JSON answer is delivered via a file.
///
/// Modern opencode is agentic and reliably *writes a file* rather than printing
/// JSON, so we point it at `rel_out` and read that back. We delete any stale file
/// first (so a failed call can't read an old answer) and fall back to scraping
/// stdout if the model answered inline instead of writing the file.
fn capture_json(llm: &Llm, work_dir: &Path, rel_out: &str, prompt: &str, label: &str) -> Result<String> {
    let abs = work_dir.join(rel_out);
    let _ = std::fs::remove_file(&abs);
    let stdout = llm.run(prompt, label)?;
    match std::fs::read_to_string(&abs) {
        Ok(s) if !s.trim().is_empty() => Ok(s),
        _ => Ok(stdout),
    }
}

fn plan(llm: &Llm, work_dir: &Path, spec: &str) -> Result<Plan> {
    let out = capture_json(llm, work_dir, PLAN_OUT, &prompts::plan_prompt(spec, PLAN_OUT), "plan")?;
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
    let rel_out = format!(".bootstrap/step-{:02}-result.llm.json", step.id);
    let out = capture_json(
        llm,
        work_dir,
        &rel_out,
        &prompts::step_prompt(spec, design, history, step, &rel_out),
        &format!("step {}", step.id),
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
    let rel_out = format!(".bootstrap/step-{:02}-review.llm.json", step.id);
    let out = capture_json(
        llm,
        work_dir,
        &rel_out,
        &prompts::review_prompt(spec, design, step, result_json, remaining, &rel_out),
        &format!("review {}", step.id),
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
