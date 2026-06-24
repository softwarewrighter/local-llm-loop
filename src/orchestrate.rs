//! The planner → executor → reviewer control loop.
//!
//! Rust owns the plan, the step cursor, and the running history. Each LLM call
//! is a stateless `opencode run`; all needed context is passed in the prompt.

use crate::llm::{self, Llm};
use crate::prompts;
use crate::types::{Plan, ReviewAction, ReviewDecision, Step, StepResult};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub struct Options {
    pub spec_file: PathBuf,
    pub model: String,
    pub work_dir: PathBuf,
    pub max_steps: usize,
    pub plan_only: bool,
    pub verbose: bool,
}

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

    // ---- PLAN ---------------------------------------------------------------
    println!("== PLAN == asking {} to design a plan from the spec…", opts.model);
    let plan = plan(&llm, &spec)?;
    persist_json(&art_dir.join("plan.json"), &plan)?;
    print_plan(&plan);

    if opts.plan_only {
        println!("\n[--plan-only] stopping after planning. Plan saved to {}", art_dir.join("plan.json").display());
        return Ok(());
    }

    // ---- EXECUTE + REVIEW LOOP ---------------------------------------------
    let mut steps: Vec<Step> = plan.steps.clone();
    let mut history: Vec<(Step, StepResult)> = Vec::new();
    let mut i = 0usize;
    let mut executed = 0usize;

    while i < steps.len() {
        if executed >= opts.max_steps {
            println!(
                "\n!! reached --max-steps ({}). Stopping to avoid a runaway loop.",
                opts.max_steps
            );
            break;
        }

        let step = steps[i].clone();
        println!("\n== STEP {}/{} == [{}] {}", i + 1, steps.len(), step.id, step.title);

        // EXECUTE
        let (result, raw_result_json) = execute(&llm, &spec, &plan.design, &history, &step)?;
        persist_json(&art_dir.join(format!("step-{:02}-result.json", step.id)), &result)?;
        print_result(&result);
        history.push((step.clone(), result));
        executed += 1;

        // REVIEW
        let remaining: Vec<Step> = steps.get(i + 1..).map(|s| s.to_vec()).unwrap_or_default();
        let decision = review(&llm, &spec, &plan.design, &step, &raw_result_json, &remaining)?;
        persist_json(&art_dir.join(format!("step-{:02}-review.json", step.id)), &decision)?;

        let action = ReviewAction::parse(&decision.action);
        println!("   review → {:?}: {}", action, decision.reason.trim());

        match action {
            ReviewAction::Continue => {
                i += 1;
            }
            ReviewAction::Insert => {
                let mut inserted = decision.new_steps.clone();
                if inserted.is_empty() {
                    println!("   (insert requested but no new_steps provided — continuing)");
                    i += 1;
                } else {
                    println!("   inserting {} new step(s) to run next", inserted.len());
                    let at = i + 1;
                    for (k, s) in inserted.iter_mut().enumerate() {
                        steps.insert(at + k, s.clone());
                    }
                    i += 1; // advance into the inserted steps
                }
            }
            ReviewAction::Skip => {
                if i + 1 < steps.len() {
                    let skipped = steps.remove(i + 1);
                    println!("   skipping next step [{}] {}", skipped.id, skipped.title);
                }
                i += 1;
            }
            ReviewAction::Stop => {
                println!("\n>> HUMAN INTERVENTION REQUESTED — halting build.");
                println!("   reason: {}", decision.reason.trim());
                println!("   completed {executed} step(s). State saved under {}", art_dir.display());
                return Ok(());
            }
        }
    }

    println!("\n== DONE == executed {executed} step(s). Artifacts in {}", art_dir.display());
    summarize(&history);
    Ok(())
}

fn plan(llm: &Llm, spec: &str) -> Result<Plan> {
    let out = llm.run(&prompts::plan_prompt(spec), "plan")?;
    llm::parse_json::<Plan>(&out, prompts::PLAN_START, prompts::PLAN_END)
        .context("planner did not return a valid plan")
}

/// Returns the parsed result plus the raw extracted JSON (handed to the reviewer
/// verbatim so it sees exactly what the executor reported).
fn execute(
    llm: &Llm,
    spec: &str,
    design: &str,
    history: &[(Step, StepResult)],
    step: &Step,
) -> Result<(StepResult, String)> {
    let out = llm.run(&prompts::step_prompt(spec, design, history, step), &format!("step {}", step.id))?;
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
    spec: &str,
    design: &str,
    step: &Step,
    result_json: &str,
    remaining: &[Step],
) -> Result<ReviewDecision> {
    let out = llm.run(
        &prompts::review_prompt(spec, design, step, result_json, remaining),
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

fn summarize(history: &[(Step, StepResult)]) {
    for (s, r) in history {
        println!("  [{}] {} → {}", s.id, s.title, if r.status.is_empty() { "unknown" } else { &r.status });
    }
}
