mod emit;
mod llm;
mod orchestrate;
mod prompts;
mod types;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(name = "bootstrap")]
#[command(about = "Drive opencode: plan/execute/review a project, or run an explicit task list")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Plan, implement, and review a project from a spec file (planner→executor→reviewer loop)
    Orchestrate(OrchestrateArgs),
    /// Run an explicit list of tasks, one opencode call per task
    Exec(ExecArgs),
    /// Validate (and canonicalize) a JSON envelope a model wrote — the helper
    /// the prompts tell each role to call from the shell after writing its JSON.
    Emit(EmitArgs),
}

#[derive(Args, Debug)]
struct OrchestrateArgs {
    /// Path to the spec file (goal + description + architectural decisions)
    spec_file: PathBuf,

    /// Model to use, in provider/model format
    #[arg(long, default_value = "llamacpp/qwable")]
    model: String,

    /// Project working directory opencode operates in (and where .bootstrap/ artifacts go)
    #[arg(long, default_value = ".")]
    work_dir: PathBuf,

    /// Max steps to execute in THIS invocation. With --resume, run one step at a
    /// time (e.g. --resume --max-steps 1) to spread a slow build over many calls.
    #[arg(long, default_value = "20")]
    max_steps: usize,

    /// Produce and print the plan, then stop (cheap way to test the planner).
    /// Also saves resumable state so you can run steps later with --resume.
    #[arg(long)]
    plan_only: bool,

    /// Continue from saved state (.bootstrap/state.json) instead of re-planning.
    #[arg(long)]
    resume: bool,

    /// Enable verbose output
    #[arg(long)]
    verbose: bool,
}

#[derive(Args, Debug)]
struct ExecArgs {
    /// One or more tasks; each is sent to opencode as a separate `run`
    #[arg(required = true)]
    tasks: Vec<String>,

    /// Model to use, in provider/model format
    #[arg(long, default_value = "llamacpp/qwable")]
    model: String,

    /// Run each task in its own fresh session (default: chain all tasks via --continue)
    #[arg(long)]
    no_chain: bool,

    /// Enable verbose output
    #[arg(long)]
    verbose: bool,
}

#[derive(Args, Debug)]
struct EmitArgs {
    /// Which envelope to validate: plan | step | review
    role: String,
    /// The JSON file the model wrote (validated and rewritten in place)
    #[arg(long)]
    file: PathBuf,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Commands::Orchestrate(args) => orchestrate::run(&orchestrate::Options {
            spec_file: args.spec_file,
            model: args.model,
            work_dir: args.work_dir,
            max_steps: args.max_steps,
            plan_only: args.plan_only,
            resume: args.resume,
            verbose: args.verbose,
        }),
        Commands::Exec(args) => run_tasks(&args),
        Commands::Emit(args) => emit::run(&args.role, &args.file),
    }
}

/// The previous behavior: run an explicit list of tasks, optionally chained
/// into a single opencode session.
fn run_tasks(args: &ExecArgs) -> Result<()> {
    let total = args.tasks.len();
    for (i, task) in args.tasks.iter().enumerate() {
        let chain = !args.no_chain && i > 0;

        if args.verbose {
            eprintln!(
                "[bootstrap] task {}/{}{}",
                i + 1,
                total,
                if chain { " (continuing session)" } else { "" }
            );
            eprintln!("[bootstrap] > {task}");
        }

        let mut cmd = Command::new("opencode");
        cmd.arg("run").arg(task).arg("--model").arg(&args.model);
        if chain {
            cmd.arg("--continue");
        }
        let output = cmd.output()?;

        if args.verbose {
            eprintln!("[bootstrap] exit code: {}", output.status.code().unwrap_or(-1));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stdout.is_empty() {
            println!("{}", stdout.trim());
        }
        if !stderr.is_empty() {
            eprintln!("{}", stderr.trim());
        }

        if !output.status.success() {
            anyhow::bail!(
                "task {}/{} failed (exit {}); stopping pipeline",
                i + 1,
                total,
                output.status.code().unwrap_or(-1)
            );
        }
    }
    Ok(())
}
