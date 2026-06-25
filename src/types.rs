//! Serialized data structures exchanged with the LLM and persisted to disk.
//!
//! The orchestrator owns all of this state in Rust; the LLM only ever sees it
//! as JSON embedded in prompts and returns JSON we parse back into these types.

use serde::{Deserialize, Serialize};

/// A single implementation step in the plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub id: u32,
    pub title: String,
    /// What the executor agent must do (tool-using).
    #[serde(default)]
    pub instructions: String,
    /// Any specifics the executor needs (paths, signatures, decisions).
    #[serde(default)]
    pub context: String,
    /// How to know the step is complete.
    #[serde(default)]
    pub acceptance: String,
}

/// The plan produced by the planner LLM from the spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    #[serde(default)]
    pub design: String,
    #[serde(default)]
    pub notes: String,
    pub steps: Vec<Step>,
}

/// The structured result the executor LLM reports after performing a step.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepResult {
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub new_or_changed_files: Vec<String>,
    #[serde(default)]
    pub changes: String,
    #[serde(default)]
    pub features: Vec<String>,
    #[serde(default)]
    pub bugs: Vec<String>,
    #[serde(default)]
    pub issues: Vec<String>,
    /// "completed" | "partial" | "blocked" | "unknown"
    #[serde(default)]
    pub status: String,
}

/// One executed step paired with the result the executor reported for it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub step: Step,
    pub result: StepResult,
}

/// The full resumable state of an orchestrate run, persisted to
/// `.bootstrap/state.json` after every step so the loop can be stopped and
/// continued (`--resume`) — e.g. to run one step at a time on a slow model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopState {
    pub design: String,
    #[serde(default)]
    pub notes: String,
    /// The working step list (mutated in place by review insert/skip).
    pub steps: Vec<Step>,
    /// Index of the next step to execute.
    #[serde(default)]
    pub cursor: usize,
    /// Total steps executed so far across all invocations.
    #[serde(default)]
    pub executed: usize,
    #[serde(default)]
    pub history: Vec<HistoryEntry>,
    /// True once the loop has run to completion or stopped.
    #[serde(default)]
    pub finished: bool,
    /// Set when the loop halted via a reviewer Stop.
    #[serde(default)]
    pub stopped_reason: Option<String>,
}

/// What the supervising (top) LLM decides to do after reviewing a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewAction {
    /// Proceed to the next planned step, unchanged.
    Continue,
    /// Insert one or more new steps to run next, then continue.
    Insert,
    /// Skip the next planned step.
    Skip,
    /// Halt and request human intervention.
    Stop,
}

impl ReviewAction {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "continue" => ReviewAction::Continue,
            "insert" => ReviewAction::Insert,
            "skip" => ReviewAction::Skip,
            // Anything unrecognized is treated as a stop, for safety.
            _ => ReviewAction::Stop,
        }
    }
}

/// The review decision as returned by the top LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewDecision {
    /// continue | insert | skip | stop
    pub action: String,
    #[serde(default)]
    pub reason: String,
    /// Required only when action == "insert".
    #[serde(default)]
    pub new_steps: Vec<Step>,
}
