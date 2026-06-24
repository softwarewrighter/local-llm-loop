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
