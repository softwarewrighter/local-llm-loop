//! Thin wrapper around the `opencode` CLI plus helpers to pull structured
//! JSON back out of the model's (possibly ANSI-decorated, prose-wrapped) output.

use anyhow::{Context, Result, bail};
use serde::de::DeserializeOwned;
use std::process::Command;

/// A handle that knows how to run one `opencode run` call.
pub struct Llm {
    pub model: String,
    /// Directory opencode operates in (passed as `--dir`); None = current dir.
    pub work_dir: Option<String>,
    pub verbose: bool,
}

impl Llm {
    pub fn new(model: String, work_dir: Option<String>, verbose: bool) -> Self {
        Self { model, work_dir, verbose }
    }

    /// Invoke `opencode run <prompt> --model <model>` and return cleaned stdout.
    /// `label` is only used for verbose logging (e.g. "plan", "step 2", "review").
    pub fn run(&self, prompt: &str, label: &str) -> Result<String> {
        if self.verbose {
            eprintln!("[llm:{label}] calling {} ({} prompt chars)", self.model, prompt.len());
        }

        let mut cmd = Command::new("opencode");
        cmd.arg("run").arg(prompt).arg("--model").arg(&self.model);
        if let Some(dir) = &self.work_dir {
            cmd.arg("--dir").arg(dir);
        }

        let output = cmd
            .output()
            .context("failed to spawn `opencode` (is it installed and on $PATH?)")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if self.verbose {
            eprintln!("[llm:{label}] exit {}", output.status.code().unwrap_or(-1));
            let err = stderr.trim();
            if !err.is_empty() {
                eprintln!("[llm:{label}] stderr: {}", truncate(err, 500));
            }
        }

        if !output.status.success() {
            bail!(
                "opencode run failed for {label} (exit {}): {}",
                output.status.code().unwrap_or(-1),
                truncate(stderr.trim(), 500)
            );
        }

        Ok(strip_ansi(&stdout))
    }
}

/// Remove ANSI/CSI escape sequences (color, cursor moves) so we can scan text.
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next(); // consume '['
                // CSI runs until a final byte in the range @..~
                while let Some(&nc) = chars.peek() {
                    chars.next();
                    if ('@'..='~').contains(&nc) {
                        break;
                    }
                }
            } else {
                // Some other escape (e.g. OSC); drop the following char too.
                chars.next();
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Extract the JSON object from model output. Prefers content between the
/// sentinels; falls back to the first `{` … last `}` span.
pub fn extract_json(text: &str, start_sentinel: &str, end_sentinel: &str) -> Result<String> {
    let slice = match (text.find(start_sentinel), text.rfind(end_sentinel)) {
        (Some(s), Some(e)) if e > s + start_sentinel.len() => &text[s + start_sentinel.len()..e],
        _ => text,
    };

    let start = slice.find('{').context("no '{' found in LLM output")?;
    let end = slice.rfind('}').context("no '}' found in LLM output")?;
    if end < start {
        bail!("malformed JSON braces in LLM output");
    }
    Ok(slice[start..=end].to_string())
}

/// Extract + deserialize in one shot, with a helpful error on failure.
pub fn parse_json<T: DeserializeOwned>(
    text: &str,
    start_sentinel: &str,
    end_sentinel: &str,
) -> Result<T> {
    let json = extract_json(text, start_sentinel, end_sentinel)?;
    serde_json::from_str(&json)
        .with_context(|| format!("failed to parse JSON from LLM output: {}", truncate(&json, 1000)))
}

pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let kept: String = s.chars().take(max).collect();
        format!("{kept}… [truncated {} chars]", s.chars().count() - max)
    }
}
