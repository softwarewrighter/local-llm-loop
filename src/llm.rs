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
        // Run under a restricted agent that disables opencode's task/todo tools.
        // Reasoning/creative models (Qwythos, qwen3-8b) wander into `task`
        // (sub-agent spawn) and `todowrite` instead of writing the envelope,
        // which breaks the loop. The `envelope` agent (opencode.json) leaves
        // file/shell tools enabled so coding still works. Override/disable with
        // OPENCODE_AGENT (set empty to skip).
        let agent = std::env::var("OPENCODE_AGENT").unwrap_or_else(|_| "envelope".to_string());
        if !agent.is_empty() {
            cmd.arg("--agent").arg(&agent);
        }
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
/// sentinels; otherwise takes the FIRST complete brace-balanced object.
///
/// We scan from the first `{` to its matching `}` (tracking string literals so
/// braces inside strings don't count) and return that one object — ignoring any
/// trailing prose OR extra objects the model appended. The previous first-`{`…
/// last-`}` span broke on models that emit two back-to-back envelopes (e.g. a
/// reasoning model writing `{...}\n{...}`), which serde then rejects as trailing
/// characters.
pub fn extract_json(text: &str, start_sentinel: &str, end_sentinel: &str) -> Result<String> {
    let slice = match (text.find(start_sentinel), text.rfind(end_sentinel)) {
        (Some(s), Some(e)) if e > s + start_sentinel.len() => &text[s + start_sentinel.len()..e],
        _ => text,
    };

    let start = slice.find('{').context("no '{' found in LLM output")?;
    let bytes = slice.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    let mut end = None;
    // Only ASCII structural bytes are matched; UTF-8 continuation bytes (>= 0x80)
    // never collide with them, so byte scanning is safe and stays on char
    // boundaries (`start`/`end` land on `{`/`}`).
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
        } else {
            match b {
                b'"' => in_string = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    let end = end.context("no matching '}' for the first '{' in LLM output")?;
    Ok(slice[start..=end].to_string())
}

/// Relax common LLM JSON malformations that `serde_json` rejects but a human
/// reading the output would accept (Postel's law: "be liberal in what you
/// accept"). The harness owns this parse, so this is the right layer to be
/// tolerant — see docs/nvidia-3090-poc-summary.md.
///
/// Currently repairs **trailing commas** before `}` or `]` (the exact failure
/// mode observed with Qwen3-14B). The scan is **string-aware**: it tracks
/// string literals (honoring `\` escapes) so commas *inside* string values are
/// never touched. Only unambiguous, safe repairs are made — things like
/// unescaped inner quotes are left alone because repairing them would require
/// guessing the author's intent.
pub fn relax_json(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(s.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut idx = 0;
    while idx < n {
        let c = chars[idx];
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            idx += 1;
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            idx += 1;
            continue;
        }
        if c == ',' {
            // A comma is "trailing" if the next non-whitespace char closes a
            // container. Skip (drop) it in that case.
            let mut j = idx + 1;
            while j < n && chars[j].is_whitespace() {
                j += 1;
            }
            if j < n && (chars[j] == '}' || chars[j] == ']') {
                idx += 1;
                continue;
            }
        }
        out.push(c);
        idx += 1;
    }
    out
}

/// Extract + deserialize in one shot, with a helpful error on failure. Tries a
/// strict parse first; on failure, retries against [`relax_json`] (liberal
/// acceptance) so a stray trailing comma doesn't abort an otherwise-good run.
pub fn parse_json<T: DeserializeOwned>(
    text: &str,
    start_sentinel: &str,
    end_sentinel: &str,
) -> Result<T> {
    let json = extract_json(text, start_sentinel, end_sentinel)?;
    match serde_json::from_str(&json) {
        Ok(value) => Ok(value),
        Err(strict_err) => {
            let relaxed = relax_json(&json);
            serde_json::from_str(&relaxed).with_context(|| {
                format!(
                    "failed to parse JSON from LLM output (strict error: {strict_err}); \
                     even after relaxing: {}",
                    truncate(&relaxed, 1000)
                )
            })
        }
    }
}

pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let kept: String = s.chars().take(max).collect();
        format!("{kept}… [truncated {} chars]", s.chars().count() - max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Step {
        id: u32,
        title: String,
    }
    #[derive(Debug, Deserialize, PartialEq)]
    struct Plan {
        design: String,
        steps: Vec<Step>,
    }

    #[test]
    fn relax_strips_trailing_comma_in_object() {
        assert_eq!(relax_json(r#"{"a":1,}"#), r#"{"a":1}"#);
    }

    #[test]
    fn relax_strips_trailing_comma_in_array() {
        assert_eq!(relax_json(r#"[1,2,3,]"#), r#"[1,2,3]"#);
    }

    #[test]
    fn relax_strips_trailing_comma_with_whitespace() {
        assert_eq!(relax_json("{\"a\":1,\n  }"), "{\"a\":1\n  }");
    }

    #[test]
    fn relax_keeps_commas_between_elements() {
        let s = r#"{"a":1,"b":2}"#;
        assert_eq!(relax_json(s), s);
    }

    #[test]
    fn relax_never_touches_commas_inside_strings() {
        // A comma right before a "}" but *inside* a string must be preserved.
        let s = r#"{"msg":"a, b,","n":1}"#;
        assert_eq!(relax_json(s), s);
        // Escaped quote inside the string must not end the string early.
        let s2 = r#"{"q":"he said \"hi,\"","n":2}"#;
        assert_eq!(relax_json(s2), s2);
    }

    #[test]
    fn parse_json_recovers_from_trailing_comma() {
        // The exact Qwen3-14B failure: a trailing comma after the last field.
        let raw = r#"{"design":"d","steps":[{"id":1,"title":"t",},]}"#;
        let plan: Plan = parse_json(raw, "<<<X>>>", "<<<Y>>>").unwrap();
        assert_eq!(
            plan,
            Plan { design: "d".into(), steps: vec![Step { id: 1, title: "t".into() }] }
        );
    }

    #[test]
    fn parse_json_strict_path_still_works() {
        let raw = r#"prose {"design":"d","steps":[]} trailing"#;
        let plan: Plan = parse_json(raw, "<<<X>>>", "<<<Y>>>").unwrap();
        assert_eq!(plan, Plan { design: "d".into(), steps: vec![] });
    }

    #[test]
    fn extract_takes_first_object_when_model_appends_a_second() {
        // The exact Qwythos break: two back-to-back objects. Take the first;
        // first-`{`..last-`}` would have spanned both → serde "trailing chars".
        let raw = "{\"design\":\"a\",\"steps\":[]}\n{\"design\":\"b\",\"steps\":[]}";
        assert_eq!(extract_json(raw, "<<<X>>>", "<<<Y>>>").unwrap(), "{\"design\":\"a\",\"steps\":[]}");
        let plan: Plan = parse_json(raw, "<<<X>>>", "<<<Y>>>").unwrap();
        assert_eq!(plan.design, "a");
    }

    #[test]
    fn extract_ignores_braces_inside_strings_and_nests() {
        let raw = r#"{"design":"use {curly} and \"quotes\"","steps":[{"id":1,"title":"t"}]} junk"#;
        let got = extract_json(raw, "<<<X>>>", "<<<Y>>>").unwrap();
        assert_eq!(got, r#"{"design":"use {curly} and \"quotes\"","steps":[{"id":1,"title":"t"}]}"#);
    }
}
