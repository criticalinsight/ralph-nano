// src/reflexion.rs - Self-Critique and Behavioral Learning for Ralph-Nano v1.1.0

use regex::Regex;

pub struct Reflexion;

impl Reflexion {
    /// Returns the system prompt for self-critique after task completion.
    pub fn critique_prompt() -> &'static str {
        r#"You are a meta-cognitive agent reviewing your recent actions.
Analyze the conversation above and extract behavioral lessons.

For each lesson, output a line in this format:
HEURISTIC: <lesson>

Examples:
HEURISTIC: When editing Rust, always run `cargo check` before committing.
HEURISTIC: User prefers concise diffs over full file rewrites.
HEURISTIC: This codebase uses `anyhow` for error handling, not `Result<T, E>`.

Focus on:
1. What worked well and should be repeated.
2. What failed and should be avoided.
3. User preferences inferred from feedback.
4. Codebase patterns worth remembering.

Output ONLY the HEURISTIC lines, nothing else."#
    }

    /// Parses the LLM response and extracts heuristics.
    pub fn parse_heuristics(response: &str) -> Vec<String> {
        let re = match Regex::new(r"(?m)^HEURISTIC:\s*(.+)$") {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        re.captures_iter(response)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().trim().to_string()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_heuristics() {
        let response = r#"
HEURISTIC: Always check for null values.
Some random text.
HEURISTIC: User prefers dark mode.
"#;
        let heuristics = Reflexion::parse_heuristics(response);
        assert_eq!(heuristics.len(), 2);
        assert_eq!(heuristics[0], "Always check for null values.");
        assert_eq!(heuristics[1], "User prefers dark mode.");
    }
}
