//! Semantic Linter Module
//!
//! Uses LLM intelligence to find logical bugs, security flaws, and
//! architectural issues that static analysis tools (like clippy) miss.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintViolation {
    pub file: String,
    pub line: usize,
    pub severity: LintSeverity,
    pub message: String,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LintSeverity {
    Critical,   // Security flaws, panics, data loss
    Warning,    // Logical bugs, performance issues
    Suggestion, // Style, readability, best practices
}

pub struct SemanticLinter;

impl SemanticLinter {
    /// Generate the system prompt for the semantic linter
    pub fn lint_prompt(code: &str, context: &str) -> String {
        format!(
            r#"You are a Semantic Code Linter - an AI expert in finding subtle bugs, security vulnerabilities, and logical errors.
Your goal is to find issues that traditional static analysis (like compiler checks or standard linters) CANNOT find.
Focus on:
1. Race conditions or concurrency bugs
2. API misuse or edge cases
3. Logic errors (off-by-one, infinite loops)
4. Security vulnerabilities (injection, auth bypass)
5. Architectural inconsistencies

CONTEXT:
{}

CODE TO LINT:
{}

OUTPUT FORMAT:
Provide a JSON array of violations. If no issues are found, return an empty array [].
Example:
[
  {{
    "file": "src/main.rs",
    "line": 42,
    "severity": "Critical",
    "message": "Potential deadlock here due to lock ordering.",
    "suggestion": "Acquire lock A before lock B."
  }}
]
"#,
            context, code
        )
    }

    /// Parse the LLM response into LintViolations
    pub fn parse_response(response: &str) -> Vec<LintViolation> {
        // Extract JSON from code blocks if present
        let json_str = if let Some(start) = response.find("```json") {
            let code_start = start + 7; // Skip "```json"
            if let Some(end) = response[code_start..].find("```") {
                let s = &response[code_start..][..end];
                s.trim()
            } else {
                response
            }
        } else if let Some(start) = response.find('[') {
            if let Some(end) = response.rfind(']') {
                &response[start..=end]
            } else {
                response
            }
        } else {
            response
        };

        serde_json::from_str::<Vec<LintViolation>>(json_str).unwrap_or_default()
    }
}
