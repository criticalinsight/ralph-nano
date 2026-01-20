//! Codebase Q&A Module
//!
//! Enables natural language questions about the codebase with
//! precise file/line answers leveraging the Knowledge Graph.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A question-answer pair with source references
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAResult {
    pub question: String,
    pub answer: String,
    pub sources: Vec<SourceLocation>,
    pub confidence: f32,
    pub related_concepts: Vec<String>,
}

/// A specific location in the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line_start: usize,
    pub line_end: usize,
    pub snippet: String,
    pub relevance: f32,
}

/// Types of questions the Q&A system can handle
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuestionType {
    Location,      // "Where is X defined?"
    Explanation,   // "How does X work?"
    Usage,         // "How do I use X?"
    Relationship,  // "What depends on X?"
    History,       // "Why was X changed?"
    Comparison,    // "What's the difference between X and Y?"
}

/// The Q&A engine
pub struct CodebaseQA {
    workspace: String,
}

impl CodebaseQA {
    pub fn new(workspace: &str) -> Self {
        Self {
            workspace: workspace.to_string(),
        }
    }

    /// Classify the type of question being asked
    pub fn classify_question(&self, question: &str) -> QuestionType {
        let q = question.to_lowercase();
        
        if q.contains("where") || q.contains("defined") || q.contains("located") || q.contains("find") {
            QuestionType::Location
        } else if q.contains("how does") || q.contains("explain") || q.contains("what does") {
            QuestionType::Explanation
        } else if q.contains("how to") || q.contains("how do i") || q.contains("example") {
            QuestionType::Usage
        } else if q.contains("depends") || q.contains("uses") || q.contains("calls") || q.contains("related") {
            QuestionType::Relationship
        } else if q.contains("why") || q.contains("changed") || q.contains("history") {
            QuestionType::History
        } else if q.contains("difference") || q.contains("compare") || q.contains("vs") {
            QuestionType::Comparison
        } else {
            QuestionType::Explanation  // Default to explanation
        }
    }

    /// Build a specialized prompt based on question type
    pub fn build_qa_prompt(&self, question: &str, context: &str) -> String {
        let question_type = self.classify_question(question);
        
        let instruction = match question_type {
            QuestionType::Location => {
                "Find the exact file(s) and line number(s) where the requested item is defined. 
                 Return the file path, line range, and a brief snippet."
            }
            QuestionType::Explanation => {
                "Explain how the requested component works. Include:
                 1. Its purpose and responsibility
                 2. Key data structures or types
                 3. Important functions or methods
                 4. How it integrates with other parts of the system"
            }
            QuestionType::Usage => {
                "Provide a practical example of how to use the requested component.
                 Include:
                 1. Required imports or setup
                 2. A concrete code example
                 3. Common parameters or configuration
                 4. Typical use cases"
            }
            QuestionType::Relationship => {
                "Analyze the relationships and dependencies:
                 1. What components depend on this?
                 2. What does this depend on?
                 3. Key interfaces or contracts
                 4. Impact of changes"
            }
            QuestionType::History => {
                "Review the history and evolution:
                 1. When was this introduced?
                 2. What significant changes were made?
                 3. Why were changes made?
                 4. Who are the main contributors?"
            }
            QuestionType::Comparison => {
                "Compare the requested items:
                 1. Key similarities
                 2. Important differences
                 3. When to use each
                 4. Trade-offs"
            }
        };

        format!(
            r#"You are a codebase expert assistant. Answer questions about the code with precision.

## Instructions
{}

## Codebase Context
{}

## Question
{}

## Response Format
Provide a clear, concise answer. For locations, use format:
FILE: path/to/file.rs
LINES: start-end
SNIPPET: relevant code

For explanations, use structured paragraphs with code examples where helpful.
"#,
            instruction, context, question
        )
    }

    /// Parse source locations from LLM response
    pub fn parse_sources(&self, response: &str) -> Vec<SourceLocation> {
        let mut sources = Vec::new();
        let mut current_file = String::new();
        let mut current_lines = (0, 0);
        let mut current_snippet = String::new();

        for line in response.lines() {
            let trimmed = line.trim();
            
            if trimmed.starts_with("FILE:") {
                // Save previous if exists
                if !current_file.is_empty() {
                    sources.push(SourceLocation {
                        file: current_file.clone(),
                        line_start: current_lines.0,
                        line_end: current_lines.1,
                        snippet: current_snippet.clone(),
                        relevance: 1.0,
                    });
                }
                current_file = trimmed[5..].trim().to_string();
                current_lines = (0, 0);
                current_snippet.clear();
            } else if trimmed.starts_with("LINES:") {
                let lines_str = trimmed[6..].trim();
                let parts: Vec<&str> = lines_str.split('-').collect();
                if parts.len() == 2 {
                    current_lines.0 = parts[0].trim().parse().unwrap_or(0);
                    current_lines.1 = parts[1].trim().parse().unwrap_or(current_lines.0);
                } else if let Ok(n) = lines_str.parse() {
                    current_lines = (n, n);
                }
            } else if trimmed.starts_with("SNIPPET:") {
                current_snippet = trimmed[8..].trim().to_string();
            }
        }

        // Don't forget the last one
        if !current_file.is_empty() {
            sources.push(SourceLocation {
                file: current_file,
                line_start: current_lines.0,
                line_end: current_lines.1,
                snippet: current_snippet,
                relevance: 1.0,
            });
        }

        sources
    }

    /// Generate related concepts from the response
    pub fn extract_concepts(&self, response: &str) -> Vec<String> {
        let mut concepts = Vec::new();
        
        // Simple extraction of code-like terms (struct names, function names, etc.)
        let code_pattern = regex::Regex::new(r"`([A-Z][a-zA-Z0-9_]+)`").unwrap();
        for cap in code_pattern.captures_iter(response) {
            if let Some(m) = cap.get(1) {
                let concept = m.as_str().to_string();
                if !concepts.contains(&concept) {
                    concepts.push(concept);
                }
            }
        }

        concepts
    }

    /// Common Q&A patterns for code navigation
    pub fn quick_answer(&self, question: &str) -> Option<String> {
        let q = question.to_lowercase();
        
        // Common patterns that can be answered without LLM
        if q.contains("what language") {
            if Path::new(&format!("{}/Cargo.toml", self.workspace)).exists() {
                return Some("This is a **Rust** project (Cargo.toml found).".to_string());
            }
            if Path::new(&format!("{}/package.json", self.workspace)).exists() {
                return Some("This is a **JavaScript/TypeScript** project (package.json found).".to_string());
            }
            if Path::new(&format!("{}/requirements.txt", self.workspace)).exists() {
                return Some("This is a **Python** project (requirements.txt found).".to_string());
            }
        }

        if q.contains("entry point") || q.contains("main function") {
            if Path::new(&format!("{}/src/main.rs", self.workspace)).exists() {
                return Some("The main entry point is `src/main.rs`.".to_string());
            }
            if Path::new(&format!("{}/src/index.ts", self.workspace)).exists() {
                return Some("The main entry point is `src/index.ts`.".to_string());
            }
        }

        None
    }
}
