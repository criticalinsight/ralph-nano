use anyhow::Result;
use serde_json::json;

pub struct Janitor;

impl Janitor {
    pub fn extraction_prompt() -> &'static str {
        "You are a Knowledge Graph Extractor.
Analyze the following conversation and extract key technical facts and user preferences.
Output STRICTLY a list of \"Semantic Triples\" in this format:
- (Subject) --[Relation]--> (Object)

Rules:
1. De-duplicate entities (e.g., use \"Ralph\" instead of \"the agent\").
2. Capture technical constraints (e.g., `(Project) --uses--> (Tokio)`).
3. Capture user preferences (e.g., `(User) --dislikes--> (Python Scripts)`).
4. No introduction or prose. List only."
    }

    pub fn parse_triples(content: &str) -> Vec<String> {
        content.lines()
            .map(|l| l.trim())
            .filter(|l| l.starts_with("- (") && l.contains(" --["))
            .map(|l| l.strip_prefix("- ").unwrap_or(l).to_string())
            .collect()
    }
}
