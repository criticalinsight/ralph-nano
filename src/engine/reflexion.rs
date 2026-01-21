use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "action", content = "args")]
pub enum AgentAction {
    WriteFile { path: PathBuf, content: String },
    ExecShell { cmd: String },
    Think { thought: String },
    Stop,
}

pub struct ReflexionEngine;

impl Default for ReflexionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ReflexionEngine {
    pub fn new() -> Self {
        Self
    }
    
    // Future methods: critique_plan, validate_action
}
