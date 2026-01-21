use std::sync::Arc;
// use tokio::sync::RwLock;
use crate::memory::MemoryStore;
use crate::safety::OverlayFS;
use super::cortex::Cortex;


use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum RalphRole {
    Supervisor,
    #[default]
    Executor,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RalphConfig {
    pub project_name: String,
    pub primary_model: String,
    pub fallback_model: String,
    #[serde(default)]
    pub autonomous_mode: bool,
    #[serde(default)]
    pub max_autonomous_loops: usize,
    #[serde(default)]
    pub role: RalphRole,
}

impl Default for RalphConfig {
    fn default() -> Self {
        Self {
            project_name: "ralph-nano".to_string(),
            primary_model: "gemini-1.5-pro".to_string(),
            fallback_model: "gemini-1.5-pro".to_string(),
            autonomous_mode: false,
            max_autonomous_loops: 50,
            role: RalphRole::Executor,
        }
    }
}

pub struct GlobalState {
    pub config: Arc<RalphConfig>,
    pub memory: Arc<dyn MemoryStore>,
    pub memory_raw: Arc<crate::memory_legacy::Memory>,
    pub overlay: Arc<OverlayFS>,
    pub cortex: Arc<Cortex>,
}

impl GlobalState {
    pub fn new(
        config: RalphConfig,
        memory: Arc<dyn MemoryStore>,
        memory_raw: Arc<crate::memory_legacy::Memory>,
        overlay: Arc<OverlayFS>,
        cortex: Arc<Cortex>,
    ) -> Self {
        Self {
            config: Arc::new(config),
            memory,
            memory_raw,
            overlay,
            cortex,
        }
    }
}
