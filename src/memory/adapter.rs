use crate::memory::MemoryStore;
use crate::memory_legacy::Memory as LegacyMemory;
use async_trait::async_trait;
use anyhow::{Result, Ok};
use serde_json::Value;
use std::sync::Arc;

pub struct LegacyMemoryAdapter {
    inner: Arc<LegacyMemory>,
}

impl LegacyMemoryAdapter {
    pub fn new(inner: Arc<LegacyMemory>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl MemoryStore for LegacyMemoryAdapter {
    async fn recall(&self, query: &str, _limit: usize) -> Result<Vec<String>> {
        // Map to recall_facts
        self.inner.recall_facts(query).await
    }

    async fn store(&self, content: &str, _metadata: Option<Value>) -> Result<String> {
        // Map to store_lesson for now, or just a generic "fact"
        // Legacy memory is specific (cache, node, lesson). 
        // We'll treat generic store as a "heuristic" or "lesson"
        self.inner.store_heuristic(content).await?;
        Ok("stored".to_string())
    }

    async fn forget(&self, _id: &str) -> Result<()> {
        // Not implemented in legacy
        Ok(())
    }
}
