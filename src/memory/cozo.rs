use crate::memory::MemoryStore;
use async_trait::async_trait;
use anyhow::Result;
use serde_json::Value;

pub struct CozoMemory;

#[async_trait]
impl MemoryStore for CozoMemory {
    async fn recall(&self, _query: &str, _limit: usize) -> Result<Vec<String>> {
        // Implementation to come in Phase B
        Ok(vec![])
    }

    async fn store(&self, _content: &str, _metadata: Option<Value>) -> Result<String> {
        // Implementation to come in Phase B
        Ok("stub_id".to_string())
    }

    async fn forget(&self, _id: &str) -> Result<()> {
        // Implementation to come in Phase B
        Ok(())
    }
}
