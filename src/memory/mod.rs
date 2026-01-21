use async_trait::async_trait;
use anyhow::Result;
use serde_json::Value;

pub mod cozo;
pub mod adapter;

#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn recall(&self, query: &str, limit: usize) -> Result<Vec<String>>;
    async fn store(&self, content: &str, metadata: Option<Value>) -> Result<String>;
    async fn forget(&self, id: &str) -> Result<()>;
}
