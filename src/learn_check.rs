use ralph_nano::memory_legacy::Memory;
mod knowledge;
use std::sync::Arc;
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let memory = Arc::new(Memory::new(".ralph/lancedb").await?);
    let engine = knowledge::KnowledgeEngine::new(memory);
    
    println!("🔍 Scanning workspace for new ideas...");
    let libs = engine.scan_all_dependencies()?;
    
    if libs.is_empty() {
        println!("📭 No new libraries detected.");
    } else {
        println!("📚 Detected {} libraries:", libs.len());
        for lib in libs {
            println!("   - {} ({}) [{:?}]", lib.name, lib.version, lib.lib_type);
        }
    }
    
    Ok(())
}
