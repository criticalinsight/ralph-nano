use ralph_nano::memory_legacy::Memory;
use ralph_nano::core::state::{GlobalState, RalphConfig};
use ralph_nano::core::r#loop::cortex_loop;
use ralph_nano::io::watcher::setup_watcher;
use ralph_nano::memory::adapter::LegacyMemoryAdapter;
use ralph_nano::safety::OverlayFS;
use ralph_nano::knowledge::KnowledgeEngine;
use ralph_nano::core::cortex::{Cortex, ThinkingLevel};

use tokio::sync::mpsc;
use anyhow::{Context, Result};
use colored::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::fs;
use std::path::{Path};
// use std::process::Command;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

pub const RALPH_DIR: &str = ".ralph";

// mod benchmarker;
// mod daemon;
// mod fingerprint;
// mod git_context;
// mod janitor;
// mod mcp_client;
// mod qa;
// mod reflexion;
// mod replicator;
// mod swarm;
// mod wasm_runtime;
// mod api_server;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    WriteFile { path: String, content: String },
    ExecShell { command: String, context: String },
    QueryMemory { query: String },
    Commit,
}

#[tokio::main]
async fn main() -> Result<()> {
    if env::args().any(|a| a == "init") {
        return init_workspace();
    }

    if env::var("RALPH_WORKER_ID").is_ok() {
        return run_worker_loop().await;
    }

    // 1. Initialize State
    let config = RalphConfig::default(); 
    
    // Core Memory (Shared)
    let memory = Arc::new(Memory::new(&format!("{}/lancedb", RALPH_DIR)).await.context("Failed to init memory")?);
    
    // Legacy Memory Adapter (Ownership of Arc<Memory>)
    let memory_adapter = Arc::new(LegacyMemoryAdapter::new(memory.clone()));
    
    // Cortex (Ownership of Arc<Memory>)
    let cortex = Arc::new(Cortex::new(config.clone(), memory.clone())?);

    // 1.1 Autodidact Phase (The Learning)
    let engine = KnowledgeEngine::new(memory.clone());
    println!("{} Scanning Local Environment for New Knowledge...", "🧠".cyan());
    if let Ok(libs) = engine.scan_all_dependencies() {
        if libs.is_empty() {
             println!("   {} No external libraries detected.", "📭".yellow());
        } else {
             println!("   {} Detected {} Libraries:", "📚".green(), libs.len());
             for lib in &libs {
                 println!("     - {} ({})", lib.name.bold(), lib.version);
             }
             
             // Sync libraries to memory to enable update checks
             if let Err(e) = engine.sync_libraries(&libs).await {
                 eprintln!("Failed to sync library metadata: {}", e);
             }
             
             // Check for library updates
             println!("   {} Checking for upstream updates...", "🌐".blue());
             match engine.check_library_updates().await {
                 Ok(updates) => {
                     if updates.is_empty() {
                         println!("     {} All libraries are current.", "✅".green());
                     } else {
                         let mut found_updates = false;
                         for (name, local, latest) in &updates {
                             if local != latest {
                                 println!("     - {} [{} -> {}]", name.bold().yellow(), local, latest);
                                 found_updates = true;
                             }
                         }
                         
                         if found_updates {
                             println!("   {} Upgrading manifests...", "📦".cyan());
                             if let Err(e) = engine.upgrade_manifest(&updates).await {
                                 eprintln!("Upgrade failed: {}", e);
                             } else {
                                 println!("   {} Upgrade complete. Please rebuild to apply changes.", "✨".green());
                             }
                         }
                     }
                 }
                 Err(e) => {
                     eprintln!("   {} Library update check failed: {}", "⚠️".red(), e);
                 }
             }
        }
    }

    // Safety Shield
    let overlay = Arc::new(OverlayFS::new(&std::env::current_dir()?, "godmode_session")?);
    println!("{} Safety Shield (OverlayFS) Active", "🛡️".green());

    let state = Arc::new(GlobalState::new(config, memory_adapter, memory.clone(), overlay, cortex));

    // 2. Setup Event Bus (Nervous System QoS)
    let (_priority_tx, priority_rx) = mpsc::channel(100);
    let (background_tx, background_rx) = mpsc::channel(1000);

    // 3. Setup File Watcher
    // Route file events to BACKGROUND channel
    let _watcher = setup_watcher(Path::new("."), background_tx.clone())?;
    
    // 4. Start Cortex Loop (Godmode)
    println!("{}", "🚀 Godmode Activated: Event Bus Online".green().bold());
    println!("{}", "   - Priority Channel: READY".yellow());
    println!("{}", "   - Background Channel: READY".blue());
    
    cortex_loop(priority_rx, background_rx, state).await;

    Ok(())
}

fn init_workspace() -> Result<()> {
    let ralph_path = Path::new(RALPH_DIR);
    if ralph_path.exists() {
        println!("{}", "✅ Ralph is already alive in this workspace.".green());
        return Ok(());
    }
    fs::create_dir_all(ralph_path.join("lancedb"))?;
    fs::create_dir_all(ralph_path.join("shadow"))?;
    let config = RalphConfig::default();
    let toml = toml::to_string_pretty(&config)?;
    fs::write(ralph_path.join("config.toml"), toml)?;
    
    let gitignore_path = Path::new(".gitignore");
    let mut gitignore = if gitignore_path.exists() {
        fs::read_to_string(gitignore_path)?
    } else {
        String::new()
    };
    if !gitignore.contains(".ralph") {
        gitignore.push_str("\n# Ralph Agent Data\n.ralph/\n");
        fs::write(".gitignore", gitignore)?;
    }
    println!("{}", "🧬 DNA REPLICATION COMPLETE.".green().bold());
    Ok(())
}

async fn run_worker_loop() -> Result<()> {
    let worker_id = env::var("RALPH_WORKER_ID").unwrap_or_else(|_| "worker-0".to_string());
    println!("🐝 Worker {} starting...", worker_id);
    dotenvy::from_filename(".env").ok();
    
    let config_path = format!("{}/config.toml", RALPH_DIR);
    let config_str = fs::read_to_string(&config_path).unwrap_or_default();
    let config: RalphConfig = toml::from_str(&config_str).unwrap_or_default();

    let memory = Arc::new(Memory::new(&format!("{}/lancedb", RALPH_DIR)).await?);
    let cortex = Cortex::new(config, Arc::clone(&memory))?;

    let task_path = format!("{}/swarm/{}/task.json", RALPH_DIR, worker_id);
    
    loop {
        if Path::new(&task_path).exists() {
            if let Ok(content) = fs::read_to_string(&task_path) {
                if let Ok(task_val) = serde_json::from_str::<Value>(&content) {
                    let task = task_val["objective"].as_str().unwrap_or_default().to_string();
                    if !task.is_empty() {
                        println!("🐝 Worker {} executing: {}", worker_id, task);
                        let prompt = format!("Execute this task autonomously:\n{}", task);
                        match cortex.generate(&prompt, ThinkingLevel::High).await {
                             Ok(_response) => {
                                 println!("✅ Worker {} COMPLETED task.", worker_id);
                             }
                             Err(e) => {
                                 eprintln!("Worker task failed: {}", e);
                             }
                        }
                        let _ = fs::remove_file(&task_path);
                    }
                }
            }
        }
        sleep(Duration::from_secs(2)).await;
    }
}
