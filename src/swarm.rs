// src/swarm.rs - Worker Swarm Orchestration for Ralph-Nano v1.0.0

use anyhow::{Context, Result};
use std::process::{Child, Command, Stdio};
use std::path::Path;
use std::fs;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

const RALPH_DIR: &str = ".ralph";

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerStatus {
    pub id: String,
    pub state: WorkerState,
    pub current_task: Option<String>,
    pub completed_tasks: usize,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum WorkerState {
    Idle,
    Working,
    Errored,
    Stopped,
}

pub struct SwarmWorker {
    pub id: String,
    process: Child,
}

impl SwarmWorker {
    fn spawn(id: &str, workspace: &Path) -> Result<Self> {
        let worker_dir = Path::new(RALPH_DIR).join("swarm").join(id);
        fs::create_dir_all(&worker_dir)?;
        
        // Initialize status file
        let status = WorkerStatus {
            id: id.to_string(),
            state: WorkerState::Idle,
            current_task: None,
            completed_tasks: 0,
        };
        fs::write(worker_dir.join("status.json"), serde_json::to_string_pretty(&status)?)?;
        
        // Spawn the worker process
        let child = Command::new("cargo")
            .args(["run", "--", "--worker-mode"])
            .current_dir(workspace)
            .env("RALPH_WORKER_ID", id)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context(format!("Failed to spawn worker {}", id))?;
        
        Ok(Self {
            id: id.to_string(),
            process: child,
        })
    }
    
    fn is_running(&mut self) -> bool {
        match self.process.try_wait() {
            Ok(Some(_)) => false,
            Ok(None) => true,
            Err(_) => false,
        }
    }
    
    fn kill(&mut self) -> Result<()> {
        self.process.kill().context("Failed to kill worker")?;
        Ok(())
    }
}

pub struct SwarmManager {
    workers: HashMap<String, SwarmWorker>,
    workspace: std::path::PathBuf,
}

impl SwarmManager {
    pub fn new(workspace: &Path) -> Self {
        Self {
            workers: HashMap::new(),
            workspace: workspace.to_path_buf(),
        }
    }
    
    pub fn spawn_workers(&mut self, count: usize) -> Result<()> {
        for i in 0..count {
            let id = format!("worker-{}", i);
            if self.workers.contains_key(&id) {
                continue;
            }
            
            println!("🐝 Spawning worker: {}", id);
            let worker = SwarmWorker::spawn(&id, &self.workspace)?;
            self.workers.insert(id, worker);
        }
        Ok(())
    }
    
    pub fn poll_workers(&mut self) -> Vec<WorkerStatus> {
        let swarm_dir = Path::new(RALPH_DIR).join("swarm");
        let mut statuses = Vec::new();
        
        for (id, worker) in &mut self.workers {
            let status_path = swarm_dir.join(id).join("status.json");
            if status_path.exists() {
                if let Ok(content) = fs::read_to_string(&status_path) {
                    if let Ok(status) = serde_json::from_str::<WorkerStatus>(&content) {
                        statuses.push(status);
                    }
                }
            }
            
            // Check if process is still alive
            if !worker.is_running() {
                println!("⚠️ Worker {} has stopped.", id);
            }
        }
        
        statuses
    }
    
    pub fn shutdown_all(&mut self) -> Result<()> {
        println!("🛑 Shutting down all workers...");
        for (id, worker) in &mut self.workers {
            println!("   Stopping: {}", id);
            let _ = worker.kill();
        }
        self.workers.clear();
        Ok(())
    }
    
    pub fn active_count(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for SwarmManager {
    fn drop(&mut self) {
        let _ = self.shutdown_all();
    }
}
