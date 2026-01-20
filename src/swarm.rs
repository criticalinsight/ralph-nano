// src/swarm.rs - Worker Swarm Orchestration for Ralph-Nano v1.0.0

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::{Child, Command, Stdio};

const RALPH_DIR: &str = ".ralph";

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerStatus {
    pub id: String,
    pub state: WorkerState,
    pub current_task: Option<String>,
    pub assigned_task: Option<String>,
    pub completed_tasks: usize,
    #[allow(dead_code)]
    pub subtasks_completed: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub enum WorkerState {
    Idle,
    Working,
    Errored,
    Stopped,
    Syncing,
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
            assigned_task: None,
            completed_tasks: 0,
            subtasks_completed: Vec::new(),
        };
        fs::write(
            worker_dir.join("status.json"),
            serde_json::to_string_pretty(&status)?,
        )?;

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

    #[allow(dead_code)]
    pub fn assign_task(&self, task: &str) -> Result<()> {
        let worker_dir = Path::new(RALPH_DIR).join("swarm").join(&self.id);
        let task_file = worker_dir.join("task.json");
        fs::write(
            task_file,
            serde_json::to_string(&json!({ "objective": task }))?,
        )?;
        Ok(())
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn delegate_tasks(&mut self, tasks: Vec<String>) -> Result<()> {
        let statuses = self.poll_workers();
        let mut idle_workers: Vec<String> = statuses
            .into_iter()
            .filter(|s| s.state == WorkerState::Idle)
            .map(|s| s.id)
            .collect();

        for task in tasks {
            if let Some(worker_id) = idle_workers.pop() {
                if let Some(worker) = self.workers.get(&worker_id) {
                    println!("🐝 Delegating task to {}: {}", worker_id, task);
                    worker.assign_task(&task)?;
                }
            } else {
                break; // No more idle workers
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn active_count(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for SwarmManager {
    fn drop(&mut self) {
        let _ = self.shutdown_all();
    }
}
