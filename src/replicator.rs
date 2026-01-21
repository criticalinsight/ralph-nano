//! Self-Replicating Agent Module
//!
//! Enables Ralph to clone itself into new workspaces, configure
//! dependencies, and start autonomous work without human intervention.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

const RALPH_DIR: &str = ".ralph";

/// Configuration for a new Ralph instance in a separate workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaConfig {
    /// Absolute path to the new workspace directory
    pub workspace: PathBuf,
    /// The primary task assigned to the new replica
    pub task: String,
    /// ID of the parent replica that spawned this instance
    pub parent_id: String,
    /// Whether to inherit the parent's memory graph
    pub inherit_memory: bool,
    /// Whether to inherit learned heuristics
    pub inherit_heuristics: bool,
    /// Maximum number of autonomous loops before stopping
    pub max_loops: usize,
    /// Whether to run in autonomous mode (self-correcting)
    pub autonomous: bool,
}

/// Status of a replica throughout its lifecycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicaStatus {
    /// Setting up the workspace and configuration
    Initializing,
    /// Installing dependencies and verifying environment
    ConfiguringDependencies,
    /// Active and executing the assigned task
    Running,
    /// Sucessfully completed the task
    Completed,
    /// Encountered a terminal error
    Failed(String),
}

/// A handle to a self-replicating agent instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Replica {
    /// Unique identifier for this replica
    pub id: String,
    /// The configuration used to create this replica
    pub config: ReplicaConfig,
    /// Current execution status
    pub status: ReplicaStatus,
    /// Process ID if the replica is currently running
    pub pid: Option<u32>,
    /// Path to the replica's execution log
    pub log_path: PathBuf,
}

/// The replication coordinator
#[allow(dead_code)]
pub struct Replicator {
    ralph_binary: PathBuf,
    global_memory_path: PathBuf,
}

impl Replicator {
    pub fn new() -> Result<Self> {
        // Find the ralph-nano binary
        let ralph_binary =
            std::env::current_exe().context("Failed to get current executable path")?;

        let global_memory_path = dirs::home_dir()
            .map(|h| h.join(".ralph/global_graph"))
            .unwrap_or_else(|| PathBuf::from(".ralph/global_graph"));

        Ok(Self {
            ralph_binary,
            global_memory_path,
        })
    }

    /// Clone Ralph into a new workspace
    #[allow(dead_code)]
    pub fn replicate(&self, config: ReplicaConfig) -> Result<Replica> {
        let id = uuid::Uuid::new_v4().to_string();
        let log_path = config.workspace.join(".ralph/replica.log");

        // Ensure workspace exists
        std::fs::create_dir_all(&config.workspace)?;
        std::fs::create_dir_all(config.workspace.join(".ralph"))?;

        let mut replica = Replica {
            id: id.clone(),
            config: config.clone(),
            status: ReplicaStatus::Initializing,
            pid: None,
            log_path: log_path.clone(),
        };

        // Initialize workspace structure
        self.init_workspace(&config)?;
        replica.status = ReplicaStatus::ConfiguringDependencies;

        // Copy memory if requested
        if config.inherit_memory {
            self.copy_global_memory(&config.workspace)?;
        }

        // Create TASKS.md with the assigned task
        self.create_tasks_file(&config)?;

        // Start the replica process
        let child = Command::new(&self.ralph_binary)
            .arg("--workspace")
            .arg(&config.workspace)
            .arg("--max-loops")
            .arg(config.max_loops.to_string())
            .arg("--autonomous")
            .current_dir(&config.workspace)
            .spawn()
            .context("Failed to spawn replica process")?;

        replica.pid = Some(child.id());
        replica.status = ReplicaStatus::Running;

        // Save replica state
        let state_path = config.workspace.join(".ralph/replica_state.json");
        std::fs::write(&state_path, serde_json::to_string_pretty(&replica)?)?;

        Ok(replica)
    }

    /// Initialize a workspace with Ralph structure
    #[allow(dead_code)]
    fn init_workspace(&self, config: &ReplicaConfig) -> Result<()> {
        let ws = &config.workspace;

        // Create .ralph directory
        std::fs::create_dir_all(ws.join(".ralph"))?;

        // Create Ralph.toml config
        let ralph_toml = format!(
            r#"# Ralph-Nano Configuration
max_loops = {}
autonomous = {}
parent_id = "{}"

[model]
primary = "gemini-3-pro-preview"
fallback = "gemini-2.5-flash"

[memory]
inherit_global = {}
"#,
            config.max_loops, config.autonomous, config.parent_id, config.inherit_memory
        );
        std::fs::write(ws.join("Ralph.toml"), ralph_toml)?;

        Ok(())
    }

    /// Copy global memory to new workspace
    #[allow(dead_code)]
    fn copy_global_memory(&self, workspace: &Path) -> Result<()> {
        let target = workspace.join(".ralph/memory");

        if self.global_memory_path.exists() {
            // Copy CozoDB files
            fs_extra::dir::copy(
                &self.global_memory_path,
                &target,
                &fs_extra::dir::CopyOptions::new(),
            )
            .ok(); // Ignore errors - memory is optional
        }

        Ok(())
    }

    /// Create initial TASKS.md file
    #[allow(dead_code)]
    fn create_tasks_file(&self, config: &ReplicaConfig) -> Result<()> {
        let tasks_content = format!(
            r#"# {}

## Objective
{}

## Tasks

- [ ] Analyze workspace and understand the codebase
- [ ] Plan implementation approach
- [ ] Execute the assigned task
- [ ] Verify changes and run tests
- [ ] Generate summary and report back

## Notes
- Parent replica: {}
- Autonomous mode: {}
- Max loops: {}
"#,
            config.task, config.task, config.parent_id, config.autonomous, config.max_loops
        );

        std::fs::write(config.workspace.join("TASKS.md"), tasks_content)?;
        Ok(())
    }

    /// Check the status of a replica
    #[allow(dead_code)]
    pub fn check_status(&self, workspace: &Path) -> Result<Replica> {
        let state_path = workspace.join(".ralph/replica_state.json");
        let content = std::fs::read_to_string(state_path)?;
        let replica: Replica = serde_json::from_str(&content)?;
        Ok(replica)
    }

    /// List all active replicas
    #[allow(dead_code)]
    pub fn list_replicas(&self) -> Result<Vec<Replica>> {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let replicas_dir = home.join(".ralph/replicas");

        let mut replicas = Vec::new();
        if replicas_dir.exists() {
            for entry in std::fs::read_dir(replicas_dir)? {
                let entry = entry?;
                let state_path = entry.path().join("replica_state.json");
                if state_path.exists() {
                    if let Ok(content) = std::fs::read_to_string(&state_path) {
                        if let Ok(replica) = serde_json::from_str(&content) {
                            replicas.push(replica);
                        }
                    }
                }
            }
        }

        Ok(replicas)
    }

    /// Spawn a replica for a specific subtask
    #[allow(dead_code)]
    pub fn spawn_subtask(&self, parent_workspace: &Path, subtask: &str) -> Result<Replica> {
        let parent_id = parent_workspace.to_string_lossy().to_string();
        let subtask_name = subtask.replace(' ', "_").to_lowercase();

        let workspace = parent_workspace
            .parent()
            .unwrap_or(parent_workspace)
            .join(format!(".ralph_subtask_{}", subtask_name));

        let config = ReplicaConfig {
            workspace,
            task: subtask.to_string(),
            parent_id,
            inherit_memory: true,
            inherit_heuristics: true,
            max_loops: 20,
            autonomous: true,
        };

        self.replicate(config)
    }

    /// Clone from a git repository
    #[allow(dead_code)]
    pub fn clone_and_replicate(&self, repo_url: &str, task: &str) -> Result<Replica> {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let workspace = home.join(".ralph/clones").join(
            repo_url
                .split('/')
                .next_back()
                .unwrap_or("repo")
                .replace(".git", ""),
        );

        // Clone the repository
        Command::new("git")
            .args(["clone", "--depth", "1", repo_url])
            .arg(&workspace)
            .output()
            .context("Failed to clone repository")?;

        let config = ReplicaConfig {
            workspace,
            task: task.to_string(),
            parent_id: "root".to_string(),
            inherit_memory: true,
            inherit_heuristics: true,
            max_loops: 50,
            autonomous: true,
        };

        self.replicate(config)
    }

    // --- PHASE 6: ROLLBACK SNAPSHOTS ---

    /// Create a rollback snapshot of the current workspace state
    pub fn create_snapshot(&self, label: &str) -> Result<String> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let snapshot_id = format!("{}_{}", timestamp, label.replace(' ', "_"));
        let snapshot_dir = Path::new(RALPH_DIR).join("snapshots").join(&snapshot_id);

        std::fs::create_dir_all(&snapshot_dir)?;

        // 1. Copy Critical Files
        let critical_files = vec![
            "Cargo.toml",
            "package.json",
            "TASKS.md",
            "README.md",
            "PRD.md",
        ];
        for file in critical_files {
            if Path::new(file).exists() {
                std::fs::copy(file, snapshot_dir.join(file))?;
            }
        }

        // 2. Snapshot Source Code (src/)
        if Path::new("src").exists() {
            let options = fs_extra::dir::CopyOptions::new()
                .overwrite(true)
                .content_only(false);
            fs_extra::dir::copy("src", &snapshot_dir, &options)?;
        }

        println!("📸 Snapshot created: {}", snapshot_id);
        Ok(snapshot_id)
    }

    /// Restore the workspace from a snapshot
    pub fn restore_snapshot(&self, snapshot_id: &str) -> Result<()> {
        let snapshot_dir = Path::new(RALPH_DIR).join("snapshots").join(snapshot_id);
        if !snapshot_dir.exists() {
            return Err(anyhow::anyhow!("Snapshot not found: {}", snapshot_id));
        }

        println!("⏪ Restoring snapshot: {}...", snapshot_id);

        // 1. Restore Files
        let _options = fs_extra::dir::CopyOptions::new()
            .overwrite(true)
            .content_only(true);
        for entry in std::fs::read_dir(&snapshot_dir)? {
            let entry = entry?;
            let path = entry.path();
            let file_name = match path.file_name() {
                Some(name) => name,
                None => continue,
            };

            if path.is_dir() {
                // recursively copy directories (like src/)
                if file_name == "src" {
                    // Remove current src to ensure clean state
                    if Path::new("src").exists() {
                        std::fs::remove_dir_all("src")?;
                    }
                    fs_extra::dir::copy(
                        &path,
                        ".",
                        &fs_extra::dir::CopyOptions::new().overwrite(true),
                    )?;
                }
            } else {
                std::fs::copy(&path, Path::new(".").join(file_name))?;
            }
        }

        println!("✅ Restoration complete.");
        Ok(())
    }
}

impl Default for Replicator {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            ralph_binary: PathBuf::from("ralph-nano"),
            global_memory_path: PathBuf::from(".ralph/global_graph"),
        })
    }
}
