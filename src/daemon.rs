//! Persistent Daemon Module
//!
//! Runs Ralph as a background service, watching for file changes
//! and automatically triggering actions like generating PRs or filing issues.

use anyhow::Result;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;
use tokio::sync::broadcast;

/// Events that the daemon can respond to
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonEvent {
    FileCreated(PathBuf),
    FileModified(PathBuf),
    FileDeleted(PathBuf),
    DirectoryCreated(PathBuf),
    TaskCompleted(String),
    ErrorOccurred(String),
}

/// Actions the daemon can take in response to events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonAction {
    RunCheck,
    RunTests,
    GeneratePR { title: String, branch: String },
    FileIssue { title: String, body: String },
    NotifyUser(String),
    TriggerBuild,
    SyncGraph,
}

/// Configuration for the daemon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    pub watch_paths: Vec<PathBuf>,
    pub ignore_patterns: Vec<String>,
    pub debounce_ms: u64,
    pub auto_check: bool,
    pub auto_test: bool,
    pub auto_pr: bool,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            watch_paths: vec![PathBuf::from("src"), PathBuf::from("tests")],
            ignore_patterns: vec![
                "target/*".to_string(),
                ".git/*".to_string(),
                "*.lock".to_string(),
                ".ralph/*".to_string(),
            ],
            debounce_ms: 500,
            auto_check: true,
            auto_test: false,
            auto_pr: false,
        }
    }
}

#[allow(dead_code)]
/// File watcher for the daemon
pub struct FileWatcher {
    _watcher: RecommendedWatcher,
    receiver: Receiver<notify::Result<Event>>,
}

impl FileWatcher {
    /// Create a new file watcher for the given paths
    pub fn new(paths: &[PathBuf]) -> Result<Self> {
        let (tx, rx) = channel();

        let mut watcher = notify::recommended_watcher(move |res| {
            let _ = tx.send(res);
        })?;

        for path in paths {
            if path.exists() {
                watcher.watch(path, RecursiveMode::Recursive)?;
            }
        }

        Ok(Self {
            _watcher: watcher,
            receiver: rx,
        })
    }

    /// Get the next event (blocking)
    pub fn next_event(&self) -> Option<DaemonEvent> {
        match self.receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(Ok(event)) => self.convert_event(event),
            _ => None,
        }
    }

    #[allow(dead_code)]
    fn convert_event(&self, event: Event) -> Option<DaemonEvent> {
        let path = event.paths.first()?.clone();

        match event.kind {
            EventKind::Create(_) => {
                if path.is_dir() {
                    Some(DaemonEvent::DirectoryCreated(path))
                } else {
                    Some(DaemonEvent::FileCreated(path))
                }
            }
            EventKind::Modify(_) => Some(DaemonEvent::FileModified(path)),
            EventKind::Remove(_) => Some(DaemonEvent::FileDeleted(path)),
            _ => None,
        }
    }
}

/// The main daemon process
pub struct Daemon {
    config: DaemonConfig,
    watcher: Option<FileWatcher>,
    event_tx: broadcast::Sender<DaemonEvent>,
    action_tx: broadcast::Sender<DaemonAction>,
}

impl Daemon {
    #[allow(dead_code)]
    pub fn new(config: DaemonConfig) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        let (action_tx, _) = broadcast::channel(100);

        Self {
            config,
            watcher: None,
            event_tx,
            action_tx,
        }
    }

    #[allow(dead_code)]
    /// Start the daemon
    pub fn start(&mut self) -> Result<()> {
        self.watcher = Some(FileWatcher::new(&self.config.watch_paths)?);
        Ok(())
    }

    #[allow(dead_code)]
    /// Subscribe to daemon events
    pub fn subscribe_events(&self) -> broadcast::Receiver<DaemonEvent> {
        self.event_tx.subscribe()
    }

    #[allow(dead_code)]
    /// Subscribe to daemon actions
    pub fn subscribe_actions(&self) -> broadcast::Receiver<DaemonAction> {
        self.action_tx.subscribe()
    }

    /// Check if a path should be ignored
    fn should_ignore(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        self.config.ignore_patterns.iter().any(|pattern| {
            // Simple glob matching
            if pattern.ends_with("/*") {
                let prefix = &pattern[..pattern.len() - 2];
                path_str.contains(prefix)
            } else if let Some(ext) = pattern.strip_prefix("*.") {
                path_str.ends_with(ext)
            } else {
                path_str.contains(pattern)
            }
        })
    }

    /// Process a file event and determine appropriate actions
    pub fn process_event(&self, event: &DaemonEvent) -> Vec<DaemonAction> {
        let mut actions = Vec::new();

        match event {
            DaemonEvent::FileModified(path) | DaemonEvent::FileCreated(path) => {
                if self.should_ignore(path) {
                    return actions;
                }

                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

                // Source file changes
                if matches!(ext, "rs" | "ts" | "js" | "py" | "go") {
                    if self.config.auto_check {
                        actions.push(DaemonAction::RunCheck);
                    }
                    actions.push(DaemonAction::SyncGraph);
                }

                // Test file changes
                if path.to_string_lossy().contains("test") && self.config.auto_test {
                    actions.push(DaemonAction::RunTests);
                }
            }
            DaemonEvent::TaskCompleted(task) => {
                if self.config.auto_pr {
                    actions.push(DaemonAction::GeneratePR {
                        title: format!("Automated: {}", task),
                        branch: format!("ralph/{}", task.to_lowercase().replace(' ', "-")),
                    });
                }
                actions.push(DaemonAction::NotifyUser(format!(
                    "Task completed: {}",
                    task
                )));
            }
            DaemonEvent::ErrorOccurred(err) => {
                actions.push(DaemonAction::NotifyUser(format!("Error: {}", err)));
            }
            _ => {}
        }

        actions
    }

    #[allow(dead_code)]
    /// Run the daemon loop (blocking)
    pub async fn run(&mut self) -> Result<()> {
        self.start()?;

        loop {
            if let Some(ref watcher) = self.watcher {
                if let Some(event) = watcher.next_event() {
                    let _ = self.event_tx.send(event.clone());

                    let actions = self.process_event(&event);
                    for action in actions {
                        let _ = self.action_tx.send(action);
                    }
                }
            }

            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

#[allow(dead_code)]
/// LaunchAgent plist generator for macOS
pub fn generate_launchd_plist(binary_path: &str, workspace: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ralph.nano.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>{}</string>
        <string>--daemon</string>
        <string>--workspace</string>
        <string>{}</string>
    </array>
    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/ralph-nano.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/ralph-nano.error.log</string>
    <key>WorkingDirectory</key>
    <string>{}</string>
</dict>
</plist>"#,
        binary_path, workspace, workspace
    )
}
