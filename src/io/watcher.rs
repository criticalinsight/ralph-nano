use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use tokio::sync::mpsc::Sender;
use crate::io::events::Impulse;
use anyhow::Result;

pub fn setup_watcher(path: &Path, tx: Sender<Impulse>) -> Result<RecommendedWatcher> {
    let tx_clone = tx.clone();
    
    let mut watcher = notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
        match res {
            Ok(event) => {
                // Filter out noise if needed (e.g. .git, target)
                let _ = tx_clone.blocking_send(Impulse::FileSystemEvent(event));
            }
            Err(e) => {
                let _ = tx_clone.blocking_send(Impulse::Error(format!("Watch error: {:?}", e)));
            }
        }
    })?;

    watcher.watch(path, RecursiveMode::Recursive)?;
    
    Ok(watcher)
}
