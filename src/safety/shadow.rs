use std::path::PathBuf;
use std::fs;

pub struct ShadowWorkspace {
    pub root: PathBuf,
}

impl ShadowWorkspace {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}

impl Drop for ShadowWorkspace {
    fn drop(&mut self) {
        // Auto-cleanup logic
        if self.root.exists() {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}
