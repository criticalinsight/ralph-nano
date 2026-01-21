use anyhow::{anyhow, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// OverlayFS provides a safe, transactional layer over the filesystem.
/// 
/// - Reads: Check overlay first, then base.
/// - Writes: Always write to overlay.
/// - Commit: Copy overlay contents to base atomically (as much as possible).
/// - Rollback: Discard overlay.
pub struct OverlayFS {
    base_dir: PathBuf,
    overlay_dir: PathBuf,
}

impl OverlayFS {
    /// Creates a new OverlayFS. 
    /// `base_dir`: The real workspace (e.g., user's project).
    /// `overlay_id`: Unique ID for this transaction (e.g., task ID).
    pub fn new(base_dir: &Path, overlay_id: &str) -> Result<Self> {
        let temp_dir = std::env::temp_dir().join("ralph_overlays").join(overlay_id);
        
        if temp_dir.exists() {
            fs::remove_dir_all(&temp_dir)?;
        }
        fs::create_dir_all(&temp_dir)?;

        Ok(Self {
            base_dir: base_dir.to_path_buf(),
            overlay_dir: temp_dir,
        })
    }

    /// Reads a file, transparently checking overlay then base.
    pub fn read_file(&self, path: &Path) -> Result<String> {
        let rel_path = self.get_relative_path(path)?;
        let overlay_path = self.overlay_dir.join(&rel_path);

        if overlay_path.exists() {
            return Ok(fs::read_to_string(overlay_path)?);
        }

        let base_path = self.base_dir.join(&rel_path);
        if base_path.exists() {
            return Ok(fs::read_to_string(base_path)?);
        }

        Err(anyhow::anyhow!("File not found in overlay or base: {:?}", path))
    }

    /// Writes a file to the overlay.
    pub fn write_file(&self, path: &Path, content: &str) -> Result<()> {
        let rel_path = self.get_relative_path(path)?;
        let overlay_path = self.overlay_dir.join(&rel_path);

        if let Some(parent) = overlay_path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(overlay_path, content)?;
        Ok(())
    }

    /// Commits changes from overlay to base.
    /// This effectively "merges" the overlay into the base.
    pub fn commit(&self) -> Result<()> {
        // Recursively copy overlay_dir to base_dir
        self.copy_dir_recursive(&self.overlay_dir, &self.base_dir)?;
        
        // Cleanup overlay after successful commit ?? 
        // Or keep it until explicit cleanup? 
        // Standard transaction: commit persists, then we are done.
        
        Ok(())
    }

    /// Discards the overlay (rollback).
    pub fn rollback(&self) -> Result<()> {
        if self.overlay_dir.exists() {
            fs::remove_dir_all(&self.overlay_dir)?;
        }
        Ok(())
    }

    /// Helper to handle absolute/relative paths and ensure they are within workspace
    fn get_relative_path(&self, path: &Path) -> Result<PathBuf> {
        if path.is_absolute() {
            if path.starts_with(&self.base_dir) {
                Ok(path.strip_prefix(&self.base_dir)?.to_path_buf())
            } else {
                // If it's absolute but NOT in base dir, we might reject it or handle it.
                // For safety, we only allow operations within base_dir.
                Err(anyhow!("Path {:?} is outside base directory {:?}", path, self.base_dir))
            }
        } else {
            Ok(path.to_path_buf())
        }
    }

    fn copy_dir_recursive(&self, src: &Path, dst: &Path) -> Result<()> {
        if !dst.exists() {
            fs::create_dir_all(dst)?;
        }

        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let ft = entry.file_type()?;
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());

            if ft.is_dir() {
                self.copy_dir_recursive(&src_path, &dst_path)?;
            } else {
                fs::copy(&src_path, &dst_path)?;
            }
        }
        Ok(())
    }
}

// Ensure cleanup on drop if not committed? 
// Ideally yes, but strict transactional logic (commit consumed) is safer.
// For now, let's leave drop explicit or rely on OS temp cleanup, 
// to avoid accidental data loss if the struct is dropped prematurely.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlay_transaction() -> Result<()> {
        let temp_root = std::env::temp_dir().join("ralph_test_safety");
        if temp_root.exists() {
            fs::remove_dir_all(&temp_root)?;
        }
        fs::create_dir_all(&temp_root)?;

        let base_file = temp_root.join("config.toml");
        fs::write(&base_file, "version = 1")?;

        let overlay = OverlayFS::new(&temp_root, "tx_1")?;

        // 1. Read base through overlay
        assert_eq!(overlay.read_file(&base_file)?, "version = 1");

        // 2. Write to overlay (shadowed)
        overlay.write_file(&base_file, "version = 2")?;
        
        // 3. Read should show new version
        assert_eq!(overlay.read_file(&base_file)?, "version = 2");

        // 4. Base should still be old
        assert_eq!(fs::read_to_string(&base_file)?, "version = 1");

        // 5. Commit
        overlay.commit()?;

        // 6. Base should now be updated
        assert_eq!(fs::read_to_string(&base_file)?, "version = 2");

        Ok(())
    }
}
