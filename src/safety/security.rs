use anyhow::{Result, anyhow};
use std::process::Command;
use regex::Regex;

pub struct Security;

impl Security {
    /// Scans the git staged changes for potential API key leaks.
    pub fn scan_git_diff() -> Result<()> {
        let output = Command::new("git").args(["diff", "--cached"]).output()?;
        let diff = String::from_utf8_lossy(&output.stdout);
        
        // Google Gemini API Key pattern
        let google_re = Regex::new(r"AIzaSy[A-Za-z0-9-_]{33}")?;
        
        if google_re.is_match(&diff) {
            return Err(anyhow!(
                "🚨 SECURITY ALERT: Google API Key detected in git staging! Operation aborted."
            ));
        }
        Ok(())
    }
}
