//! Git-Aware Planning Module
//!
//! Parses git log, git blame, and repository history to understand
//! file authorship patterns, change frequency, and historical context
//! before modifying code.

use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};
use git2::{BlameOptions, Repository};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[allow(dead_code)]
/// Summary of a file's git history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileHistory {
    pub path: String,
    pub total_commits: usize,
    pub authors: Vec<AuthorStats>,
    pub recent_changes: Vec<CommitSummary>,
    pub change_frequency: ChangeFrequency,
    pub hotspots: Vec<LineRange>, // Frequently modified sections
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorStats {
    pub name: String,
    pub email: String,
    pub commit_count: usize,
    pub lines_contributed: usize,
    pub last_commit: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitSummary {
    pub hash: String,
    pub message: String,
    pub author: String,
    pub date: String,
    pub files_changed: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ChangeFrequency {
    Stable,   // < 1 commit/month
    Moderate, // 1-4 commits/month
    Active,   // 5-10 commits/month
    Volatile, // > 10 commits/month
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineRange {
    pub start: usize,
    pub end: usize,
    pub change_count: usize,
}

/// Blame information for a specific line
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlameInfo {
    pub line: usize,
    pub author: String,
    pub commit: String,
    pub date: String,
    pub original_line: usize,
}

#[allow(dead_code)]
/// Git-aware context provider
pub struct GitContext {
    repo: Repository,
}

impl GitContext {
    #[allow(dead_code)]
    /// Open a repository at the given path
    pub fn open(path: &Path) -> Result<Self> {
        let repo = Repository::discover(path).context("Failed to open git repository")?;
        Ok(Self { repo })
    }

    #[allow(dead_code)]
    /// Get the history summary for a specific file
    pub fn file_history(&self, file_path: &str, max_commits: usize) -> Result<FileHistory> {
        let mut revwalk = self.repo.revwalk()?;
        revwalk.push_head()?;
        revwalk.set_sorting(git2::Sort::TIME)?;

        let mut commits: Vec<CommitSummary> = Vec::new();
        let mut author_stats: HashMap<String, AuthorStats> = HashMap::new();
        let mut total_commits = 0;

        for oid in revwalk.take(1000) {
            let oid = oid?;
            let commit = self.repo.find_commit(oid)?;

            // Check if this commit touches our file
            if let Ok(parent) = commit.parent(0) {
                let diff = self.repo.diff_tree_to_tree(
                    Some(&parent.tree()?),
                    Some(&commit.tree()?),
                    None,
                )?;

                let touches_file = diff.deltas().any(|d| {
                    d.new_file()
                        .path()
                        .map(|p| p.to_string_lossy().contains(file_path))
                        .unwrap_or(false)
                });

                if touches_file {
                    total_commits += 1;
                    let author = commit.author();
                    let name = author.name().unwrap_or("Unknown").to_string();
                    let email = author.email().unwrap_or("").to_string();
                    let time = commit.time();
                    let date = Utc
                        .timestamp_opt(time.seconds(), 0)
                        .single()
                        .map(|dt| dt.to_rfc3339())
                        .unwrap_or_default();

                    // Update author stats
                    let entry = author_stats.entry(email.clone()).or_insert(AuthorStats {
                        name: name.clone(),
                        email: email.clone(),
                        commit_count: 0,
                        lines_contributed: 0,
                        last_commit: date.clone(),
                    });
                    entry.commit_count += 1;

                    // Collect recent commits
                    if commits.len() < max_commits {
                        commits.push(CommitSummary {
                            hash: oid.to_string().chars().take(8).collect(),
                            message: commit
                                .message()
                                .unwrap_or("")
                                .lines()
                                .next()
                                .unwrap_or("")
                                .to_string(),
                            author: name,
                            date,
                            files_changed: diff.deltas().count(),
                        });
                    }
                }
            }
        }

        // Calculate change frequency based on commit density
        let change_frequency = if total_commits < 6 {
            ChangeFrequency::Stable
        } else if total_commits < 24 {
            ChangeFrequency::Moderate
        } else if total_commits < 60 {
            ChangeFrequency::Active
        } else {
            ChangeFrequency::Volatile
        };

        let authors: Vec<AuthorStats> = author_stats.into_values().collect();

        Ok(FileHistory {
            path: file_path.to_string(),
            total_commits,
            authors,
            recent_changes: commits,
            change_frequency,
            hotspots: vec![], // Would require more complex analysis
        })
    }

    #[allow(dead_code)]
    /// Get blame information for a file
    pub fn blame_file(&self, file_path: &str) -> Result<Vec<BlameInfo>> {
        let path = Path::new(file_path);
        let blame = self
            .repo
            .blame_file(path, Some(BlameOptions::new().track_copies_same_file(true)))?;

        let mut blame_info = Vec::new();
        for (line_num, hunk) in blame.iter().enumerate() {
            let sig = hunk.final_signature();
            let commit_id = hunk.final_commit_id();

            blame_info.push(BlameInfo {
                line: line_num + 1,
                author: sig.name().unwrap_or("Unknown").to_string(),
                commit: commit_id.to_string().chars().take(8).collect(),
                date: Utc
                    .timestamp_opt(sig.when().seconds(), 0)
                    .single()
                    .map(|dt| dt.to_rfc3339())
                    .unwrap_or_default(),
                original_line: hunk.orig_start_line(),
            });
        }

        Ok(blame_info)
    }

    #[allow(dead_code)]
    /// Get recent commits across the entire repository
    pub fn recent_commits(&self, count: usize) -> Result<Vec<CommitSummary>> {
        let mut revwalk = self.repo.revwalk()?;
        revwalk.push_head()?;
        revwalk.set_sorting(git2::Sort::TIME)?;

        let mut commits = Vec::new();
        for oid in revwalk.take(count) {
            let oid = oid?;
            let commit = self.repo.find_commit(oid)?;
            let author = commit.author();
            let time = commit.time();

            commits.push(CommitSummary {
                hash: oid.to_string().chars().take(8).collect(),
                message: commit
                    .message()
                    .unwrap_or("")
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_string(),
                author: author.name().unwrap_or("Unknown").to_string(),
                date: Utc
                    .timestamp_opt(time.seconds(), 0)
                    .single()
                    .map(|dt| dt.to_rfc3339())
                    .unwrap_or_default(),
                files_changed: 0, // Would need diff to calculate
            });
        }

        Ok(commits)
    }

    #[allow(dead_code)]
    /// Get planning context for modifying a file
    pub fn planning_context(&self, file_path: &str) -> Result<String> {
        let history = self.file_history(file_path, 5)?;

        let mut context = format!("## Git Context for `{}`\n\n", file_path);

        // Change frequency assessment
        context.push_str(&format!(
            "**Stability:** {:?} ({} commits in history)\n\n",
            history.change_frequency, history.total_commits
        ));

        // Primary authors
        if !history.authors.is_empty() {
            context.push_str("**Primary Authors:**\n");
            for author in history.authors.iter().take(3) {
                context.push_str(&format!(
                    "- {} ({} commits)\n",
                    author.name, author.commit_count
                ));
            }
            context.push('\n');
        }

        // Recent changes
        if !history.recent_changes.is_empty() {
            context.push_str("**Recent Changes:**\n");
            for commit in &history.recent_changes {
                context.push_str(&format!(
                    "- `{}` {} ({})\n",
                    commit.hash, commit.message, commit.author
                ));
            }
            context.push('\n');
        }

        // Planning advice based on history
        context.push_str("**Planning Advice:**\n");
        match history.change_frequency {
            ChangeFrequency::Stable => {
                context.push_str("- This file is stable. Changes should be well-justified.\n");
                context.push_str("- Consider whether this is the right place for modifications.\n");
            }
            ChangeFrequency::Moderate => {
                context.push_str("- This file sees occasional updates. Standard review process.\n");
            }
            ChangeFrequency::Active => {
                context.push_str(
                    "- This file is actively developed. Coordinate with recent authors.\n",
                );
                context.push_str("- Consider potential merge conflicts.\n");
            }
            ChangeFrequency::Volatile => {
                context.push_str("- ⚠️ This file is a hotspot with frequent changes.\n");
                context.push_str("- High risk of merge conflicts. Keep changes minimal.\n");
                context.push_str("- Consider refactoring to reduce churn.\n");
            }
        }

        Ok(context)
    }
}
