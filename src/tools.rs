use crate::RALPH_DIR;
// use crate::mcp_client::McpManager;
use crate::memory::{GraphNode, Memory};
use anyhow::{Result};
use chrono::Utc;
use colored::*;
// use fs_extra::dir::CopyOptions;
use rayon::prelude::*;
use regex::Regex;
// use similar::{ChangeTag, TextDiff};
use std::collections::HashMap;
// use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path};
use std::process::Command;
use syn::{
    visit::{self, Visit},
    ItemEnum, ItemFn, ItemImpl, ItemStruct,
};
// use tokio::io::{AsyncReadExt, AsyncWriteExt};
use walkdir::WalkDir;
use quote::ToTokens;

const INCLUDE_EXTENSIONS: &[&str] = &["rs", "py", "js", "ts", "html", "css", "toml", "md"];
const SKIP_DIRS: &[&str] = &[
    "target",
    ".git",
    ".ralph",
    "node_modules",
    "lancedb",
    "shadow",
];

// Re-export ContextManager if it's needed here, or assume it's passed or defined here?
// Ideally ContextManager should also be in a shared place or here, but it's small.
// For now, I'll redefine a simple interface trait if needed or assume it's public in main.
// Actually, to make this work without circular deps or public reshuffling, 
// I will move ContextManager here too if it's tightly coupled, or just accept `&impl ContextManagerTrait`.
// But simpler is to assume ContextManager is moving or I can see its definition.
// Wait, ContextManager is in main.rs lines 92-113. It's small. I'll move it here too for cohesion.

pub struct ContextManager {
    pub hot_files: HashMap<String, std::time::SystemTime>,
}

impl ContextManager {
    pub fn new() -> Self {
        Self {
            hot_files: HashMap::new(),
        }
    }

    pub fn mark_hot(&mut self, path: &str) {
        self.hot_files
            .insert(path.to_string(), std::time::SystemTime::now());
    }

    pub fn is_hot(&self, path: &str) -> bool {
        if let Some(time) = self.hot_files.get(path) {
            if let Ok(elapsed) = time.elapsed() {
                return elapsed.as_secs() < 300; // 5 mins hot window
            }
        }
        false
    }
}

pub struct Tools;

pub struct FileSymbols {
    pub symbols: Vec<String>,
    pub dependencies: Vec<String>,
}

struct SemanticVisitor {
    symbols: Vec<String>,
    dependencies: Vec<String>,
}

impl<'ast> Visit<'ast> for SemanticVisitor {
    fn visit_item_fn(&mut self, i: &'ast ItemFn) {
        self.symbols.push(format!("fn {}", i.sig.ident));
        visit::visit_item_fn(self, i);
    }

    fn visit_item_struct(&mut self, i: &'ast ItemStruct) {
        self.symbols.push(format!("struct {}", i.ident));
        visit::visit_item_struct(self, i);
    }

    fn visit_item_enum(&mut self, i: &'ast ItemEnum) {
        self.symbols.push(format!("enum {}", i.ident));
        visit::visit_item_enum(self, i);
    }

    fn visit_item_impl(&mut self, i: &'ast ItemImpl) {
        let ty = i.self_ty.to_token_stream().to_string();
        if let Some((_, trait_path, _)) = &i.trait_ {
            let tr = trait_path.to_token_stream().to_string();
            self.symbols.push(format!("impl {} for {}", tr, ty));
        } else if let Some((_, path)) = ty.split_once(' ') {
            self.symbols.push(format!("impl {}", path));
        } else {
            self.symbols.push(format!("impl {}", ty));
        }
        visit::visit_item_impl(self, i);
    }
    // Simple dependency tracking: look for type identifiers
    fn visit_type_path(&mut self, i: &'ast syn::TypePath) {
        if let Some(segment) = i.path.segments.last() {
            self.dependencies.push(segment.ident.to_string());
        }
        visit::visit_type_path(self, i);
    }
}

impl Tools {
    /// Scans the workspace and builds a context string based on "hot" (recently used) files.
    /// 
    /// # Complexity Analysis
    /// * **Time Complexity**: O(N) where N is the total number of files in the workspace.
    ///   It iterates through all files once. For each file, it performs constant time checks (hot check, extension check).
    ///   Reading file content or extracting symbols depends on file size (M), so effectively O(N * M).
    /// * **Space Complexity**: O(S) where S is the size of the generated context string.
    ///   The string grows linearly with the content of hot files and symbol lists of cold files.
    pub fn scan_workspace(root: &Path, manager: &ContextManager) -> String {
        let mut context = String::new();
        for entry in WalkDir::new(root)
            .into_iter()
            .filter_entry(|e| !is_hidden(e))
            .flatten()
        {
            if entry.file_type().is_file() {
                    let path = entry.path();
                    let rel_path = path.to_string_lossy().into_owned();

                    if manager.is_hot(&rel_path)
                        || rel_path.ends_with("main.rs")
                        || rel_path.ends_with("TASKS.md")
                    {
                        if let Ok(content) = fs::read_to_string(path) {
                            context.push_str(&format!(
                                "\n--- FILE (HOT): {:?} ---\n{}\n",
                                path, content
                            ));
                        }
                    } else {
                        // Cold files: symbols only
                        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                        if INCLUDE_EXTENSIONS.contains(&ext) {
                            let file_data = Self::extract_symbols_from_file(path, ext);
                            context
                                .push_str(&format!("\n--- FILE (COLD/SYMBOLS): {:?} ---\n", path));
                            for sym in file_data.symbols {
                                context.push_str(&format!("  SYMBOL: {}\n", sym));
                            }
                        }
                    }
                }
            }
        context
    }

    /// Scans the workspace to build a map of symbols (functions, structs, etc.) for all supported files.
    /// Uses caching to avoid re-parsing unchanged files.
    /// 
    /// # Complexity Analysis
    /// * **Time Complexity**: O(N) parallelized. Processing each file takes O(M) where M is file size.
    ///   Cache lookup is O(1) (hash map). Walking directory is O(N).
    /// * **Space Complexity**: O(N) for storing the symbol cache and results vector.
    pub fn scan_symbols(root: &Path) -> String {
        let cache_path = Path::new(RALPH_DIR).join("symbol_cache.json");
        let cache: HashMap<String, (u64, Vec<String>)> = if cache_path.exists() {
            fs::read_to_string(&cache_path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default()
        } else {
            HashMap::new()
        };

        let mut map = String::new();
        map.push_str("WORKSPACE SYMBOL MAP (HIERARCHY & SIGNATURES):\n");

        // Collect all valid entries first (WalkDir is sync)
        let entries: Vec<_> = WalkDir::new(root)
            .into_iter()
            .filter_entry(|e| !is_hidden(e))
            .filter_map(|e| e.ok())
            .collect();

        type SymbolResult = (String, Option<String>, Option<(u64, Vec<String>)>);

        // Process entries in parallel where possible
        let processed_results: Vec<SymbolResult> = entries
            .par_iter()
            .map(|entry| {
                let path = entry.path();
                let rel_path = path.to_string_lossy().into_owned();

                if entry.file_type().is_dir() {
                    return (rel_path, Some(format!("DIR: {:?}\n", path)), None);
                } else if entry.file_type().is_file() {
                    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                    if !INCLUDE_EXTENSIONS.contains(&ext) {
                        return (rel_path, None, None);
                    }

                    let meta = fs::metadata(path).ok();
                    let mtime = meta
                        .and_then(|m| {
                            m.modified()
                                .ok()
                                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        })
                        .map(|d| d.as_secs())
                        .unwrap_or(0);

                    let symbols = if let Some((cached_mtime, cached_symbols)) = cache.get(&rel_path)
                    {
                        if *cached_mtime == mtime {
                            cached_symbols.clone()
                        } else {
                            Self::extract_symbols_from_file(path, ext).symbols
                        }
                    } else {
                        Self::extract_symbols_from_file(path, ext).symbols
                    };

                    let mut file_map = format!("  FILE: {:?}\n", path);
                    for sym in &symbols {
                        file_map.push_str(&format!("    SYMBOL: {}\n", sym));
                    }
                    return (rel_path, Some(file_map), Some((mtime, symbols)));
                }
                (rel_path, None, None)
            })
            .collect();

        let mut new_cache = HashMap::new();
        for (rel_path, map_opt, cache_opt) in processed_results {
            let rel_p: String = rel_path;
            if let Some(part) = map_opt {
                let p: String = part;
                map.push_str(&p);
            }
            if let Some(cp) = cache_opt {
                new_cache.insert(rel_p, cp);
            }
        }

        if let Err(e) = fs::write(
            cache_path,
            serde_json::to_string(&new_cache).unwrap_or_default(),
        ) {
            eprintln!("Cannon Warning: Failed to save symbol cache: {}", e);
        }
        map
    }

    /// Extracts symbols from a single file based on its extension.
    /// Supports Rust (via `syn`), TS/JS/Python (via Regex).
    /// 
    /// # Complexity Analysis
    /// * **Time Complexity**: O(L) where L is the length of the file content. 
    ///   Regex scanning is linear. `syn` parsing is also linear w.r.t input size.
    /// * **Space Complexity**: O(S) where S is the number of symbols extracted.
    pub fn extract_symbols_from_file(path: &Path, ext: &str) -> FileSymbols {
        let mut result = FileSymbols {
            symbols: Vec::new(),
            dependencies: Vec::new(),
        };
        let content = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Cannon Warning: Failed to read {:?}: {}", path, e);
                return result;
            }
        };

        if ext == "rs" {
            match syn::parse_file(&content) {
                Ok(file) => {
                    let mut visitor = SemanticVisitor {
                        symbols: Vec::new(),
                        dependencies: Vec::new(),
                    };
                    visitor.visit_file(&file);
                    result.symbols = visitor.symbols;
                    result.dependencies = visitor.dependencies;
                    return result;
                }
                Err(e) => {
                    eprintln!("Cannon Warning: Parse failed for {:?}: {}. Falling back to Regex.", path, e);
                }
            }
        }

        // Fallback for non-Rust or parse failures
        let re_str = match ext {
            "rs" => Some(r"(?m)^(pub\s+)?(fn|struct|enum|type|trait|impl)\s+([A-Za-z0-9_]+)"),
            "js" | "ts" => Some(r"(?m)^(export\s+)?(function|class|interface|type|const)\s+([A-Za-z0-9_]+)"),
            "py" => Some(r"(?m)^def\s+([A-Za-z0-9_]+)"),
            _ => None,
        };

        if let Some(pattern) = re_str {
            if let Ok(re) = Regex::new(pattern) {
                for cap in re.captures_iter(&content) {
                    if let Some(m) = cap.get(0) {
                        result.symbols.push(m.as_str().to_string());
                    }
                }
            }
        }
        result
    }

    /// Synchronizes the workspace symbols into the memory graph.
    /// 
    /// # Complexity Analysis
    /// * **Time Complexity**: O(N) where N is number of files.
    /// * **Space Complexity**: O(M) where M is number of symbols.
    pub async fn sync_graph(root: &Path, memory: &Memory) -> Result<()> {
        let entries: Vec<_> = WalkDir::new(root)
            .into_iter()
            .filter_entry(|e| !is_hidden(e))
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .collect();

        for entry in entries {
            let path = entry.path();
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            if INCLUDE_EXTENSIONS.contains(&ext) {
                let data = Self::extract_symbols_from_file(path, ext);
                for sym in &data.symbols {
                    let sym_str: String = sym.to_string();
                    let node = GraphNode {
                        id: format!("{}:{}", path.display(), sym_str),
                        content: sym_str,
                        node_type: if sym.starts_with("fn") { "fn" } else { "type" }.to_string(),
                        path: path.to_string_lossy().to_string(),
                        edges: data.dependencies.clone(),
                    };
                    if let Err(e) = memory.add_node(&node).await {
                        eprintln!("Failed to index symbol {}: {}", sym, e);
                    }
                }
            }
        }
        Ok(())
    }

    pub async fn read_url(url: &str) -> Result<String> {
        let client = reqwest::Client::new();
        let body = client.get(url).send().await?.text().await?;
        // Simple HTML to text (keep it Nano)
        let text = if let Ok(re) = Regex::new(r"<[^>]*>") {
            re.replace_all(&body, " ").to_string()
        } else {
            body // Fallback: return raw HTML if regex fails
        };
        Ok(text)
    }

    pub async fn search_web(query: &str) -> Result<String> {
        // Placeholder for real search (e.g. Tavily)
        Ok(format!(
            "Search results for '{}': [Placeholder - implement via MCP for best results]",
            query
        ))
    }

    pub fn capture_screen(description: Option<&str>) -> Result<String> {
        let captures_dir = Path::new(RALPH_DIR).join("captures");
        fs::create_dir_all(&captures_dir)?;

        let timestamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let filename = format!("capture_{}.png", timestamp);
        let filepath = captures_dir.join(&filename);

        // Use macOS screencapture command
        let output = Command::new("screencapture")
            .arg("-x") // No sound
            .arg("-C") // Capture cursor
            .arg(&filepath)
            .output()?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "screencapture failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        println!("📸 Captured: {}", filepath.display());
        if let Some(desc) = description {
            println!("   Context: {}", desc);
        }

        Ok(filepath.to_string_lossy().to_string())
    }

    pub fn read_capture_as_base64(path: &str) -> Result<String> {
        use std::io::Read;
        let mut file = fs::File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        Ok(base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            &buffer,
        ))
    }

    pub fn exec_shell(cmd: &str, autonomous: bool) -> Result<String> {
        if autonomous {
            println!(
                "{}",
                format!("⚡ AUTONOMOUS EXECUTION: {}", cmd).yellow().bold()
            );
            let output = Command::new("sh").arg("-c").arg(cmd).output()?;
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            if !stdout.is_empty() {
                println!("{}", stdout.dimmed());
            }
            if !stderr.is_empty() {
                eprintln!("{}", stderr.red().dimmed());
            }

            Ok(format!("STDOUT:\n{}\nSTDERR:\n{}", stdout, stderr))
        } else {
            println!("{}", format!("> {}", cmd).cyan());
            print!("Execute? [y/N]: ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;

            if input.trim().eq_ignore_ascii_case("y") {
                let output = Command::new("sh").arg("-c").arg(cmd).output()?;
                io::stdout().write_all(&output.stdout)?;
                io::stderr().write_all(&output.stderr)?;

                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                Ok(format!("STDOUT:\n{}\nSTDERR:\n{}", stdout, stderr))
            } else {
                println!("{}", "Skipped.".yellow());
                Ok("User skipped execution.".to_string())
            }
        }
    }

    pub fn update_tasks_md() -> Result<()> {
        let tasks_path = Path::new("TASKS.md");
        if !tasks_path.exists() {
            return Ok(());
        }
        let content = fs::read_to_string(tasks_path)?;

        // Final Sync: Archive completed tasks if they exceed limit
        let compressed = Self::compress_tasks(&content);
        fs::write(tasks_path, compressed)?;
        Ok(())
    }

    fn compress_tasks(content: &str) -> String {
        let mut completed = Vec::new();
        let mut active = Vec::new();
        let mut header = Vec::new();

        for line in content.lines() {
            if line.contains("[x]") {
                completed.push(line.to_string());
            } else if line.contains("[ ]") {
                active.push(line.to_string());
            } else if !line.trim().is_empty() {
                header.push(line.to_string());
            }
        }

        if completed.len() > 5 {
            let archived_count = completed.len() - 2;
            let mut final_content = header.join("\n");
            final_content.push_str(&format!(
                "\n\n> [!NOTE]\n> Archived {} completed tasks.\n\n",
                archived_count
            ));
            for line in completed.iter().skip(archived_count) {
                final_content.push_str(&format!("{}\n", line));
            }
            for line in active {
                final_content.push_str(&format!("{}\n", line));
            }
            final_content
        } else {
            content.to_string()
        }
    }

    pub fn check_all_tasks_complete() -> bool {
        let path = Path::new("TASKS.md");
        if !path.exists() {
            return false;
        }
        if let Ok(content) = fs::read_to_string(path) {
            return !content.contains("[ ]");
        }
        false
    }

    pub fn git_sync() -> Result<()> {
        // We need to re-implement or call Security::scan_git_diff provided it's moved or accessible.
        // For now, let's assume Security is also moving or we'll move it.
        // Actually Security struct is small and in main.rs. I'll move it here too.
        Security::scan_git_diff()?;
        Command::new("git").args(["add", "."]).output()?;
        let timestamp = Utc::now().to_rfc3339();
        let msg = format!("Ralph: Autonomous update [{}]", timestamp);
        let _ = Command::new("git").args(["commit", "-m", &msg]).output(); // Ignore commit failure (e.g. no changes)
        let out = Command::new("git").args(["push"]).output()?;
        if out.status.success() {
            println!("{}", "🚀 Active Persistence: Synced to git.".green());
        }
        Ok(())
    }
}

pub struct Security;

impl Security {
    pub fn scan_git_diff() -> Result<()> {
        let output = Command::new("git").args(["diff", "--cached"]).output()?;
        let diff = String::from_utf8_lossy(&output.stdout);
        let google_re = match Regex::new(r"AIzaSy[A-Za-z0-9-_]{33}") {
            Ok(re) => re,
            Err(_) => return Ok(()), // Regex fail -> assume safe (fail open to avoid blocking git)
        };
        if google_re.is_match(&diff) {
            return Err(anyhow::anyhow!(
                "🚨 SECURITY ALERT: Google API Key detected in git staging! Operation aborted."
            ));
        }
        Ok(())
    }
}

fn is_hidden(entry: &walkdir::DirEntry) -> bool {
    let name = entry.file_name().to_str().unwrap_or("");
    if name == "." || name.is_empty() {
        return false;
    }
    if SKIP_DIRS.contains(&name) {
        return true;
    }
    if name.starts_with(".") && name != ".env" && name != ".ralph" && name != ".gitignore" {
        return true;
    }
    false
}
