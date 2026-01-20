use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use colored::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
mod api_server;
mod benchmarker;
mod daemon;
mod fingerprint;
mod git_context;
mod janitor;
mod knowledge;
mod mcp_client;
mod memory;
mod qa;
mod reflexion;
mod replicator;
mod swarm;
mod wasm_runtime;
// Phase 6: Debate
mod debate;
mod lint;

use debate::{Debate, DebateSynthesis};
use fs_extra::dir::CopyOptions;
use futures::StreamExt;
use janitor::Janitor;
use knowledge::KnowledgeEngine;
use lint::{LintViolation, SemanticLinter};
use mcp_client::McpManager;
use memory::{GraphNode, Memory};
use quote::ToTokens;
use rayon::prelude::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use similar::{ChangeTag, TextDiff};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use syn::{
    visit::{self, Visit},
    ItemEnum, ItemFn, ItemImpl, ItemStruct,
};
use tokio::time::{sleep, Duration};
use walkdir::WalkDir;

// --- CONSTANTS ---
const PRIMARY_MODEL: &str = "gemini-3-pro-preview";
const FALLBACK_MODEL: &str = "gemini-3-pro-preview";
const COMPRESSION_MODEL: &str = "gemini-3.0-flash";
#[allow(dead_code)]
const COMPRESSION_FALLBACK: &str = "gemini-2.5-flash";
const RALPH_DIR: &str = ".ralph";
const INCLUDE_EXTENSIONS: &[&str] = &["rs", "py", "js", "ts", "html", "css", "toml", "md"];
const SKIP_DIRS: &[&str] = &[
    "target",
    ".git",
    ".ralph",
    "node_modules",
    "lancedb",
    "shadow",
];
#[allow(dead_code)]
const COMPRESSION_THRESHOLD: usize = 10000; // Compress prompts > 10k chars

// --- CORE TYPES ---

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
enum RalphRole {
    Supervisor,
    Executor,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct RalphConfig {
    project_name: String,
    primary_model: String,
    fallback_model: String,
    #[serde(default)]
    autonomous_mode: bool,
    #[serde(default = "default_max_loops")]
    max_autonomous_loops: usize,
    #[serde(default = "default_role")]
    role: RalphRole,
}

struct ContextManager {
    hot_files: HashMap<String, std::time::SystemTime>,
}

impl ContextManager {
    fn new() -> Self {
        Self {
            hot_files: HashMap::new(),
        }
    }

    fn mark_hot(&mut self, path: &str) {
        self.hot_files
            .insert(path.to_string(), std::time::SystemTime::now());
    }

    fn is_hot(&self, path: &str) -> bool {
        if let Some(time) = self.hot_files.get(path) {
            if let Ok(elapsed) = time.elapsed() {
                return elapsed.as_secs() < 300; // Hot for 5 minutes
            }
        }
        false
    }
}

fn default_max_loops() -> usize {
    50
}
fn default_role() -> RalphRole {
    RalphRole::Executor
}

impl Default for RalphConfig {
    fn default() -> Self {
        Self {
            project_name: "unnamed_workspace".to_string(),
            primary_model: PRIMARY_MODEL.to_string(),
            fallback_model: FALLBACK_MODEL.to_string(),
            autonomous_mode: false,
            max_autonomous_loops: 50,
            role: RalphRole::Executor,
        }
    }
}
// --- TELEMETRY (Haptic Feedback) ---

struct Telemetry;
impl Telemetry {
    fn notify(title: &str, message: &str) {
        let script = format!(
            "display notification \"{}\" with title \"{}\" sound name \"Hero\"",
            message, title
        );
        let _ = Command::new("osascript").arg("-e").arg(script).spawn();
    }

    fn speak(text: &str) {
        let _ = Command::new("say").arg(text).arg("-r").arg("220").spawn();
    }
}

// --- CORTEX (The Brain) ---

struct Cortex {
    api_key: String,
    client: reqwest::Client,
    config: RalphConfig,
    memory: Arc<Memory>,
}

impl Cortex {
    fn new(config: RalphConfig, memory: Arc<Memory>) -> Result<Self> {
        let api_key = env::var("GEMINI_API_KEY")
            .context("CRITICAL: GEMINI_API_KEY not found in .env or environment")?;

        Ok(Self {
            api_key,
            client: reqwest::Client::new(),
            config,
            memory,
        })
    }

    async fn create_context_cache(&self, context: &str) -> Result<String> {
        let mut hasher = Sha256::new();
        hasher.update(context.as_bytes());
        let hash = hex::encode(hasher.finalize());

        if let Ok(Some(cached_id)) = self.memory.get_kv_cache(&hash).await {
            return Ok(cached_id);
        }

        println!("{}", "🧠 Creating Gemini Context Cache...".cyan());
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/cachedContents?key={}",
            self.api_key
        );

        // Context caching requires at least 32,768 tokens.
        // For Nano, we'll try it if the context is large enough.
        let payload = serde_json::json!({
            "model": format!("models/{}", self.config.primary_model),
            "contents": [{ "parts": [{ "text": context }] }],
            "ttl": "3600s"
        });

        let res = self.client.post(&url).json(&payload).send().await?;
        let status = res.status();
        if !status.is_success() {
            let err_text = res.text().await.unwrap_or_default();
            return Err(anyhow!("Failed to create cache: {} - {}", status, err_text));
        }

        let val: Value = res.json().await?;
        let cache_id = val["name"]
            .as_str()
            .context("Cache ID not found in response")?
            .to_string();

        let _ = self.memory.set_kv_cache(&hash, &cache_id).await;
        Ok(cache_id)
    }

    async fn stream_api(
        &self,
        model: &str,
        prompt: &str,
        cache_id: Option<&str>,
        images: Option<Vec<String>>,
    ) -> Result<impl futures::Stream<Item = Result<String>>> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?key={}",
            model, self.api_key
        );

        let mut parts = vec![serde_json::json!({ "text": prompt })];

        if let Some(imgs) = images {
            for img_data in imgs {
                parts.push(serde_json::json!({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_data
                    }
                }));
            }
        }

        let mut payload = serde_json::json!({
            "contents": [{ "parts": parts }]
        });

        if let Some(cid) = cache_id {
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("cachedContent".to_string(), Value::String(cid.to_string()));
            }
        }

        let res = self.client.post(&url).json(&payload).send().await?;

        if res.status() == 429 {
            return Err(anyhow::anyhow!("RATE_LIMIT"));
        }
        if !res.status().is_success() {
            return Err(anyhow::anyhow!("API Error: {}", res.status()));
        }

        let stream = res.bytes_stream().map(|item| {
            item.map_err(anyhow::Error::from).map(|bytes| {
                let s = String::from_utf8_lossy(&bytes).to_string();
                let sanitized = s
                    .trim()
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .trim_start_matches(',');
                if sanitized.is_empty() {
                    return String::new();
                }

                match serde_json::from_str::<serde_json::Value>(sanitized) {
                    Ok(val) => {
                        let text = val["candidates"][0]["content"]["parts"][0]["text"]
                            .as_str()
                            .unwrap_or("")
                            .to_string();
                        text
                    }
                    Err(_) => String::new(),
                }
            })
        });
        Ok(stream)
    }

    async fn stream_local_llm(
        &self,
        prompt: &str,
    ) -> Result<impl futures::Stream<Item = Result<String>> + Send + Unpin> {
        let url = "http://localhost:11434/api/generate";
        let payload = serde_json::json!({
            "model": "llama3.2",
            "prompt": prompt,
            "stream": true
        });
        let res = self.client.post(url).json(&payload).send().await?;

        let stream = res.bytes_stream().map(|item| {
            item.map_err(anyhow::Error::from).map(|bytes| {
                let s = String::from_utf8_lossy(&bytes).to_string();
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&s) {
                    let text = val["response"].as_str().unwrap_or("").to_string();
                    text
                } else {
                    String::new()
                }
            })
        });
        Ok(stream.boxed())
    }

    async fn stream_generate(
        &self,
        prompt: &str,
        context: Option<&str>,
        images: Option<Vec<String>>,
    ) -> Result<futures::stream::BoxStream<'static, Result<String>>> {
        use futures::StreamExt;

        let cache_id = if let Some(ctx) = context {
            if ctx.len() > 10000 && images.is_none() {
                // Only cache text-only contexts for now
                self.create_context_cache(ctx).await.ok()
            } else {
                None
            }
        } else {
            None
        };

        match self
            .stream_api(
                &self.config.primary_model,
                prompt,
                cache_id.as_deref(),
                images.clone(),
            )
            .await
        {
            Ok(s) => Ok(s.boxed()),
            Err(_) => match self
                .stream_api(&self.config.fallback_model, prompt, None, images)
                .await
            {
                Ok(s) => Ok(s.boxed()),
                Err(_) => {
                    let s = self.stream_local_llm(prompt).await?;
                    Ok(s.boxed())
                }
            },
        }
    }

    #[allow(dead_code)]
    async fn compress_prompt(&self, prompt: &str) -> Result<String> {
        if prompt.len() < COMPRESSION_THRESHOLD {
            return Ok(prompt.to_string());
        }

        println!("{}", "🗜️ Compressing context...".dimmed());
        let compression_prompt = format!(
            "Summarize the following context into a concise but complete form. \
             Preserve all critical technical details, file paths, and code snippets. \
             Remove redundancy and verbose explanations.\n\nCONTEXT:\n{}",
            prompt
        );

        // Try primary compression model, fallback if needed
        let result = match self
            .generate_sync(COMPRESSION_MODEL, &compression_prompt)
            .await
        {
            Ok(s) => s,
            Err(_) => match self
                .generate_sync(COMPRESSION_FALLBACK, &compression_prompt)
                .await
            {
                Ok(s) => s,
                Err(_) => return Ok(prompt.to_string()), // Fallback to original
            },
        };

        let original_len = prompt.len();
        let compressed_len = result.len();
        println!(
            "   Compressed: {} -> {} chars ({:.0}% reduction)",
            original_len,
            compressed_len,
            ((original_len - compressed_len) as f64 / original_len as f64) * 100.0
        );

        Ok(result)
    }

    async fn generate_sync(&self, model: &str, prompt: &str) -> Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model, self.api_key
        );

        let payload = serde_json::json!({
            "contents": [{"parts": [{"text": prompt}]}]
        });

        let res = self.client.post(&url).json(&payload).send().await?;
        let body: Value = res.json().await?;
        body["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .map(|s| s.to_string())
            .context("No text in response")
    }

    /// Phase 6: Self-Correcting Debate
    async fn conduct_debate(&self, topic: &str, context: &str) -> Result<DebateSynthesis> {
        println!(
            "{}",
            format!("\n⚖️  CONDUCTING DEBATE: {}", topic)
                .yellow()
                .bold()
        );

        let debate = Debate::security_vs_performance();
        let prompts = debate.generate_prompts(context, topic);

        // Run persona critiques in parallel
        let mut rounds = Vec::new();
        let mut handles = Vec::new();

        for (persona_name, prompt) in prompts {
            let model = self.config.primary_model.clone();
            let p_name = persona_name.clone();
            let cortex_ref = self.client.clone(); // Clone reqwest client (cheap)
            let api_key = self.api_key.clone();

            handles.push(tokio::spawn(async move {
                // Inline generation to avoid referencing self across threads
                let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}", model, api_key);
                let payload = serde_json::json!({ "contents": [{"parts": [{"text": prompt}]}] });

                if let Ok(res) = cortex_ref.post(&url).json(&payload).send().await {
                    if let Ok(body) = res.json::<Value>().await {
                    if let Some(text) = body["candidates"][0]["content"]["parts"][0]["text"].as_str() {
                        return Some((p_name, text.to_string()));
                    }
                } }
                None
            }));
        }

        for handle in handles {
            if let Ok(Some((name, response))) = handle.await {
                let round = Debate::parse_response(&name, &response);
                println!(
                    "  🗣️  {}: Found {} issues.",
                    name.cyan(),
                    round.suggestions.len()
                );
                rounds.push(round);
            }
        }

        Ok(Debate::synthesize(&rounds))
    }

    /// Phase 6: Semantic Linting
    async fn perform_lint(&self, path: &str) -> Result<Vec<LintViolation>> {
        println!("{}", format!("\n🕵️  LINTING: {}", path).yellow().bold());

        let code = if Path::new(path).exists() {
            fs::read_to_string(path).unwrap_or_default()
        } else {
            return Err(anyhow::anyhow!("File not found: {}", path));
        };

        if code.is_empty() {
            return Ok(vec![]);
        }

        // Get context from graph for this file
        let context = match self.memory.get_neighborhood(path).await {
            Ok(neighbors) => neighbors.join("\n"),
            Err(_) => String::new(),
        };

        let prompt = SemanticLinter::lint_prompt(&code, &context);
        let response = self
            .generate_sync(&self.config.primary_model, &prompt)
            .await?;

        Ok(SemanticLinter::parse_response(&response))
    }
}

struct IncrementalParser {
    buffer: String,
    processed_index: usize,
}

impl IncrementalParser {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            processed_index: 0,
        }
    }

    fn push(&mut self, token: &str) -> Vec<Action> {
        self.buffer.push_str(token);
        let mut actions = Vec::new();
        let master_re =
            Regex::new(r"```(?P<lang>\w+)?\s*(?P<header>.*)\n(?P<content>[\s\S]*?)```").unwrap();

        // Scan new content
        let sub_buffer = &self.buffer[self.processed_index..];
        for cap in master_re.captures_iter(sub_buffer) {
            let full_match = cap.get(0).unwrap();
            let block_str = full_match.as_str();

            let block_actions = extract_code_blocks(block_str);
            actions.extend(block_actions);

            self.processed_index += full_match.end();
        }
        actions
    }
}

// --- TOOLS (The Hands) ---

struct Tools;

impl Tools {
    fn scan_workspace(root: &Path, manager: &ContextManager) -> String {
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

    fn scan_symbols(root: &Path) -> String {
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
}

pub struct FileSymbols {
    pub symbols: Vec<String>,
    pub dependencies: Vec<String>,
}

impl Tools {
    fn extract_symbols_from_file(path: &Path, ext: &str) -> FileSymbols {
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

    async fn sync_graph(root: &Path, memory: &Memory) -> Result<()> {
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

    async fn read_url(url: &str) -> Result<String> {
        let client = reqwest::Client::new();
        let body = client.get(url).send().await?.text().await?;
        // Simple HTML to text (keep it Nano)
        let re = Regex::new(r"<[^>]*>").unwrap();
        let text = re.replace_all(&body, " ").to_string();
        Ok(text)
    }

    async fn search_web(query: &str) -> Result<String> {
        // Placeholder for real search (e.g. Tavily)
        Ok(format!(
            "Search results for '{}': [Placeholder - implement via MCP for best results]",
            query
        ))
    }

    fn capture_screen(description: Option<&str>) -> Result<String> {
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

    fn read_capture_as_base64(path: &str) -> Result<String> {
        use std::io::Read;
        let mut file = fs::File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        Ok(base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            &buffer,
        ))
    }

    fn exec_shell(cmd: &str, autonomous: bool) -> Result<String> {
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

    fn update_tasks_md() -> Result<()> {
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

    fn check_all_tasks_complete() -> bool {
        let path = Path::new("TASKS.md");
        if !path.exists() {
            return false;
        }
        if let Ok(content) = fs::read_to_string(path) {
            return !content.contains("[ ]");
        }
        false
    }

    fn git_sync() -> Result<()> {
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

// --- SHADOW WORKSPACE ---

struct ShadowWorkspace {
    root: PathBuf,
}

impl ShadowWorkspace {
    fn new() -> Self {
        Self {
            root: Path::new(RALPH_DIR).join("shadow"),
        }
    }

    fn prepare_shadow(&self) -> Result<()> {
        if !self.root.exists() {
            fs::create_dir_all(&self.root)?;
        }

        // --- Shared Cache: Symlink target to reduce compile times ---
        let main_target = env::current_dir()?.join("target");
        let shadow_target = self.root.join("target");
        if main_target.exists() && !shadow_target.exists() {
            #[cfg(unix)]
            let _ = std::os::unix::fs::symlink(&main_target, &shadow_target);
        }

        let options = CopyOptions::new().overwrite(true).content_only(true);
        // --- Lazy Shadows: Differential Mirroring ---
        for entry in fs::read_dir(".")? {
            let entry = entry?;
            let name = entry.file_name().to_str().unwrap_or("").to_string();
            let path = entry.path();

            if SKIP_DIRS.contains(&name.as_str()) || name == "target" || name == "node_modules" {
                continue;
            }

            let dest_path = self.root.join(&name);

            // Basic Delta Check: only copy if dest doesn't exist or size/mtime differs
            let source_meta = fs::metadata(&path)?;
            let should_copy = if !dest_path.exists() {
                true
            } else {
                let dest_meta = fs::metadata(&dest_path)?;
                source_meta.len() != dest_meta.len()
                    || source_meta.modified()? != dest_meta.modified()?
            };

            if should_copy {
                if entry.file_type()?.is_dir() {
                    fs_extra::dir::copy(&path, &self.root, &options)?;
                } else {
                    fs::copy(&path, &dest_path)?;
                }
            }
        }
        Ok(())
    }

    fn is_safe_path(path: &str) -> bool {
        let p = Path::new(path);
        !p.is_absolute() && !path.contains("..")
    }

    fn stage_edit(&self, rel_path: &str, content: &str) -> Result<()> {
        if !Self::is_safe_path(rel_path) {
            return Err(anyhow::anyhow!("CRITICAL: Path traversal attempt: {}", rel_path));
        }
        let full_path = self.root.join(rel_path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(full_path, content)?;
        Ok(())
    }

    fn commit(&self, rel_path: &str) -> Result<()> {
        if !Self::is_safe_path(rel_path) {
            return Err(anyhow::anyhow!("CRITICAL: Path traversal attempt: {}", rel_path));
        }
        let shadow_path = self.root.join(rel_path);
        let real_path = Path::new(rel_path);
        if let Some(parent) = real_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(shadow_path, real_path)?;
        Ok(())
    }
}

struct HotCheck {
    child: Option<std::process::Child>,
    root: PathBuf,
}

impl HotCheck {
    fn new() -> Self {
        Self {
            child: None,
            root: Path::new(RALPH_DIR).join("shadow"),
        }
    }

    fn trigger(&mut self) -> Result<()> {
        // Kill existing check if running
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
        }

        let child = Command::new("cargo")
            .arg("check")
            .arg("--message-format=json")
            .current_dir(&self.root)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()?;

        self.child = Some(child);
        Ok(())
    }

    fn poll_results(&mut self) -> Result<bool> {
        if let Some(mut child) = self.child.take() {
            let status = child.wait()?;
            return Ok(status.success());
        }
        Ok(true)
    }
}

fn show_diff(path: &str, new_content: &str) -> Result<()> {
    let old_content = if Path::new(path).exists() {
        fs::read_to_string(path)?
    } else {
        String::new()
    };
    let diff = TextDiff::from_lines(old_content.as_str(), new_content);
    println!("{}", format!("\n--- DIFF: {} ---", path).cyan().bold());
    for change in diff.iter_all_changes() {
        let sign = match change.tag() {
            ChangeTag::Delete => "-".red(),
            ChangeTag::Insert => "+".green(),
            ChangeTag::Equal => " ".dimmed(),
        };
        print!("{}{}", sign, change);
    }
    println!("\n------------------");
    Ok(())
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

// --- SECURITY ---

struct Security;

impl Security {
    fn scan_git_diff() -> Result<()> {
        let output = Command::new("git").args(["diff", "--cached"]).output()?;
        let diff = String::from_utf8_lossy(&output.stdout);
        let google_re = Regex::new(r"AIzaSy[A-Za-z0-9-_]{33}").unwrap();
        if google_re.is_match(&diff) {
            return Err(anyhow::anyhow!(
                "🚨 SECURITY ALERT: Google API Key detected in git staging! Operation aborted."
            ));
        }
        Ok(())
    }
}

// --- SMART PARSER ---

#[derive(Debug, Clone)]
enum Action {
    WriteFile(String, String),
    ExecShell(String),
    Directive(serde_json::Value),
    CallMcpTool(String, String, serde_json::Value),
    ReadUrl(String),
    SearchWeb(String),
    CaptureUI(Option<String>),
    CallWasm(String, Vec<i32>), // function name, args
    Debate(String),             // topic/proposal
    Snapshot(String),           // label
    Restore(String),            // id
    Lint(String),               // path
}

fn extract_code_blocks(response: &str) -> Vec<Action> {
    let mut actions = Vec::new();
    let master_re = Regex::new(r"```(\w+)?\s*(.*)\n([\s\S]*?)```").unwrap();
    let known_langs = [
        "rust",
        "python",
        "bash",
        "sh",
        "javascript",
        "typescript",
        "toml",
        "json",
        "yaml",
        "html",
        "css",
        "sql",
        "text",
        "md",
    ];

    for cap in master_re.captures_iter(response) {
        let lang = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let header_path_raw = cap.get(2).map(|m| m.as_str().trim()).unwrap_or("");

        let header_path = header_path_raw
            .trim_start_matches("<!--")
            .trim_end_matches("-->")
            .trim_start_matches("//")
            .trim_start_matches("#")
            .trim_start_matches(":")
            .trim();

        let content = cap.get(3).map(|m| m.as_str()).unwrap_or("");

        // Priority 1: Structured JSON Directives
        if (lang == "json" || content.trim().starts_with('{')) && content.trim().ends_with('}') {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(content.trim()) {
                println!("{}", "📋 STRUCTURED DIRECTIVE RECEIVED".cyan().bold());
                actions.push(Action::Directive(val));
                continue;
            }
        }

        let is_shell_lang = lang == "bash" || lang == "sh" || lang == "shell" || lang == "zsh";
        let looks_like_file = header_path.contains('.') || header_path.contains('/');
        let is_ambiguous_lang = known_langs.contains(&header_path.to_lowercase().as_str());

        // Priority 2: Header Check (Sanitized)
        if !header_path.is_empty() && !is_shell_lang && looks_like_file && !is_ambiguous_lang {
            actions.push(Action::WriteFile(
                header_path.to_string(),
                content.to_string(),
            ));
            continue;
        }

        let mut found_path = None;
        for line in content.lines().take(2) {
            let text = line.trim();
            if text.starts_with("// ") || text.starts_with("# ") {
                let potential_path = text[2..].trim();
                if potential_path.contains('.') || potential_path.contains('/') {
                    found_path = Some(potential_path.to_string());
                    break;
                }
            }
        }
        if let Some(p) = found_path {
            actions.push(Action::WriteFile(p, content.to_string()));
            continue;
        }

        let shell_keywords = [
            "pip ", "cargo ", "npm ", "git ", "apt-get ", "ls ", "cd ", "echo ", "rm ", "mkdir ",
            "touch ",
        ];
        let has_keyword = shell_keywords
            .iter()
            .any(|kw| content.trim().starts_with(kw));

        if is_shell_lang || has_keyword {
            let mut final_cmd = content.trim().to_string();
            if final_cmd.is_empty() && !header_path.is_empty() && !looks_like_file {
                final_cmd = header_path.to_string();
            }
            if !final_cmd.is_empty() {
                actions.push(Action::ExecShell(final_cmd));
            }
        }

        if lang == "mcp" {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(content) {
                let server = val["server"].as_str().unwrap_or("").to_string();
                let tool = val["tool"].as_str().unwrap_or("").to_string();
                let args = val["arguments"].clone();
                if !server.is_empty() && !tool.is_empty() {
                    actions.push(Action::CallMcpTool(server, tool, args));
                }
            }
        }

        if lang == "url" {
            actions.push(Action::ReadUrl(content.trim().to_string()));
        }

        if lang == "search" {
            actions.push(Action::SearchWeb(content.trim().to_string()));
        }

        if lang == "capture" {
            actions.push(Action::CaptureUI(Some(content.trim().to_string())));
        }

        if lang == "wasm" {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(content) {
                let func = val["function"].as_str().unwrap_or("run").to_string();
                let args = val["args"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|v| v.as_i64().map(|i| i as i32))
                    .collect();
                actions.push(Action::CallWasm(func, args));
            }
        }

        if lang == "debate" {
            actions.push(Action::Debate(content.trim().to_string()));
        }

        if lang == "snapshot" {
            actions.push(Action::Snapshot(content.trim().to_string()));
        }

        if lang == "restore" {
            actions.push(Action::Restore(content.trim().to_string()));
        }

        if lang == "lint" {
            actions.push(Action::Lint(content.trim().to_string()));
        }
    }
    actions
}

fn init_workspace() -> Result<()> {
    let ralph_path = Path::new(RALPH_DIR);
    if ralph_path.exists() {
        println!("{}", "✅ Ralph is already alive in this workspace.".green());
        return Ok(());
    }
    fs::create_dir_all(ralph_path.join("lancedb"))?;
    fs::create_dir_all(ralph_path.join("shadow"))?;
    let config = RalphConfig::default();
    let toml = toml::to_string_pretty(&config)?;
    fs::write(ralph_path.join("config.toml"), toml)?;
    let gitignore_path = Path::new(".gitignore");
    let mut gitignore = if gitignore_path.exists() {
        fs::read_to_string(gitignore_path)?
    } else {
        String::new()
    };
    if !gitignore.contains(".ralph") {
        use std::fmt::Write as _;
        writeln!(gitignore, "\n# Ralph Agent Data\n.ralph/")?;
        fs::write(".gitignore", gitignore)?;
    }
    println!("{}", "🧬 DNA REPLICATION COMPLETE.".green().bold());
    Ok(())
}

/// Worker loop for Swarm parallelism - executes tasks from the queue
async fn run_worker_loop() -> Result<()> {
    let worker_id = env::var("RALPH_WORKER_ID").unwrap_or_else(|_| "worker-0".to_string());
    println!("🐝 Worker {} starting...", worker_id);

    dotenvy::from_filename(".env").ok();
    let config_path = format!("{}/config.toml", RALPH_DIR);
    let config_str = fs::read_to_string(&config_path).unwrap_or_default();
    let config: RalphConfig = toml::from_str(&config_str).unwrap_or_default();

    let memory = Arc::new(Memory::new(&format!("{}/lancedb", RALPH_DIR)).await?);
    let cortex = Cortex::new(config, Arc::clone(&memory))?;

    let status_path = format!("{}/swarm/{}/status.json", RALPH_DIR, worker_id);
    let task_path = format!("{}/swarm/{}/task.json", RALPH_DIR, worker_id);

    let mut completed_tasks_count = 0;

    loop {
        // Check for a task file
        if Path::new(&task_path).exists() {
            if let Ok(content) = fs::read_to_string(&task_path) {
                if let Ok(task_val) = serde_json::from_str::<Value>(&content) {
                    let task = task_val["objective"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string();
                    if !task.is_empty() {
                        println!("🐝 Worker {} executing: {}", worker_id, task);

                        // Update status to Working
                        let status = swarm::WorkerStatus {
                            id: worker_id.clone(),
                            state: swarm::WorkerState::Working,
                            current_task: Some(task.clone()),
                            assigned_task: Some(task.clone()),
                            completed_tasks: completed_tasks_count,
                            subtasks_completed: Vec::new(),
                        };
                        let _ = fs::write(
                            &status_path,
                            serde_json::to_string_pretty(&status).unwrap_or_default(),
                        );

                        // Execute the task
                        let prompt = format!("Execute this task autonomously:\n{}", task);
                        if let Ok(mut stream) = cortex.stream_generate(&prompt, None, None).await {
                            let mut response = String::new();
                            while let Some(chunk) = stream.next().await {
                                if let Ok(token) = chunk {
                                    response.push_str(&token);
                                }
                            }
                            println!("✅ Worker {} COMPLETED task.", worker_id);
                            completed_tasks_count += 1;

                            // Report completion in status
                            let final_status = swarm::WorkerStatus {
                                id: worker_id.clone(),
                                state: swarm::WorkerState::Idle,
                                current_task: None,
                                assigned_task: None,
                                completed_tasks: completed_tasks_count,
                                subtasks_completed: vec![task.clone()],
                            };
                            let _ = fs::write(
                                &status_path,
                                serde_json::to_string_pretty(&final_status).unwrap_or_default(),
                            );
                        }

                        // Remove task file to signal completion
                        let _ = fs::remove_file(&task_path);
                    }
                }
            }
        }

        sleep(Duration::from_secs(2)).await;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "init" {
        return init_workspace();
    }
    if args.len() > 1 && args[1] == "--version" {
        println!("ralph-nano v0.2.4");
        return Ok(());
    }
    if args.contains(&"--worker-mode".to_string()) {
        return run_worker_loop().await;
    }
    if !Path::new(RALPH_DIR).exists() {
        anyhow::bail!("Ralph not initialized. Run `ralph-nano init` to replicate DNA here.");
    }
    dotenvy::from_filename(".env").ok();
    let config_path = format!("{}/config.toml", RALPH_DIR);
    let config_str = fs::read_to_string(&config_path)
        .context(format!("Could not read config at {}", config_path))?;
    let config: RalphConfig = toml::from_str(&config_str)?;

    let memory = Arc::new(Memory::new(&format!("{}/lancedb", RALPH_DIR)).await?);
    let cortex = Cortex::new(config, Arc::clone(&memory))?;

    let mut loop_count = 0;
    let mut last_output = String::new();
    let mut idle_announced = false;
    let mut context_manager = ContextManager::new();
    let mut hot_check = HotCheck::new();
    let mut mcp_manager = McpManager::new();
    let mut wasm_runtime = wasm_runtime::WasmRuntime::new();
    let mut last_captured_image: Option<String> = None;

    println!("{}", "⚡ RALPH-NANO v0.2.4: ONLINE".green().bold());
    Telemetry::speak("Ralph Nano is online.");
    if cortex.config.autonomous_mode {
        println!("{}", "⚠️ WARNING: AUTONOMOUS MODE ENABLED".yellow().bold());
        Telemetry::notify("Autonomy Active", "Ralph is operating in headless mode.");
    }

    let mut swarm_manager = swarm::SwarmManager::new(Path::new("."));

    // Initialize Swarm Manager for parallel task execution
    if cortex.config.autonomous_mode {
        println!("{}", "🐝 Spawning worker swarm...".cyan());
        if let Err(e) = swarm_manager.spawn_workers(3) {
            eprintln!("Failed to spawn workers: {}", e);
        }
    }

    // Populate Knowledge Graph from workspace
    let memory_sync = Arc::clone(&memory);
    tokio::spawn(async move {
        if let Err(e) = Tools::sync_graph(Path::new("."), &memory_sync).await {
            eprintln!("Initial graph sync failed: {}", e);
        }
    });

    // --- AUTO-DIDACT: Knowledge Engine & Learning Phase ---
    let knowledge = KnowledgeEngine::new(Arc::clone(&memory));
    let knowledge_sync = knowledge.clone();
    let _autonomous_learning = cortex.config.autonomous_mode;

    tokio::spawn(async move {
        sleep(Duration::from_secs(5)).await; // Wait for core boot to finish
        println!("🧠 Auto-Didact Phase: Scanning for documentation...");
        if let Err(e) = knowledge_sync.ingest_workspace(Path::new(".")).await {
            eprintln!("Auto-Didact ingestion failed: {}", e);
        }

        // --- REGISTRY MONITOR ---
        if let Ok(updates) = knowledge_sync.check_library_updates().await {
            if !updates.is_empty() {
                println!("\n🔔 {}", "REGISTRY UPDATES AVAILABLE:".yellow().bold());
                for (lib, _local, latest) in updates {
                    println!(
                        "  * {} has version {} available.",
                        lib.cyan(),
                        latest.green()
                    );
                }
                Telemetry::notify(
                    "Documentation Updates",
                    "New library versions are available in the registry.",
                );
            }
        }
    });

    // Spawn shadow prep in background
    tokio::spawn(async move {
        let s = ShadowWorkspace::new();
        let _ = s.prepare_shadow();
    });

    // Detect project fingerprint
    let fingerprint = fingerprint::ProjectFingerprint::detect(Path::new("."));
    println!("📊 Project Type: {:?}", fingerprint.project_type);
    if !fingerprint.tech_stack.is_empty() {
        println!("   Tech Stack: {}", fingerprint.tech_stack.join(", "));
    }
    if !fingerprint.dependencies.is_empty() {
        println!(
            "   Top Deps: {}",
            fingerprint
                .dependencies
                .iter()
                .take(5)
                .map(|d| d.name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!("DEBUG: Fingerprint done.");

    // Load MCP Servers from local config
    let mcp_config_path = Path::new("mcp_config.json");
    if mcp_config_path.exists() {
        if let Ok(content) = fs::read_to_string(mcp_config_path) {
            if let Ok(val) = serde_json::from_str::<Value>(&content) {
                if let Some(servers) = val["mcpServers"].as_object() {
                    for (name, server) in servers {
                        let cmd = server["command"].as_str().unwrap_or("");
                        let args_val = server["args"].as_array();
                        let args: Vec<&str> = if let Some(av) = args_val {
                            av.iter().filter_map(|v| v.as_str()).collect()
                        } else {
                            Vec::new()
                        };

                        if !cmd.is_empty() {
                            if let Err(e) = mcp_manager.add_server(name, cmd, &args).await {
                                eprintln!("Failed to start MCP server {}: {}", name, e);
                            } else {
                                println!("🔌 Connected to MCP server: {}", name.cyan());
                            }
                        }
                    }
                }
            }
        }
    }

    println!("DEBUG: MCP checks done.");
    // --- AUTO-DIDACT: Knowledge Engine Scan ---
    println!("DEBUG: Starting Knowledge Engine Scan...");
    let knowledge = KnowledgeEngine::new(Arc::clone(&memory));
    if let Ok(libs) = knowledge.scan_all_dependencies() {
        println!("DEBUG: Scanned dependencies. Fetching known libs...");
        let known_libs = memory.get_known_libraries().await.unwrap_or_default();
        println!("DEBUG: Got known libs: {}", known_libs.len());
        let missing_libs: Vec<_> = libs
            .iter()
            .filter(|l| !known_libs.contains(&l.name))
            .collect();

        if !missing_libs.is_empty() {
            if cortex.config.autonomous_mode {
                for lib in missing_libs {
                    let _ = knowledge.ingest_library(lib).await;
                }
            } else {
                print!(
                    "📚 Found new libraries: {}. Ingest docs? [y/N]: ",
                    missing_libs
                        .iter()
                        .map(|l| l.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                        .cyan()
                );
                io::stdout().flush()?;
                let mut confirm = String::new();
                io::stdin().read_line(&mut confirm)?;
                if confirm.trim().eq_ignore_ascii_case("y") {
                    for lib in missing_libs {
                        let _ = knowledge.ingest_library(lib).await;
                    }
                }
            }
        }
    }

    println!("DEBUG: Knowledge Engine Scan done.");
    // --- REGISTRY MONITOR ---
    let knowledge_sync = knowledge.clone();
    tokio::spawn(async move {
        sleep(Duration::from_secs(10)).await; // Wait for initial boot spikes to settle
        if let Ok(updates) = knowledge_sync.check_library_updates().await {
            let updates: Vec<(String, String, String)> = updates;
            if !updates.is_empty() {
                println!("\n🔔 {}", "REGISTRY UPDATES AVAILABLE:".yellow().bold());
                for (lib, _local, latest) in updates {
                    println!(
                        "  * {} has version {} available.",
                        lib.cyan(),
                        latest.green()
                    );
                }
                Telemetry::notify(
                    "Documentation Updates",
                    "New library versions are available in the registry.",
                );
            }
        }
    });

    loop {
        // POLLING STOP CONDITION
        if Tools::check_all_tasks_complete() {
            if !idle_announced {
                println!(
                    "{}",
                    "\n✅ All tasks complete. Watching TASKS.md for new items..."
                        .green()
                        .bold()
                );
                let _ = Tools::git_sync();
                idle_announced = true;
            }
            sleep(Duration::from_secs(5)).await;
            continue;
        }

        // Resume from idle
        if idle_announced {
            println!("{}", "🚀 New tasks detected! Resuming autonomy...".green());
            Telemetry::speak("New tasks detected. Resuming autonomy.");
            idle_announced = false;
        }

        // Loop Cap (The Circuit Breaker)
        if cortex.config.autonomous_mode {
            loop_count += 1;
            if loop_count > cortex.config.max_autonomous_loops {
                println!(
                    "{}",
                    "\n🛑 MAX AUTONOMOUS LOOPS REACHED. STOPPING.".red().bold()
                );
                Telemetry::notify(
                    "Circuit Breaker",
                    "Max loop count reached. Stopping safely.",
                );
                Telemetry::speak("Autonomous loop capacity reached. Shutting down.");
                break;
            }
            println!(
                "\n🌀 Loop {}/{}",
                loop_count, cortex.config.max_autonomous_loops
            );
        }

        let input;

        if cortex.config.autonomous_mode {
            if !last_output.is_empty() {
                input = format!("Previous command output:\n{}\n\nProceed next.", last_output);
                last_output.clear();
            } else {
                input = "Proceed with the next unchecked task in TASKS.md.".to_string();
            }
        } else {
            print!("{}", "\nralph> ".blue().bold());
            io::stdout().flush()?;
            let mut buf = String::new();
            if io::stdin().read_line(&mut buf)? == 0 {
                break;
            }
            input = buf.trim().to_string();
            if input.is_empty() {
                continue;
            }
            if input == "exit" {
                break;
            }
        }

        println!("{}", "reading workspace...".dimmed());

        let config_clone = cortex.config.clone();
        let hot_list = context_manager
            .hot_files
            .keys()
            .cloned()
            .collect::<Vec<_>>();

        let hot_list_sync = hot_list.clone();
        let context_handle = tokio::spawn(async move {
            let mut temp_manager = ContextManager::new();
            for f in hot_list_sync {
                temp_manager.mark_hot(&f);
            }

            if config_clone.role == RalphRole::Supervisor {
                Tools::scan_symbols(Path::new("."))
            } else {
                Tools::scan_workspace(Path::new("."), &temp_manager)
            }
        });

        if let Ok(facts) = memory.recall_facts(&input).await {
            let facts: Vec<String> = facts;
            if !facts.is_empty() {
                println!("\n🧠 {}", "RECALLED FACTS:".green().bold());
                for fact in &facts {
                    println!("  * {}", fact.cyan());
                }
            }
        }

        // Recall heuristics from Reflexion
        let mut heuristics_context = String::new();
        if let Ok(heuristics) = memory.recall_heuristics(&input).await {
            if !heuristics.is_empty() {
                println!("\n💡 {}", "RECALLED HEURISTICS:".yellow().bold());
                for h in &heuristics {
                    println!("  * {}", h.dimmed());
                    heuristics_context.push_str(&format!("- {}\n", h));
                }
            }
        }

        let context = context_handle.await.context("Context scan failed")?;

        // --- AUTO-DIDACT: Search Official Docs (with Re-ranking) ---
        let mut docs_context = String::new();
        // Phase 5: Fetch more candidates (20) and re-rank locally to top 3
        if let Ok(candidates) = memory.search_library(&input, 20).await {
            if !candidates.is_empty() {
                let docs = match memory.rerank(&input, candidates, 3).await {
                    Ok(ranked) => ranked,
                    Err(e) => {
                        eprintln!("Re-ranking failed: {}, falling back to raw search.", e);
                        memory.search_library(&input, 3).await.unwrap_or_default()
                    }
                };

                if !docs.is_empty() {
                    println!(
                        "\n📚 {}",
                        "OFFICIAL DOCS CONTEXT (RE-RANKED):".cyan().bold()
                    );
                    docs_context.push_str("[OFFICIAL_DOCS_CONTEXT]\n");
                    for doc in &docs {
                        println!("  * Found relevant documentation chunk.");
                        docs_context.push_str(&format!("- {}\n", doc));
                    }
                    docs_context.push('\n');
                }
            }
        }

        // Gather MCP Tools
        let mut mcp_context = String::new();
        if let Ok(all_tools) = mcp_manager.get_all_tools().await {
            for (server, tools) in all_tools {
                mcp_context.push_str(&format!("\nMCP SERVER: {}\n", server));
                for tool in tools {
                    mcp_context.push_str(&format!(
                        "  TOOL: {}\n    DESC: {}\n    SCHEMA: {}\n",
                        tool.name,
                        tool.description,
                        serde_json::to_string(&tool.input_schema).unwrap_or_default()
                    ));
                }
            }
        }

        // --- PREDICTIVE CONTEXT: Graph Neighborhood ---
        let mut predictive_context = String::new();
        if !hot_list.is_empty() {
            for f in hot_list.iter().take(3) {
                if let Ok(neighbors) = memory.get_neighborhood(f).await {
                    if !neighbors.is_empty() {
                        predictive_context.push_str(&format!("\nPREDICTIVE CONTEXT for {}:\n", f));
                        for n in neighbors {
                            predictive_context.push_str(&format!("- {}\n", n));
                        }
                    }
                }
            }
        }

        let cacheable_context = format!(
            "SYSTEM: You are Ralph, acting as a {} for an autonomous engineering system (Antigravity).\n\
             PROJECT TYPE: {}\n{}\
             WORKSPACE CONTEXT:\n{}\n\n\
             {}{}{}{}",
             if cortex.config.role == RalphRole::Supervisor { "TECHNICAL SUPERVISOR" } else { "AUTONOMOUS EXECUTOR" },
             fingerprint.specialized_prompt(),
             if !fingerprint.dependency_context().is_empty() { format!("{}\n", fingerprint.dependency_context()) } else { String::new() },
             context,
             if !mcp_context.is_empty() { format!("AVAILABLE MCP TOOLS:\n{}\n\n", mcp_context) } else { String::new() },
             if !heuristics_context.is_empty() { format!("LEARNED HEURISTICS:\n{}\n", heuristics_context) } else { String::new() },
             docs_context,
             if !predictive_context.is_empty() { format!("PREDICTIVE GRAPH CONTEXT:\n{}\n", predictive_context) } else { String::new() }
        );

        let instructions = if cortex.config.role == RalphRole::Supervisor {
            "INSTRUCTIONS (SUPERVISOR MODE):\n\
             1. Your primary goal is to guide Antigravity to solutions with maximum token efficiency.\n\
             2. Analyze the WORKSPACE SYMBOL MAP provided above.\n\
             3. ALWAYS output a structured JSON directive in a code block:\n\
                ```json\n\
                { \"directives\": [\"...\"], \"tasks_to_add\": [\"...\"], \"reasoning\": \"...\" }\n\
                ```\n\
             4. Populate or update `TASKS.md` with a high-level technical checklist.\n\
             5. Do NOT write full file implementations. Use pseudo-code or structural definitions ONLY.\n\
             6. Review the progress of unchecked tasks in `TASKS.md` based on the status reported.\n"
        } else if cortex.config.autonomous_mode {
            "INSTRUCTIONS (HEADLESS MODE):\n\
             1. You are running AUTONOMOUSLY.\n\
             2. REJECT any plan that deletes files outside of `.ralph/shadow` or performs `git push --force`.\n\
             3. Generate code edits or shell commands to solve the next task.\n\
             4. Mark tasks complete by saying 'task is complete'.\n"
        } else {
            "INSTRUCTIONS:\n\
             1. To edit files, use markdown code blocks with the filename in the header or first line comment.\n\
             2. To run commands, use markdown code blocks with 'bash' or 'shell'.\n\
             3. Mark tasks complete by saying 'task is complete'.\n"
        };

        let dynamic_query = format!("USER REQUEST/STATUS: {}\n\n{}", input, instructions);

        // Check semantic cache first
        let full_prompt = format!("{}\n{}", cacheable_context, dynamic_query);
        let cached_response = memory.check_cache(&full_prompt).await.ok().flatten();
        let (response, _from_cache) = if let Some(cached) = cached_response {
            println!("{}", "⚡ Cache Hit! Using stored response...".green());
            (cached, true)
        } else {
            println!("{}", "thinking...".dimmed());
            let images = if let Some(path) = &last_captured_image {
                if let Ok(b64) = Tools::read_capture_as_base64(path) {
                    Some(vec![b64])
                } else {
                    None
                }
            } else {
                None
            };

            let mut stream = cortex
                .stream_generate(&dynamic_query, Some(&cacheable_context), images)
                .await?;
            // Clear used image
            last_captured_image = None;

            let mut resp = String::new();
            while let Some(chunk) = stream.next().await {
                let token = chunk?;
                print!("{}", token);
                io::stdout().flush()?;
                resp.push_str(&token);
            }
            // Store in cache for future use
            let _ = memory.store_cache(&full_prompt, &resp).await;
            (resp, false)
        };

        let mut parser = IncrementalParser::new();
        let shadow = ShadowWorkspace::new();

        // Parse all actions from response (works for both cached and fresh)
        let actions = parser.push(&response);
        for action in &actions {
            match action {
                Action::WriteFile(path, content) => {
                    if cortex.config.role == RalphRole::Executor {
                        println!("\n🛠️ Staging {}...", path.yellow());
                        if let Err(e) = shadow.stage_edit(path, content) {
                            println!("Failed to stage: {}", e);
                            continue;
                        }
                        context_manager.mark_hot(path);
                        let _ = hot_check.trigger();
                    }
                }
                Action::ExecShell(cmd) => {
                    println!("\n🐚 Ready to exec: {}", cmd.cyan());
                }
                Action::Directive(val) => {
                    println!("\n📋 Directive: {:?}", val);
                }
                Action::CallMcpTool(server, tool, args) => {
                    println!(
                        "\n🔌 Calling MCP Tool: {}/{}...",
                        server.cyan(),
                        tool.cyan()
                    );
                    match mcp_manager.call_tool(server, tool, args.clone()).await {
                        Ok(res) => {
                            println!(
                                "✅ MCP Result: {}",
                                serde_json::to_string_pretty(&res)
                                    .unwrap_or_default()
                                    .dimmed()
                            );
                            last_output.push_str(&format!("MCP Tool {} output: {}\n", tool, res));
                        }
                        Err(e) => println!("❌ MCP tool error: {}", e),
                    }
                }
                Action::ReadUrl(url) => {
                    println!("\n🌐 Reading URL: {}...", url.cyan());
                    match Tools::read_url(url).await {
                        Ok(text) => {
                            println!("✅ Read {} chars.", text.len());
                            last_output.push_str(&format!("Content from {}:\n{}\n", url, text));
                        }
                        Err(e) => println!("❌ Web error: {}", e),
                    }
                }
                Action::SearchWeb(query) => {
                    println!("\n🔎 Searching: {}...", query.cyan());
                    match Tools::search_web(query).await {
                        Ok(res) => {
                            println!("✅ Search results received.");
                            last_output
                                .push_str(&format!("Search results for {}:\n{}\n", query, res));
                        }
                        Err(e) => println!("❌ Search error: {}", e),
                    }
                }
                Action::CaptureUI(desc) => {
                    println!("\n📸 Capturing UI...");
                    match Tools::capture_screen(desc.as_deref()) {
                        Ok(path) => {
                            println!("✅ Capture saved: {}", path.cyan());
                            last_output.push_str(&format!("Visual Capture saved to: {}\nAnalyze this screenshot for visual verification.\n", path));
                            last_captured_image = Some(path);
                        }
                        Err(e) => println!("❌ Capture failed: {}", e),
                    }
                }
                Action::CallWasm(func, args) => {
                    println!("\n🧬 Calling WASM Plugin: {}...", func.cyan());
                    // For MVP: Search for .wasm files in .ralph/plugins
                    let plugin_path = format!("{}/plugins/{}.wasm", RALPH_DIR, func);
                    if Path::new(&plugin_path).exists() {
                        if let Ok(wasm_bytes) = fs::read(&plugin_path) {
                            match wasm_runtime.execute(&wasm_bytes, func, args.clone()) {
                                Ok(res) => {
                                    println!("✅ WASM Result: {}", res);
                                    last_output
                                        .push_str(&format!("WASM Tool {} output: {}\n", func, res));
                                }
                                Err(e) => println!("❌ WASM execution error: {}", e),
                            }
                        }
                    } else {
                        println!("❌ WASM Plugin not found: {}", plugin_path);
                    }
                }
                Action::Debate(_) | Action::Snapshot(_) | Action::Restore(_) | Action::Lint(_) => {
                    // These are handled in the final pass after streaming
                }
            }
        }

        println!(
            "\n{}",
            "------------------------------------------------".dimmed()
        );

        // Final Action Pass (for any blocks that were completed at the very end)
        let final_actions = extract_code_blocks(&response);
        for action in final_actions {
            match action {
                Action::WriteFile(path, content) => {
                    // Already staged in stream if block was closed, but we can verify here
                    println!("{}", format!("🔍 Verifying {}...", path).dimmed());
                    match hot_check.poll_results() {
                        Ok(true) => {
                            if let Err(e) = show_diff(&path, &content) {
                                println!("Diff error: {}", e);
                            }
                            let apply = if cortex.config.autonomous_mode {
                                true
                            } else {
                                print!("Context verified. Apply change? [y/N]: ");
                                io::stdout().flush()?;
                                let mut confirm = String::new();
                                io::stdin().read_line(&mut confirm)?;
                                confirm.trim().eq_ignore_ascii_case("y")
                            };
                            if apply {
                                if let Err(e) = shadow.commit(&path) {
                                    println!("Commit error: {}", e);
                                }
                                println!("{}", format!("✅ Applied to {}", path).green());
                            }
                        }
                        Ok(false) => println!("{}", "❌ Verification Failed.".red()),
                        Err(e) => println!("Verification error: {}", e),
                    }
                }
                Action::ExecShell(cmd) => {
                    if cmd.contains("git push") {
                        let _ = Tools::git_sync();
                    } else {
                        let res = Tools::exec_shell(&cmd, cortex.config.autonomous_mode)?;
                        if cortex.config.autonomous_mode {
                            last_output.push_str(&res);
                            last_output.push('\n');
                        }
                    }
                }
                Action::Directive(val) => {
                    if let Some(directives) = val.get("directives").and_then(|v| v.as_array()) {
                        for d in directives {
                            if let Some(s) = d.as_str() {
                                last_output.push_str(&format!("Directive: {}\n", s));
                            }
                        }
                    }
                }
                Action::CallMcpTool(server, tool, _args) => {
                    println!("🏁 Final Pass: MCP Tool {}/{} noted.", server, tool);
                }
                Action::ReadUrl(url) => {
                    println!("🏁 Final Pass: URL {} noted.", url);
                }
                Action::SearchWeb(query) => {
                    println!("🏁 Final Pass: Search '{}' noted.", query);
                }
                Action::CaptureUI(desc) => {
                    println!(
                        "🏁 Final Pass: Capture '{}' noted.",
                        desc.unwrap_or_default()
                    );
                }
                Action::CallWasm(func, _args) => {
                    println!("🏁 Final Pass: WASM Tool '{}' noted.", func);
                }
                Action::Debate(topic) => {
                    println!("🏁 Final Pass: Debate '{}' noted.", topic);
                }
                Action::Snapshot(label) => {
                    match replicator::Replicator::new() {
                        Ok(replicator) => {
                            match replicator.create_snapshot(&label) {
                                Ok(id) => println!("✅ Snapshot created: {}", id.green()),
                                Err(e) => println!("❌ Snapshot failed: {}", e),
                            }
                        }
                        Err(e) => println!("❌ Failed to initialize replicator: {}", e),
                    }
                }
                Action::Restore(id) => {
                    match replicator::Replicator::new() {
                        Ok(replicator) => {
                            match replicator.restore_snapshot(&id) {
                                Ok(_) => println!("✅ System restored to {}", id.green()),
                                Err(e) => println!("❌ Restore failed: {}", e),
                            }
                        }
                        Err(e) => println!("❌ Failed to initialize replicator for restore: {}", e),
                    }
                }
                Action::Lint(path) => {
                    if let Ok(violations) = cortex.perform_lint(&path).await {
                        if violations.is_empty() {
                            println!("✅ No semantic issues found in {}", path.green());
                        } else {
                            println!("\n🚨 {}", "SEMANTIC ISSUES FOUND:".red().bold());
                            for v in &violations {
                                let severity = match v.severity {
                                    lint::LintSeverity::Critical => "CRITICAL".red().bold(),
                                    lint::LintSeverity::Warning => "WARNING".yellow(),
                                    lint::LintSeverity::Suggestion => "SUGGESTION".blue(),
                                };
                                println!("  [{}] {}:{} - {}", severity, v.file, v.line, v.message);
                                if let Some(sugg) = &v.suggestion {
                                    println!("    👉 {}", sugg.dimmed());
                                }
                            }

                            // Auto-save violations to memory context for next turn
                            let report = format!(
                                "Linting report for {}: Found {} issues.",
                                path,
                                violations.len()
                            );
                            let _ = memory.store_lesson(&report).await;
                        }
                    }
                }
            }
        }

        // Handle Debates triggered during streaming or final pass
        // We do this after the loop to avoid nesting async calls in the loop
        let debate_actions: Vec<_> = actions
            .iter()
            .filter_map(|a| {
                if let Action::Debate(t) = a {
                    Some(t.clone())
                } else {
                    None
                }
            })
            .collect();
        for topic in debate_actions {
            if let Ok(synthesis) = cortex.conduct_debate(&topic, &input).await {
                println!("\n📜 {}", "DEBATE VERDICT:".yellow().bold());
                println!("SUMMARY: {}", synthesis.summary);
                println!("RECOMMENDATION: {}", synthesis.final_recommendation.bold());

                if !synthesis.action_items.is_empty() {
                    println!("\nREQUIRED ACTIONS:");
                    for item in synthesis.action_items {
                        println!(
                            "  [{}] {}",
                            if item.priority == 1 {
                                "CRITICAL".red()
                            } else {
                                "SUGGESTED".cyan()
                            },
                            item.description
                        );
                    }
                }

                // If critical, we might want to feed this back into memory or stop execution.
                // For now, we store it as a lesson.
                let lesson = format!(
                    "DEBATE DECISION on '{}': {}",
                    topic, synthesis.final_recommendation
                );
                let _ = memory.store_lesson(&lesson).await;
            }
        }

        if response.to_lowercase().contains("task is complete") || response.contains("[x]") {
            Tools::update_tasks_md()?;
        }

        // Janitor Extraction (Autonomous Learning)
        if cortex.config.autonomous_mode {
            let history = format!("User: {}\nRalph: {}", input, response);
            let janitor_prompt = format!(
                "{}\n\nCONVERSATION:\n{}",
                Janitor::extraction_prompt(),
                history
            );
            if let Ok(mut stream) = cortex.stream_generate(&janitor_prompt, None, None).await {
                let mut extraction = String::new();
                while let Some(chunk) = stream.next().await {
                    if let Ok(token) = chunk {
                        extraction.push_str(&token);
                    }
                }
                let triples = Janitor::parse_triples(&extraction);
                for triple in triples {
                    if let Err(e) = memory.store_lesson(&triple).await {
                        eprintln!("Failed to store lesson: {}", e);
                    } else {
                        println!("✨ Lesson Learned: {}", triple.dimmed());
                    }
                }
            }
        }

        // Reflexion Pass (Self-Critique and Heuristic Learning)
        if cortex.config.autonomous_mode {
            let history = format!("User: {}\nRalph: {}", input, response);
            let reflexion_prompt = format!(
                "{}\n\nCONVERSATION:\n{}",
                reflexion::Reflexion::critique_prompt(),
                history
            );
            if let Ok(mut stream) = cortex.stream_generate(&reflexion_prompt, None, None).await {
                let mut extraction = String::new();
                while let Some(chunk) = stream.next().await {
                    if let Ok(token) = chunk {
                        extraction.push_str(&token);
                    }
                }
                let heuristics = reflexion::Reflexion::parse_heuristics(&extraction);
                for heuristic in heuristics {
                    if let Err(e) = memory.store_heuristic(&heuristic).await {
                        eprintln!("Failed to store heuristic: {}", e);
                    } else {
                        println!("💡 Heuristic Stored: {}", heuristic.dimmed());
                    }
                }
            }
        }

        if cortex.config.autonomous_mode {
            sleep(Duration::from_secs(2)).await;
        }
    }
    Ok(())
}
