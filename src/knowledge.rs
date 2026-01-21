use crate::memory_legacy::Memory;
use anyhow::{anyhow, Result};
use regex::Regex;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use url::Url;

use futures::StreamExt;
use sha2::{Digest, Sha256};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum LibraryType {
    Rust,
    Node,
    Python,
}

pub struct DetectedLibrary {
    pub name: String,
    pub version: String,
    pub lib_type: LibraryType,
}

#[derive(Clone)]
pub struct KnowledgeEngine {
    pub memory: std::sync::Arc<Memory>,
    client: reqwest::Client,
}

impl KnowledgeEngine {
    pub fn new(memory: std::sync::Arc<Memory>) -> Self {
        let client = reqwest::Client::builder()
            .user_agent("Ralph-Nano/0.3.5 (Performance Optimized)")
            .timeout(std::time::Duration::from_secs(30))
            .pool_max_idle_per_host(10)
            .tcp_nodelay(true)
            .build()
            .unwrap_or_default();
        Self { memory, client }
    }

    pub async fn detect_libraries(&self, _path: &Path) -> Result<Vec<DetectedLibrary>> {
        let mut all_libs = Vec::new();
        if let Ok(rust_libs) = self.scan_rust_dependencies() {
            all_libs.extend(rust_libs);
        }
        if let Ok(node_libs) = self.scan_node_dependencies() {
            all_libs.extend(node_libs);
        }
        if let Ok(py_libs) = self.scan_python_dependencies() {
            all_libs.extend(py_libs);
        }
        Ok(all_libs)
    }

    pub fn scan_all_dependencies(&self) -> Result<Vec<DetectedLibrary>> {
        println!("DEBUG: Inside scan_all_dependencies");
        let mut libs = Vec::new();

        // Rust
        if let Ok(rust_libs) = self.scan_rust_dependencies() {
            println!("DEBUG: Scanned Rust - {} libs", rust_libs.len());
            libs.extend(rust_libs);
        }

        // Node
        if let Ok(node_libs) = self.scan_node_dependencies() {
            println!("DEBUG: Scanned Node - {} libs", node_libs.len());
            libs.extend(node_libs);
        }

        // Python
        if let Ok(py_libs) = self.scan_python_dependencies() {
            println!("DEBUG: Scanned Python - {} libs", py_libs.len());
            libs.extend(py_libs);
        }

        libs.sort_by(|a, b| a.name.cmp(&b.name));
        libs.dedup_by(|a, b| a.name == b.name && a.lib_type == b.lib_type);
        Ok(libs)
    }

    pub fn scan_rust_dependencies(&self) -> Result<Vec<DetectedLibrary>> {
        let path = Path::new("Cargo.toml");
        if !path.exists() {
            return Ok(vec![]);
        }
        let content = fs::read_to_string(path)?;
        let value: toml::Value = toml::from_str(&content)?;
        let mut libs = Vec::new();

        let mut process_table = |table: &toml::value::Table| {
            for (name, val) in table {
                let version = match val {
                    toml::Value::String(s) => s.clone(),
                    toml::Value::Table(t) => t
                        .get("version")
                        .and_then(|v| v.as_str())
                        .unwrap_or("latest")
                        .to_string(),
                    _ => "latest".to_string(),
                };
                libs.push(DetectedLibrary {
                    name: name.clone(),
                    version,
                    lib_type: LibraryType::Rust,
                });
            }
        };

        if let Some(deps) = value.get("dependencies").and_then(|d| d.as_table()) {
            process_table(deps);
        }
        if let Some(workspace) = value.get("workspace").and_then(|w| w.as_table()) {
            if let Some(deps) = workspace.get("dependencies").and_then(|d| d.as_table()) {
                process_table(deps);
            }
        }
        Ok(libs)
    }

    pub fn scan_node_dependencies(&self) -> Result<Vec<DetectedLibrary>> {
        let path = Path::new("package.json");
        if !path.exists() {
            return Ok(vec![]);
        }
        let content = fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&content)?;
        let mut libs = Vec::new();

        if let Some(deps) = v.get("dependencies").and_then(|d| d.as_object()) {
            for (name, ver) in deps {
                libs.push(DetectedLibrary {
                    name: name.clone(),
                    version: ver.as_str().unwrap_or("latest").to_string(),
                    lib_type: LibraryType::Node,
                });
            }
        }
        Ok(libs)
    }

    pub fn scan_python_dependencies(&self) -> Result<Vec<DetectedLibrary>> {
        let mut libs = Vec::new();
        // requirements.txt
        let req_path = Path::new("requirements.txt");
        if req_path.exists() {
            if let Ok(content) = fs::read_to_string(req_path) {
                for line in content.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    let parts: Vec<&str> = line.split(&['=', '>', '<', '~'][..]).collect();
                    if !parts[0].trim().is_empty() {
                        libs.push(DetectedLibrary {
                            name: parts[0].trim().to_string(),
                            version: if parts.len() > 1 {
                                parts[1].trim().to_string()
                            } else {
                                "latest".to_string()
                            },
                            lib_type: LibraryType::Python,
                        });
                    }
                }
            }
        }
        // pyproject.toml
        let py_toml = Path::new("pyproject.toml");
        if py_toml.exists() {
            if let Ok(content) = fs::read_to_string(py_toml) {
                if let Ok(value) = toml::from_str::<toml::Value>(&content) {
                    if let Some(deps) = value
                        .get("project")
                        .and_then(|p| p.get("dependencies"))
                        .and_then(|d| d.as_array())
                    {
                        for d in deps {
                            if let Some(s) = d.as_str() {
                                let parts: Vec<&str> =
                                    s.split(&['=', '>', '<', '~', ' '][..]).collect();
                                libs.push(DetectedLibrary {
                                    name: parts[0].trim().to_string(),
                                    version: "latest".to_string(),
                                    lib_type: LibraryType::Python,
                                });
                            }
                        }
                    }
                }
            }
        }
        Ok(libs)
    }

    pub async fn ingest_workspace(&self, path: &Path) -> Result<()> {
        println!("🧠 Mapping workspace: {:?}", path);

        // 1. Scan for Rust documentation
        if let Err(e) = self.ingest_rust_docs_from_path(path).await {
            eprintln!("Rust doc ingestion warning: {}", e);
        }

        // 2. Scan for local files
        let walker = walkdir::WalkDir::new(path)
            .into_iter()
            .filter_entry(|e| {
                let name = e.file_name().to_str().unwrap_or("");
                !name.starts_with('.') && name != "target" && name != "node_modules"
            });

        for entry in walker.filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let _ = self.ingest_file(entry.path()).await;
            }
        }

        Ok(())
    }

    pub async fn ingest_file(&self, path: &Path) -> Result<()> {
        if !path.is_file() {
            return Ok(());
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        // Skip if not code/markdown
        if !matches!(ext, "rs" | "js" | "ts" | "py" | "md" | "txt") {
            return Ok(());
        }

        let content = fs::read_to_string(path)?;

        // 1. Check sync status
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let current_hash = hex::encode(hasher.finalize());
        let path_str = path.to_str().unwrap_or_default();

        if let Ok(Some((_, old_hash))) = self.memory.check_sync_status(path_str).await {
            if old_hash == current_hash {
                return Ok(());
            }
        }

        println!("📝 Re-indexing changed file: {}", path_str);

        // 2. Extract Symbols
        let symbols = self.extract_symbols(&content, ext);
        for mut node in symbols {
            node.path = path_str.to_string();
            if let Err(e) = self.memory.add_node(&node).await {
                eprintln!("Failed to add node {}: {}", node.id, e);
            }
        }

        self.memory.update_sync_status(path_str, &current_hash).await?;
        Ok(())
    }

    fn extract_symbols(&self, content: &str, ext: &str) -> Vec<crate::memory_legacy::GraphNode> {
        use crate::memory_legacy::GraphNode;
        let mut nodes = Vec::new();

        // Very basic regex-based extraction for Godmode v2.4
        // In a real system we'd use tree-sitter or syn/swc
        match ext {
            "rs" => {
                let re_fn = regex::Regex::new(r"pub\s+fn\s+([a-zA-Z0-9_]+)").unwrap();
                let re_struct = regex::Regex::new(r"pub\s+struct\s+([a-zA-Z0-9_]+)").unwrap();

                for cap in re_fn.captures_iter(content) {
                    nodes.push(GraphNode {
                        id: format!("fn:{}", &cap[1]),
                        content: format!("Function definition: {}", &cap[0]),
                        node_type: "fn".to_string(),
                        path: String::new(),
                        edges: Vec::new(),
                    });
                }
                for cap in re_struct.captures_iter(content) {
                    nodes.push(GraphNode {
                        id: format!("struct:{}", &cap[1]),
                        content: format!("Struct definition: {}", &cap[0]),
                        node_type: "struct".to_string(),
                        path: String::new(),
                        edges: Vec::new(),
                    });
                }
            }
            "py" => {
                let re_fn = regex::Regex::new(r"def\s+([a-zA-Z0-9_]+)\(").unwrap();
                for cap in re_fn.captures_iter(content) {
                    nodes.push(GraphNode {
                        id: format!("py_fn:{}", &cap[1]),
                        content: format!("Python function: Def {}", &cap[1]),
                        node_type: "fn".to_string(),
                        path: String::new(),
                        edges: Vec::new(),
                    });
                }
            }
            "md" => {
                nodes.push(GraphNode {
                    id: format!("doc:{}", content.len()),
                    content: content.chars().take(500).collect(),
                    node_type: "doc".to_string(),
                    path: String::new(),
                    edges: Vec::new(),
                });
            }
            _ => {}
        }

        nodes
    }

    pub async fn ingest_rust_docs_from_path(&self, path: &Path) -> Result<()> {
        println!("Scanning path {:?} for Rust documentation...", path);
        // Logic to scan Cargo.toml and trigger ingest_library for each dependency
        if let Ok(libs) = self.detect_libraries(path).await {
            for lib in libs {
                if let Err(e) = self.ingest_library(&lib).await {
                    eprintln!("Failed to ingest {}: {}", lib.name, e);
                }
            }
        }
        Ok(())
    }

    pub async fn ingest_library(&self, lib: &DetectedLibrary) -> Result<()> {
        match lib.lib_type {
            LibraryType::Rust => self.ingest_rust_docs(&lib.name, &lib.version).await,
            LibraryType::Node => self.ingest_npm_docs(&lib.name, &lib.version).await,
            LibraryType::Python => self.ingest_pypi_docs(&lib.name, &lib.version).await,
        }
    }

    async fn ingest_rust_docs(&self, crate_name: &str, version: &str) -> Result<()> {
        let base_url = format!(
            "https://docs.rs/{}/latest/{}/",
            crate_name,
            crate_name.replace('-', "_")
        );
        println!("📚 Auto-learning (Concurrent): {}...", crate_name);

        // 1. Fetch main page carefully
        let res = self.client.get(&base_url).send().await?;
        if !res.status().is_success() {
            return Ok(());
        }
        let html = res.text().await?;

        // 2. Discover modules (Drop doc before await)
        let mut urls_to_process = vec![base_url.clone()];
        {
            let doc = Html::parse_document(&html);
            if let Ok(mod_selector) = Selector::parse(".module-item a.mod, .item-left a.mod") {
                if let Ok(parsed_base_url) = Url::parse(&base_url) {
                    for link in doc.select(&mod_selector) {
                        if let Some(href) = link.value().attr("href") {
                            if let Ok(abs_url) = parsed_base_url.join(href) {
                                urls_to_process.push(abs_url.to_string());
                            }
                        }
                    }
                }
            } else {
                eprintln!("Cannon Warning: Failed to parse module selector for {}", base_url);
            }
        }

        // 3. Concurrent ingestion (limit to 5 URLs total for Nano scale)
        urls_to_process.truncate(5); // Limit to 5 URLs total for Nano scale
        let stream = futures::stream::iter(urls_to_process)
            .map(|url| {
                let client = self.client.clone();
                let _name = crate_name.to_string();
                let _ver = version.to_string();
                async move {
                    let res = client.get(&url).send().await?;
                    if res.status().is_success() {
                        let html = res.text().await?;
                        Ok::<(String, String), anyhow::Error>((url, html))
                    } else {
                        Err(anyhow!("Failed to fetch {}", url))
                    }
                }
            })
            .buffer_unordered(3);

        let results = stream.collect::<Vec<_>>().await;

        for res in results.into_iter().flatten() {
            let (url, html) = res;

            // Phase 5: Differential Sync - compute hash of fetched content
            let mut hasher = Sha256::new();
            hasher.update(html.as_bytes());
            let current_hash = hex::encode(hasher.finalize());

            // Check if already ingested
            if let Ok(Some((_, old_hash))) = self.memory.check_sync_status(&url).await {
                if old_hash == current_hash {
                    continue; // Skip re-indexing
                }
            }

            if self
                .process_and_store_html(
                    &url,
                    &html,
                    crate_name,
                    version,
                    LibraryType::Rust,
                    url == base_url,
                )
                .await
                .is_ok()
            {
                let _ = self.memory.update_sync_status(&url, &current_hash).await;
            }
        }

        Ok(())
    }

    async fn ingest_npm_docs(&self, name: &str, version: &str) -> Result<()> {
        let url = format!("https://www.npmjs.com/package/{}", name);
        println!("📚 Learning JS: {}...", name);
        let res = self.client.get(&url).send().await?;
        if res.status().is_success() {
            let html = res.text().await?;

            let mut hasher = Sha256::new();
            hasher.update(html.as_bytes());
            let current_hash = hex::encode(hasher.finalize());

            if let Ok(Some((_, old_hash))) = self.memory.check_sync_status(&url).await {
                if old_hash == current_hash {
                    return Ok(());
                }
            }

            self.process_and_store_html(&url, &html, name, version, LibraryType::Node, true)
                .await?;
            let _ = self.memory.update_sync_status(&url, &current_hash).await;
        }
        Ok(())
    }

    async fn ingest_pypi_docs(&self, name: &str, version: &str) -> Result<()> {
        let url = format!("https://pypi.org/project/{}/", name);
        println!("📚 Learning Py: {}...", name);
        let res = self.client.get(&url).send().await?;
        if res.status().is_success() {
            let html = res.text().await?;

            let mut hasher = Sha256::new();
            hasher.update(html.as_bytes());
            let current_hash = hex::encode(hasher.finalize());

            if let Ok(Some((_, old_hash))) = self.memory.check_sync_status(&url).await {
                if old_hash == current_hash {
                    return Ok(());
                }
            }

            self.process_and_store_html(&url, &html, name, version, LibraryType::Python, true)
                .await?;
            let _ = self.memory.update_sync_status(&url, &current_hash).await;
        }
        Ok(())
    }

    async fn process_and_store_html(
        &self,
        url: &str,
        html: &str,
        name: &str,
        version: &str,
        lib_type: LibraryType,
        is_main: bool,
    ) -> Result<()> {
        let mut full_text = String::new();
        {
            let doc = Html::parse_document(html);
            let selector_str = match lib_type {
                LibraryType::Rust => "#main-content, .docblock",
                LibraryType::Node => "#readme, .package-description-section",
                LibraryType::Python => "#description, .project-description",
            };
            let selector = match Selector::parse(selector_str) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Selector parse error: {}", e);
                    return Ok(());
                }
            };

            for element in doc.select(&selector) {
                if let Ok(text) = html2text::from_read(element.html().as_bytes(), 80) {
                    full_text.push_str(&text);
                    full_text.push_str("\n\n");
                }
            }

            if full_text.is_empty() {
                full_text = html2text::from_read(html.as_bytes(), 80).unwrap_or_default();
            }
        }

        // Phase 5: Linguistic Pruning - strip excessive comments/boilerplate
        full_text = self.prune_linguistics(&full_text);

        let lang_str = match lib_type {
            LibraryType::Rust => "rust",
            LibraryType::Node => "javascript",
            LibraryType::Python => "python",
        };

        // Semantic Chunking
        let chunks = self.semantic_chunk(&full_text, 1200);

        // Batch Embed
        let embeddings = self.memory.batch_embed(&chunks)?;

        let mut entries = Vec::new();
        for (i, (chunk, embedding)) in chunks.into_iter().zip(embeddings.into_iter()).enumerate() {
            let chunk_type = if chunk.contains("fn ")
                || chunk.contains("class ")
                || chunk.contains("def ")
                || chunk.contains("struct ")
            {
                "definition"
            } else if is_main && i == 0 {
                "overview"
            } else {
                "general"
            };

            let id = format!("doc:{}:{}:{}", name, url.len(), i);
            entries.push((
                id,
                name.to_string(),
                version.to_string(),
                chunk,
                lang_str.to_string(),
                chunk_type.to_string(),
                embedding,
            ));
        }

        // Bulk Insert
        self.memory.batch_add_library_entries(entries).await?;

        Ok(())
    }

    fn semantic_chunk(&self, text: &str, target_size: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current = String::new();

        for block in text.split("\n\n") {
            if current.len() + block.len() > target_size && !current.is_empty() {
                chunks.push(current);
                current = String::new();
            }
            current.push_str(block);
            current.push_str("\n\n");
        }

        if !current.is_empty() {
            chunks.push(current);
        }
        chunks
    }

    pub async fn check_library_updates(&self) -> Result<Vec<(String, String, String)>> {
        let known_libs = self.memory.get_known_libraries_with_versions().await?;
        let mut updates = Vec::new();

        for (lib_name, local_version) in known_libs {
            // Hardcoded check for docs.rs (Rust) for now as MVP
            if let Ok(latest) = self
                .check_upstream_version(&lib_name, LibraryType::Rust)
                .await
            {
                updates.push((lib_name, local_version, latest));
            }
        }
        Ok(updates)
    }

    pub async fn upgrade_manifest(&self, updates: &Vec<(String, String, String)>) -> Result<()> {
        // Upgrade Cargo.toml
        if let Err(e) = self.upgrade_rust_manifest(updates).await {
            eprintln!("Rust upgrade error: {}", e);
        }
        Ok(())
    }

    pub async fn sync_libraries(&self, libs: &[DetectedLibrary]) -> Result<()> {
        for lib in libs {
            if let Err(e) = self.memory.register_library(&lib.name, &lib.version).await {
                eprintln!("Failed to sync library {}: {}", lib.name, e);
            }
        }
        Ok(())
    }

    async fn upgrade_rust_manifest(&self, updates: &Vec<(String, String, String)>) -> Result<()> {
        let path = "Cargo.toml";
        if !Path::new(path).exists() { return Ok(()); }
        let mut content = fs::read_to_string(path)?;
        let mut changed = false;

        for (name, _local, latest) in updates {
            // Pattern 1: name = "version"
            let re1_str = format!(r#"(?m)^(?P<prefix>\s*{}\s*=\s*")[\d\.]+"#, regex::escape(name));
            if let Ok(re1) = Regex::new(&re1_str) {
                if re1.is_match(&content) {
                    content = re1.replace_all(&content, format!("${{prefix}}{}", latest)).to_string();
                    changed = true;
                }
            }

            // Pattern 2: name = { version = "version", ... }
            let re2_str = format!(r#"(?P<prefix>{}\s*=\s*\{{[^}}]*version\s*=\s*")[\d\.]+"#, regex::escape(name));
            if let Ok(re2) = Regex::new(&re2_str) {
                if re2.is_match(&content) {
                    content = re2.replace_all(&content, format!("${{prefix}}{}", latest)).to_string();
                    changed = true;
                }
            }
        }

        if changed {
            fs::write(path, content)?;
        }
        Ok(())
    }

    fn prune_linguistics(&self, text: &str) -> String {
        let mut cleaned = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim();
            // Skip common boilerplate or heavy comment blocks
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with("//!") || trimmed.starts_with("///") || trimmed.starts_with("//")
            {
                // Keep brief comments but skip massive doc-comment blocks for efficiency
                if trimmed.len() > 200 {
                    continue;
                }
            }
            if trimmed.contains("Copyright (c)") || trimmed.contains("Licensed under") {
                continue;
            }
            cleaned.push(line);
        }
        cleaned.join("\n")
    }

    async fn check_upstream_version(&self, name: &str, lib_type: LibraryType) -> Result<String> {
        match lib_type {
            LibraryType::Rust => {
                let url = format!("https://crates.io/api/v1/crates/{}", name);
                let res = self
                    .client
                    .get(&url)
                    .header("User-Agent", "Ralph-Nano-Monitor/0.3.5")
                    .send()
                    .await?;
                if res.status().is_success() {
                    let val: serde_json::Value = res.json().await?;
                    if let Some(v) = val["crate"]["max_version"].as_str() {
                        return Ok(v.to_string());
                    }
                }
            }
            LibraryType::Node => {
                let url = format!("https://registry.npmjs.org/{}/latest", name);
                let res = self.client.get(&url).send().await?;
                if res.status().is_success() {
                    let val: serde_json::Value = res.json().await?;
                    if let Some(v) = val["version"].as_str() {
                        return Ok(v.to_string());
                    }
                }
            }
            LibraryType::Python => {
                let url = format!("https://pypi.org/pypi/{}/json", name);
                let res = self.client.get(&url).send().await?;
                if res.status().is_success() {
                    let val: serde_json::Value = res.json().await?;
                    if let Some(v) = val["info"]["version"].as_str() {
                        return Ok(v.to_string());
                    }
                }
            }
        }
        Err(anyhow!("Version check failed"))
    }
}
