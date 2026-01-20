//! Ralph-Nano: High-Performance Local Autonomous Agent
//! 
//! Architecture: Rust + Gemini 1.5 Pro (via API) + LanceDB (Embedded)
//! Philosophy: "Maximum Intelligence, Minimum RAM"
//! Target: Apple Silicon (M-series) optimization

use anyhow::{Context, Result};
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray, Float32Array, Int64Array};
use arrow_schema::{DataType, Field, Schema};
use chrono::{DateTime, Duration, Utc};
use futures::StreamExt;
use lancedb::connect;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration as TokioDuration};
use tracing::{debug, error, info, warn, Level};
use uuid::Uuid;
use walkdir::WalkDir;

// ============================================================================
// CONSTANTS & CONFIGURATION
// ============================================================================

const SYSTEM_RULES: &str = r#"
You are Ralph-Nano, a high-performance autonomous coding agent.

## Core Directives
1. **Precision**: Execute tasks with surgical accuracy.
2. **Safety**: Never execute destructive operations without explicit confirmation.
3. **Efficiency**: Minimize token usage while maximizing output quality.
4. **Memory**: Learn from past interactions and apply lessons consistently.

## Behavioral Rules
- Always analyze the full codebase context before proposing changes.
- Generate complete, production-ready code—no placeholders.
- Validate all shell commands for safety before execution.
- Document your reasoning in the response.

## Output Format
When proposing code changes:
```
[FILE: path/to/file.rs]
<complete file content>
```

When proposing shell commands:
```shell
<command>
```
"#;

const JANITOR_INTERVAL_SECS: u64 = 300; // 5 minutes
const MEMORY_STALE_HOURS: i64 = 24;
const MIN_ACCESS_COUNT_FOR_RETENTION: i64 = 3;
const MAX_SESSION_TURNS: usize = 20;
const LANCEDB_TABLE_NAME: &str = "ralph_memories";

// Directories to skip during codebase traversal
const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    ".next",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".cargo",
    ".rustup",
];

// File extensions to include
const INCLUDE_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "jsx", "tsx", "go", "java", "c", "cpp", "h", "hpp",
    "md", "toml", "yaml", "yml", "json", "sh", "bash", "zsh", "sql", "html", "css",
];

// ============================================================================
// GEMINI API CLIENT
// ============================================================================

#[derive(Clone)]
pub struct GeminiClient {
    api_key: String,
    client: reqwest::Client,
    model: String,
}

#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
}

#[derive(Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Serialize)]
struct GenerationConfig {
    temperature: f32,
    #[serde(rename = "topP")]
    top_p: f32,
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: u32,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    error: Option<GeminiError>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContentResponse,
}

#[derive(Deserialize)]
struct GeminiContentResponse {
    parts: Vec<GeminiPartResponse>,
}

#[derive(Deserialize)]
struct GeminiPartResponse {
    text: String,
}

#[derive(Deserialize)]
struct GeminiError {
    message: String,
}

impl GeminiClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            client: reqwest::Client::new(),
            model: "gemini-1.5-pro".to_string(),
        }
    }

    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let request = GeminiRequest {
            contents: vec![GeminiContent {
                parts: vec![GeminiPart {
                    text: prompt.to_string(),
                }],
            }],
            generation_config: Some(GenerationConfig {
                temperature: 0.7,
                top_p: 0.95,
                max_output_tokens: 8192,
            }),
        };

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Gemini API")?;

        let gemini_response: GeminiResponse = response
            .json()
            .await
            .context("Failed to parse Gemini API response")?;

        if let Some(error) = gemini_response.error {
            anyhow::bail!("Gemini API error: {}", error.message);
        }

        gemini_response
            .candidates
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.content.parts.into_iter().next())
            .map(|p| p.text)
            .ok_or_else(|| anyhow::anyhow!("No response from Gemini API"))
    }
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Represents a stored memory/lesson in LanceDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub content: String,
    pub memory_type: MemoryType,
    pub embedding: Vec<f32>,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: i64,
    pub relevance_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryType {
    Lesson,
    Rule,
    SessionSummary,
    CriticalError,
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryType::Lesson => write!(f, "lesson"),
            MemoryType::Rule => write!(f, "rule"),
            MemoryType::SessionSummary => write!(f, "session_summary"),
            MemoryType::CriticalError => write!(f, "critical_error"),
        }
    }
}

impl From<&str> for MemoryType {
    fn from(s: &str) -> Self {
        match s {
            "lesson" => MemoryType::Lesson,
            "rule" => MemoryType::Rule,
            "session_summary" => MemoryType::SessionSummary,
            "critical_error" => MemoryType::CriticalError,
            _ => MemoryType::Lesson,
        }
    }
}

/// A single turn in the conversation history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub role: String, // "user" or "assistant"
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

/// Shared application state
pub struct AppState {
    pub session_history: VecDeque<ConversationTurn>,
    pub current_codebase_path: PathBuf,
    pub total_tokens_used: u64,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            session_history: VecDeque::with_capacity(MAX_SESSION_TURNS * 2),
            current_codebase_path: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            total_tokens_used: 0,
        }
    }
}

/// Result of the Reflexion critique
#[derive(Debug, Clone)]
pub struct CritiqueResult {
    pub approved: bool,
    pub issues: Vec<String>,
    pub suggestions: Vec<String>,
    pub safety_score: f32,
}

/// Draft plan before execution
#[derive(Debug, Clone)]
pub struct DraftPlan {
    pub description: String,
    pub proposed_actions: Vec<String>,
    pub files_to_modify: Vec<PathBuf>,
    pub shell_commands: Vec<String>,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

// ============================================================================
// VECTOR STORE WRAPPER (LanceDB)
// ============================================================================

pub struct VectorStore {
    db: lancedb::Connection,
    table_name: String,
    embedding_model: fastembed::TextEmbedding,
}

impl VectorStore {
    /// Initialize the vector store with LanceDB
    pub async fn new(db_path: &str) -> Result<Self> {
        info!("Initializing LanceDB at: {}", db_path);
        
        let db = connect(db_path)
            .execute()
            .await
            .context("Failed to connect to LanceDB")?;

        // Initialize the embedding model (all-MiniLM-L6-v2 for efficiency)
        let embedding_model = fastembed::TextEmbedding::try_new(
            fastembed::InitOptions::new(fastembed::EmbeddingModel::AllMiniLML6V2)
                .with_show_download_progress(true),
        )
        .context("Failed to initialize embedding model")?;

        let store = Self {
            db,
            table_name: LANCEDB_TABLE_NAME.to_string(),
            embedding_model,
        };

        // Ensure table exists
        store.ensure_table_exists().await?;

        Ok(store)
    }

    /// Create the memories table if it doesn't exist
    async fn ensure_table_exists(&self) -> Result<()> {
        let tables = self.db.table_names().execute().await?;
        
        if !tables.contains(&self.table_name) {
            info!("Creating memories table: {}", self.table_name);
            
            // Create empty table with schema
            let id_array = StringArray::from(Vec::<String>::new());
            let content_array = StringArray::from(Vec::<String>::new());
            let type_array = StringArray::from(Vec::<String>::new());
            let created_array = StringArray::from(Vec::<String>::new());
            let accessed_array = StringArray::from(Vec::<String>::new());
            let count_array = Int64Array::from(Vec::<i64>::new());
            let score_array = Float32Array::from(Vec::<f32>::new());

            let simple_schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("content", DataType::Utf8, false),
                Field::new("memory_type", DataType::Utf8, false),
                Field::new("created_at", DataType::Utf8, false),
                Field::new("last_accessed", DataType::Utf8, false),
                Field::new("access_count", DataType::Int64, false),
                Field::new("relevance_score", DataType::Float32, false),
            ]));

            let batch = RecordBatch::try_new(
                simple_schema.clone(),
                vec![
                    Arc::new(id_array),
                    Arc::new(content_array),
                    Arc::new(type_array),
                    Arc::new(created_array),
                    Arc::new(accessed_array),
                    Arc::new(count_array),
                    Arc::new(score_array),
                ],
            )?;

            let batches = RecordBatchIterator::new(vec![Ok(batch)], simple_schema);
            self.db.create_table(&self.table_name, Box::new(batches))
                .execute()
                .await?;
        }

        Ok(())
    }

    /// Generate embeddings for text
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embedding_model.embed(vec![text], None)?;
        Ok(embeddings.into_iter().next().unwrap_or_default())
    }

    /// Store a new memory
    pub async fn store_memory(&self, content: &str, memory_type: MemoryType) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let _embedding = self.embed(content)?;

        info!("Storing memory: {} (type: {})", &id[..8], memory_type);

        let table = self.db.open_table(&self.table_name).execute().await?;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("memory_type", DataType::Utf8, false),
            Field::new("created_at", DataType::Utf8, false),
            Field::new("last_accessed", DataType::Utf8, false),
            Field::new("access_count", DataType::Int64, false),
            Field::new("relevance_score", DataType::Float32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![id.clone()])),
                Arc::new(StringArray::from(vec![content.to_string()])),
                Arc::new(StringArray::from(vec![memory_type.to_string()])),
                Arc::new(StringArray::from(vec![now.to_rfc3339()])),
                Arc::new(StringArray::from(vec![now.to_rfc3339()])),
                Arc::new(Int64Array::from(vec![0i64])),
                Arc::new(Float32Array::from(vec![1.0f32])),
            ],
        )?;

        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        table.add(Box::new(batches)).execute().await?;

        Ok(id)
    }

    /// Query memories by semantic similarity
    pub async fn query_memories(&self, query: &str, limit: usize) -> Result<Vec<Memory>> {
        let _query_embedding = self.embed(query)?;
        
        let table = self.db.open_table(&self.table_name).execute().await?;
        
        // Simple query without vector search for now (full implementation requires index)
        let mut results = table
            .query()
            .limit(limit)
            .execute()
            .await?;

        let mut memories = Vec::new();
        
        while let Some(batch_result) = results.next().await {
            let batch = batch_result?;
            
            let id_col = batch.column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let content_col = batch.column_by_name("content")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let type_col = batch.column_by_name("memory_type")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let created_col = batch.column_by_name("created_at")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let accessed_col = batch.column_by_name("last_accessed")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let count_col = batch.column_by_name("access_count")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
            let score_col = batch.column_by_name("relevance_score")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            if let (Some(ids), Some(contents), Some(types), Some(created), Some(accessed), Some(counts), Some(scores)) = 
                (id_col, content_col, type_col, created_col, accessed_col, count_col, score_col) {
                for i in 0..batch.num_rows() {
                    memories.push(Memory {
                        id: ids.value(i).to_string(),
                        content: contents.value(i).to_string(),
                        memory_type: MemoryType::from(types.value(i)),
                        embedding: vec![], // Not loading embeddings for display
                        created_at: DateTime::parse_from_rfc3339(created.value(i))
                            .map(|dt| dt.with_timezone(&Utc))
                            .unwrap_or_else(|_| Utc::now()),
                        last_accessed: DateTime::parse_from_rfc3339(accessed.value(i))
                            .map(|dt| dt.with_timezone(&Utc))
                            .unwrap_or_else(|_| Utc::now()),
                        access_count: counts.value(i),
                        relevance_score: scores.value(i),
                    });
                }
            }
        }

        Ok(memories)
    }

    /// Delete stale memories (older than threshold with low access count)
    pub async fn prune_stale_memories(&self) -> Result<usize> {
        let cutoff = Utc::now() - Duration::hours(MEMORY_STALE_HOURS);
        let cutoff_str = cutoff.to_rfc3339();

        info!("Pruning memories older than {} with access count < {}", 
              cutoff_str, MIN_ACCESS_COUNT_FOR_RETENTION);

        let table = self.db.open_table(&self.table_name).execute().await?;
        
        // LanceDB delete with filter
        let filter = format!(
            "last_accessed < '{}' AND access_count < {}",
            cutoff_str, MIN_ACCESS_COUNT_FOR_RETENTION
        );
        
        table.delete(&filter).await?;
        
        // Return estimated count (LanceDB doesn't return delete count directly)
        Ok(0)
    }

    /// Get total memory count
    pub async fn memory_count(&self) -> Result<usize> {
        let table = self.db.open_table(&self.table_name).execute().await?;
        let count = table.count_rows(None).await?;
        Ok(count)
    }
}

// ============================================================================
// CONTEXT CANNON (Codebase Reader)
// ============================================================================

/// Recursively reads the codebase into a single string
pub fn load_codebase_context(root: &Path) -> Result<String> {
    let mut context = String::with_capacity(1024 * 1024); // Pre-allocate 1MB
    let mut file_count = 0;
    let mut total_lines = 0;

    info!("Loading codebase from: {}", root.display());

    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| !should_skip_entry(e))
    {
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                warn!("Error reading entry: {}", err);
                continue;
            }
        };

        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        
        // Check extension
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if !INCLUDE_EXTENSIONS.contains(&ext) {
                continue;
            }
        } else {
            continue;
        }

        // Read file content
        match std::fs::read_to_string(path) {
            Ok(content) => {
                let relative_path = path.strip_prefix(root).unwrap_or(path);
                let line_count = content.lines().count();
                
                context.push_str(&format!(
                    "\n--- FILE: {} ({} lines) ---\n",
                    relative_path.display(),
                    line_count
                ));
                context.push_str(&content);
                context.push_str("\n--- END FILE ---\n");
                
                file_count += 1;
                total_lines += line_count;
            }
            Err(err) => {
                debug!("Skipping binary/unreadable file {}: {}", path.display(), err);
            }
        }
    }

    info!("Loaded {} files, {} total lines", file_count, total_lines);
    Ok(context)
}

/// Check if a directory entry should be skipped
fn should_skip_entry(entry: &walkdir::DirEntry) -> bool {
    entry.file_name()
        .to_str()
        .map(|name| SKIP_DIRS.contains(&name) || name.starts_with('.'))
        .unwrap_or(false)
}

// ============================================================================
// REFLEXION ENGINE (The Critic)
// ============================================================================

pub struct ReflexionEngine {
    client: GeminiClient,
}

impl ReflexionEngine {
    pub fn new(client: GeminiClient) -> Self {
        Self { client }
    }

    /// Generate a draft plan from the LLM response
    pub fn extract_plan(&self, response: &str) -> DraftPlan {
        let mut files_to_modify = Vec::new();
        let mut shell_commands = Vec::new();
        let mut proposed_actions = Vec::new();

        // Parse file modifications
        for line in response.lines() {
            if line.starts_with("[FILE:") && line.ends_with("]") {
                let path = line
                    .trim_start_matches("[FILE:")
                    .trim_end_matches(']')
                    .trim();
                files_to_modify.push(PathBuf::from(path));
                proposed_actions.push(format!("Modify file: {}", path));
            }
        }

        // Capture shell commands between ```shell and ```
        let mut in_shell_block = false;
        for line in response.lines() {
            if line.starts_with("```shell") || line.starts_with("```bash") {
                in_shell_block = true;
                continue;
            }
            if line.starts_with("```") && in_shell_block {
                in_shell_block = false;
                continue;
            }
            if in_shell_block && !line.is_empty() {
                shell_commands.push(line.to_string());
                proposed_actions.push(format!("Execute: {}", line));
            }
        }

        // Assess risk level
        let risk_level = self.assess_risk(&shell_commands, &files_to_modify);

        DraftPlan {
            description: response.lines().take(3).collect::<Vec<_>>().join(" "),
            proposed_actions,
            files_to_modify,
            shell_commands,
            risk_level,
        }
    }

    /// Assess the risk level of proposed actions
    fn assess_risk(&self, commands: &[String], files: &[PathBuf]) -> RiskLevel {
        let dangerous_commands = ["rm", "sudo", "chmod", "chown", "mv", "kill", "pkill"];
        let sensitive_files = ["Cargo.toml", "package.json", ".env", "config"];

        let mut risk_score = 0;

        for cmd in commands {
            for dangerous in &dangerous_commands {
                if cmd.contains(dangerous) {
                    risk_score += 10;
                }
            }
        }

        for file in files {
            let filename = file.file_name().and_then(|n| n.to_str()).unwrap_or("");
            for sensitive in &sensitive_files {
                if filename.contains(sensitive) {
                    risk_score += 5;
                }
            }
        }

        match risk_score {
            0..=5 => RiskLevel::Low,
            6..=15 => RiskLevel::Medium,
            16..=30 => RiskLevel::High,
            _ => RiskLevel::Critical,
        }
    }

    /// Critique the draft plan for safety and logic errors
    pub async fn critique(&self, plan: &DraftPlan, context: &str) -> Result<CritiqueResult> {
        let critique_prompt = format!(
            r#"You are a senior code reviewer and safety auditor.

## Task
Critically analyze this proposed plan for:
1. **Safety Issues**: Destructive operations, data loss risks, security vulnerabilities
2. **Logic Errors**: Incorrect assumptions, missing edge cases, flawed reasoning
3. **Best Practices**: Code quality, idiomatic patterns, maintainability

## Proposed Plan
Description: {}

Actions:
{}

Files to modify: {:?}
Shell commands: {:?}
Risk Level: {:?}

## Context (truncated)
{}

## Response Format
Respond with a JSON object:
{{
  "approved": true/false,
  "safety_score": 0.0-1.0,
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1", "suggestion2"]
}}

IMPORTANT: Only output the JSON object, nothing else."#,
            plan.description,
            plan.proposed_actions.join("\n"),
            plan.files_to_modify,
            plan.shell_commands,
            plan.risk_level,
            &context[..context.len().min(2000)]
        );

        let response = self.client
            .generate(&critique_prompt)
            .await
            .context("Critique LLM call failed")?;

        // Parse JSON response
        let response_text = response.trim();
        
        // Try to extract JSON from response
        let json_str = if response_text.starts_with('{') {
            response_text
        } else {
            // Find JSON object in response
            response_text
                .find('{')
                .and_then(|start| {
                    response_text.rfind('}').map(|end| &response_text[start..=end])
                })
                .unwrap_or(response_text)
        };

        #[derive(Deserialize)]
        struct CritiqueJson {
            approved: bool,
            safety_score: f32,
            issues: Vec<String>,
            suggestions: Vec<String>,
        }

        match serde_json::from_str::<CritiqueJson>(json_str) {
            Ok(parsed) => Ok(CritiqueResult {
                approved: parsed.approved,
                issues: parsed.issues,
                suggestions: parsed.suggestions,
                safety_score: parsed.safety_score,
            }),
            Err(_) => {
                // Fallback: conservative rejection on parse failure
                warn!("Failed to parse critique response, defaulting to rejection");
                Ok(CritiqueResult {
                    approved: false,
                    issues: vec!["Failed to parse safety critique".to_string()],
                    suggestions: vec!["Review the plan manually".to_string()],
                    safety_score: 0.5,
                })
            }
        }
    }

    /// Run the full reflexion loop
    pub async fn reflexion_loop(
        &self,
        initial_response: &str,
        context: &str,
        max_iterations: usize,
    ) -> Result<(DraftPlan, CritiqueResult)> {
        let mut current_response = initial_response.to_string();
        
        for iteration in 0..max_iterations {
            info!("Reflexion iteration {}/{}", iteration + 1, max_iterations);
            
            let plan = self.extract_plan(&current_response);
            let critique = self.critique(&plan, context).await?;

            if critique.approved {
                info!("Plan approved with safety score: {:.2}", critique.safety_score);
                return Ok((plan, critique));
            }

            if iteration < max_iterations - 1 {
                warn!("Plan rejected. Issues: {:?}", critique.issues);
                
                // Generate refined response
                let refinement_prompt = format!(
                    r#"Your previous plan was rejected by the safety review.

## Issues Found
{}

## Suggestions
{}

## Your Previous Response
{}

## Task
Revise your plan to address all issues. Maintain the same output format."#,
                    critique.issues.join("\n- "),
                    critique.suggestions.join("\n- "),
                    current_response
                );

                current_response = self.client
                    .generate(&refinement_prompt)
                    .await
                    .context("Refinement LLM call failed")?;
            }
        }

        // Return last attempt even if not fully approved
        let final_plan = self.extract_plan(&current_response);
        let final_critique = self.critique(&final_plan, context).await?;
        
        Ok((final_plan, final_critique))
    }
}

// ============================================================================
// THE JANITOR (Background Hygiene Task)
// ============================================================================

pub async fn janitor_task(
    vector_store: Arc<VectorStore>,
    state: Arc<RwLock<AppState>>,
    gemini_client: GeminiClient,
) {
    info!("Janitor task started. Interval: {}s", JANITOR_INTERVAL_SECS);
    
    let mut interval = interval(TokioDuration::from_secs(JANITOR_INTERVAL_SECS));

    loop {
        interval.tick().await;
        info!("Janitor waking up...");

        // Task 1: Prune stale memories
        match vector_store.prune_stale_memories().await {
            Ok(count) => info!("Pruned {} stale memories", count),
            Err(err) => error!("Memory pruning failed: {}", err),
        }

        // Task 2: Summarize long sessions
        let should_summarize = {
            let state = state.read().await;
            state.session_history.len() >= MAX_SESSION_TURNS
        };

        if should_summarize {
            info!("Session history exceeds {} turns, summarizing...", MAX_SESSION_TURNS);
            
            let history_text = {
                let state = state.read().await;
                state.session_history
                    .iter()
                    .map(|turn| format!("[{}]: {}", turn.role, turn.content))
                    .collect::<Vec<_>>()
                    .join("\n\n")
            };

            let summarization_prompt = format!(
                r#"Summarize this coding session into a concise "Lesson" that captures:
1. Key decisions made
2. Problems solved
3. Patterns to remember
4. Mistakes to avoid

Session History:
{}

Output a single paragraph summary (max 200 words)."#,
                history_text
            );

            match gemini_client.generate(&summarization_prompt).await {
                Ok(summary) => {
                    match vector_store.store_memory(&summary, MemoryType::SessionSummary).await {
                        Ok(id) => {
                            info!("Stored session summary: {}", &id[..8]);
                            // Clear RAM history
                            let mut state = state.write().await;
                            state.session_history.clear();
                        }
                        Err(err) => error!("Failed to store session summary: {}", err),
                    }
                }
                Err(err) => error!("Session summarization failed: {}", err),
            }
        }

        // Report stats
        match vector_store.memory_count().await {
            Ok(count) => info!("Current memory count: {}", count),
            Err(err) => error!("Failed to get memory count: {}", err),
        }
    }
}

// ============================================================================
// THE CONTEXT CANNON (Main Loop)
// ============================================================================

pub async fn context_cannon_loop(
    vector_store: Arc<VectorStore>,
    state: Arc<RwLock<AppState>>,
    gemini_client: GeminiClient,
) -> Result<()> {
    let reflexion_engine = ReflexionEngine::new(gemini_client.clone());

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                     RALPH-NANO v0.1.0                        ║");
    println!("║            Maximum Intelligence, Minimum RAM                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Commands:                                                   ║");
    println!("║    /exit    - Exit the agent                                 ║");
    println!("║    /clear   - Clear session history                          ║");
    println!("║    /status  - Show memory and session stats                  ║");
    println!("║    /path    - Set codebase path                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Prompt
        print!("ralph> ");
        stdout.flush()?;

        // Read user input
        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        // Handle commands
        match input {
            "/exit" | "/quit" | "/q" => {
                println!("Goodbye!");
                break;
            }
            "/clear" => {
                let mut state = state.write().await;
                state.session_history.clear();
                println!("Session history cleared.");
                continue;
            }
            "/status" => {
                let state = state.read().await;
                let memory_count = vector_store.memory_count().await.unwrap_or(0);
                println!("Session turns: {}", state.session_history.len());
                println!("Codebase path: {}", state.current_codebase_path.display());
                println!("Memory count: {}", memory_count);
                println!("Tokens used: {}", state.total_tokens_used);
                continue;
            }
            cmd if cmd.starts_with("/path ") => {
                let path = cmd.trim_start_matches("/path ").trim();
                let mut state = state.write().await;
                state.current_codebase_path = PathBuf::from(path);
                println!("Codebase path set to: {}", path);
                continue;
            }
            _ => {}
        }

        // Store user turn
        {
            let mut state = state.write().await;
            state.session_history.push_back(ConversationTurn {
                role: "user".to_string(),
                content: input.to_string(),
                timestamp: Utc::now(),
            });
        }

        println!("\n⏳ Loading context...");

        // Step 1: Load codebase context
        let codebase_path = {
            let state = state.read().await;
            state.current_codebase_path.clone()
        };
        
        let codebase_context = match load_codebase_context(&codebase_path) {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("Failed to load codebase: {}", err);
                String::new()
            }
        };

        // Step 2: Retrieve relevant memories
        let memories = match vector_store.query_memories(input, 5).await {
            Ok(mems) => mems,
            Err(err) => {
                warn!("Memory retrieval failed: {}", err);
                vec![]
            }
        };

        let memories_context = if memories.is_empty() {
            String::new()
        } else {
            let mem_texts: Vec<String> = memories
                .iter()
                .map(|m| format!("[{}] {}", m.memory_type, m.content))
                .collect();
            format!("\n## Retrieved Lessons\n{}\n", mem_texts.join("\n\n"))
        };

        // Step 3: Construct the mega-prompt
        let full_prompt = format!(
            r#"{}

{}

## Current Codebase
{}

## User Request
{}"#,
            SYSTEM_RULES,
            memories_context,
            &codebase_context[..codebase_context.len().min(100_000)], // Cap at 100k chars
            input
        );

        println!("⏳ Generating response...");

        // Step 4: Generate initial response
        let initial_response = match gemini_client.generate(&full_prompt).await {
            Ok(resp) => resp,
            Err(err) => {
                eprintln!("LLM call failed: {}", err);
                continue;
            }
        };

        // Step 5: Run Reflexion loop
        println!("⏳ Running safety critique...");
        
        let (plan, critique) = match reflexion_engine
            .reflexion_loop(&initial_response, &codebase_context, 3)
            .await
        {
            Ok(result) => result,
            Err(err) => {
                eprintln!("Reflexion failed: {}", err);
                // Fall back to showing unvalidated response
                println!("\n📝 Response (unvalidated):\n{}\n", initial_response);
                continue;
            }
        };

        // Step 6: Present results
        if critique.approved {
            println!("\n✅ Plan approved (safety: {:.0}%)", critique.safety_score * 100.0);
            println!("\n📝 Response:\n{}\n", initial_response);
            
            if !plan.shell_commands.is_empty() {
                println!("⚠️  Proposed shell commands:");
                for cmd in &plan.shell_commands {
                    println!("    $ {}", cmd);
                }
                print!("\nExecute these commands? [y/N]: ");
                stdout.flush()?;
                
                let mut confirm = String::new();
                stdin.lock().read_line(&mut confirm)?;
                
                if confirm.trim().to_lowercase() == "y" {
                    for cmd in &plan.shell_commands {
                        println!("Executing: {}", cmd);
                        let output = std::process::Command::new("sh")
                            .arg("-c")
                            .arg(cmd)
                            .output();
                        
                        match output {
                            Ok(out) => {
                                if !out.stdout.is_empty() {
                                    println!("{}", String::from_utf8_lossy(&out.stdout));
                                }
                                if !out.stderr.is_empty() {
                                    eprintln!("{}", String::from_utf8_lossy(&out.stderr));
                                }
                            }
                            Err(err) => eprintln!("Command failed: {}", err),
                        }
                    }
                }
            }
        } else {
            println!("\n⚠️  Plan requires review (safety: {:.0}%)", critique.safety_score * 100.0);
            println!("\nIssues found:");
            for issue in &critique.issues {
                println!("  - {}", issue);
            }
            println!("\nSuggestions:");
            for suggestion in &critique.suggestions {
                println!("  - {}", suggestion);
            }
            println!("\n📝 Response (requires manual review):\n{}\n", initial_response);
        }

        // Store assistant turn
        {
            let mut state = state.write().await;
            state.session_history.push_back(ConversationTurn {
                role: "assistant".to_string(),
                content: initial_response.clone(),
                timestamp: Utc::now(),
            });
        }
    }

    Ok(())
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .compact()
        .init();

    info!("Starting Ralph-Nano v0.1.0");

    // Load environment variables
    dotenvy::dotenv().ok();
    
    let api_key = std::env::var("GEMINI_API_KEY")
        .context("GEMINI_API_KEY environment variable not set")?;

    // Initialize Gemini client
    let gemini_client = GeminiClient::new(&api_key);
    info!("Gemini client initialized");

    // Initialize vector store
    let db_path = std::env::var("LANCEDB_PATH")
        .unwrap_or_else(|_| ".ralph-nano/lancedb".to_string());
    
    let vector_store = Arc::new(
        VectorStore::new(&db_path)
            .await
            .context("Failed to initialize vector store")?
    );
    info!("Vector store initialized at: {}", db_path);

    // Initialize shared state
    let state = Arc::new(RwLock::new(AppState::default()));

    // Spawn the Janitor background task
    let janitor_vs = Arc::clone(&vector_store);
    let janitor_state = Arc::clone(&state);
    let janitor_client = GeminiClient::new(&api_key);
    
    tokio::spawn(async move {
        janitor_task(janitor_vs, janitor_state, janitor_client).await;
    });
    info!("Janitor task spawned");

    // Run the main Context Cannon loop
    context_cannon_loop(
        Arc::clone(&vector_store),
        Arc::clone(&state),
        gemini_client,
    ).await?;

    info!("Ralph-Nano shutting down");
    Ok(())
}
