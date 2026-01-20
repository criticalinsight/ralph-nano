//! Natural Language API Server
//!
//! Exposes Ralph's tools as a local HTTP/WebSocket API for external
//! apps and scripts to invoke agent capabilities.

use anyhow::Result;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};

#[allow(dead_code)]
/// API Server state
pub struct ApiState {
    pub workspace: String,
    pub is_processing: bool,
    pub last_response: Option<String>,
}

impl ApiState {
    #[allow(dead_code)]
    pub fn new(workspace: String) -> Self {
        Self {
            workspace,
            is_processing: false,
            last_response: None,
        }
    }
}

#[allow(dead_code)]
pub type SharedState = Arc<RwLock<ApiState>>;

// --- Request/Response Types ---

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct AskRequest {
    pub question: String,
    #[serde(default)]
    pub context: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub struct AskResponse {
    pub answer: String,
    pub sources: Vec<SourceReference>,
    pub confidence: f32,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub struct SourceReference {
    pub file: String,
    pub line_start: usize,
    pub line_end: usize,
    pub snippet: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ExecuteRequest {
    pub action: String,
    pub params: serde_json::Value,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub struct ExecuteResponse {
    pub success: bool,
    pub result: serde_json::Value,
    pub error: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub status: String,
    pub workspace: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub memory_entries: usize,
    pub active_workers: usize,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub file_types: Option<Vec<String>>,
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub file: String,
    pub line: usize,
    pub content: String,
    pub score: f32,
}

// --- Handler Functions ---

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "ralph-nano",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

async fn get_status(State(state): State<SharedState>) -> impl IntoResponse {
    let state = state.read().await;
    Json(StatusResponse {
        status: if state.is_processing {
            "processing"
        } else {
            "idle"
        }
        .to_string(),
        workspace: state.workspace.clone(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: 0, // Would track actual uptime
        memory_entries: 0, // Would query memory
        active_workers: 0, // Would query swarm
    })
}

async fn ask_question(
    State(state): State<SharedState>,
    Json(request): Json<AskRequest>,
) -> impl IntoResponse {
    // Mark as processing
    {
        let mut s = state.write().await;
        s.is_processing = true;
    }

    // TODO: Integrate with actual Memory and Cortex
    // This is a placeholder that would:
    // 1. Search knowledge graph for relevant context
    // 2. Build prompt with context
    // 3. Query LLM for answer
    // 4. Extract source references

    let response = AskResponse {
        answer: format!("Processing question: {}", request.question),
        sources: vec![],
        confidence: 0.0,
    };

    // Mark as done
    {
        let mut s = state.write().await;
        s.is_processing = false;
        s.last_response = Some(response.answer.clone());
    }

    Json(response)
}

async fn execute_action(
    State(state): State<SharedState>,
    Json(request): Json<ExecuteRequest>,
) -> impl IntoResponse {
    // TODO: Map action to actual tool execution
    // This would integrate with the Tools struct

    let result = match request.action.as_str() {
        "read_file" => {
            let path = request.params["path"].as_str().unwrap_or("");
            match std::fs::read_to_string(path) {
                Ok(content) => ExecuteResponse {
                    success: true,
                    result: serde_json::json!({ "content": content }),
                    error: None,
                },
                Err(e) => ExecuteResponse {
                    success: false,
                    result: serde_json::Value::Null,
                    error: Some(e.to_string()),
                },
            }
        }
        "list_files" => {
            let state = state.read().await;
            let path = request.params["path"].as_str().unwrap_or(&state.workspace);
            match std::fs::read_dir(path) {
                Ok(entries) => {
                    let files: Vec<String> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.path().to_string_lossy().to_string())
                        .collect();
                    ExecuteResponse {
                        success: true,
                        result: serde_json::json!({ "files": files }),
                        error: None,
                    }
                }
                Err(e) => ExecuteResponse {
                    success: false,
                    result: serde_json::Value::Null,
                    error: Some(e.to_string()),
                },
            }
        }
        "search_code" => {
            // Placeholder for code search
            ExecuteResponse {
                success: true,
                result: serde_json::json!({ "results": [] }),
                error: None,
            }
        }
        _ => ExecuteResponse {
            success: false,
            result: serde_json::Value::Null,
            error: Some(format!("Unknown action: {}", request.action)),
        },
    };

    Json(result)
}

async fn search_codebase(
    State(_state): State<SharedState>,
    Json(_request): Json<SearchRequest>,
) -> impl IntoResponse {
    // TODO: Integrate with Memory for semantic search
    // This would use the knowledge graph

    Json(SearchResponse {
        results: vec![],
        total: 0,
    })
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<SharedState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_websocket(socket, state))
}

async fn handle_websocket(mut socket: WebSocket, _state: SharedState) {
    // Send welcome message
    let welcome_msg = serde_json::to_string(&serde_json::json!({
        "type": "connected",
        "message": "Ralph-Nano WebSocket API"
    }))
    .unwrap();
    let _ = socket.send(Message::Text(welcome_msg.into())).await;

    // Handle incoming messages
    while let Some(msg) = socket.recv().await {
        if let Ok(Message::Text(text)) = msg {
            // Parse message as JSON command
            if let Ok(cmd) = serde_json::from_str::<serde_json::Value>(&text) {
                let cmd_type = cmd["type"].as_str().unwrap_or("");

                let response = match cmd_type {
                    "ask" => {
                        let question = cmd["question"].as_str().unwrap_or("");
                        serde_json::json!({
                            "type": "answer",
                            "answer": format!("Processing: {}", question)
                        })
                    }
                    "subscribe" => {
                        serde_json::json!({
                            "type": "subscribed",
                            "channel": cmd["channel"].as_str().unwrap_or("events")
                        })
                    }
                    "ping" => {
                        serde_json::json!({ "type": "pong" })
                    }
                    _ => {
                        serde_json::json!({
                            "type": "error",
                            "message": format!("Unknown command: {}", cmd_type)
                        })
                    }
                };

                let resp_str = serde_json::to_string(&response).unwrap();
                let _ = socket.send(Message::Text(resp_str.into())).await;
            }
        }
    }
}

#[allow(dead_code)]
/// Build the API router
pub fn build_router(state: SharedState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health_check))
        .route("/status", get(get_status))
        .route("/ask", post(ask_question))
        .route("/execute", post(execute_action))
        .route("/search", post(search_codebase))
        .route("/ws", get(websocket_handler))
        .layer(cors)
        .with_state(state)
}

#[allow(dead_code)]
/// Start the API server
pub async fn start_server(workspace: String, port: u16) -> Result<()> {
    let state = Arc::new(RwLock::new(ApiState::new(workspace)));
    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port)).await?;
    println!(
        "🌐 Ralph-Nano API server listening on http://127.0.0.1:{}",
        port
    );

    axum::serve(listener, app).await?;
    Ok(())
}
