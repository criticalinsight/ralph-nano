use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use std::sync::Arc;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: Value,
    pub id: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    pub result: Option<Value>,
    pub error: Option<Value>,
    pub id: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

pub struct McpClient {
    _child: Child,
    stdin: Arc<Mutex<tokio::process::ChildStdin>>,
    pending_requests: Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<McpResponse>>>>,
    next_id: Arc<Mutex<u64>>,
}

impl McpClient {
    pub async fn new(command: &str, args: &[&str]) -> Result<Self> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to spawn MCP server")?;

        let stdin = child.stdin.take().context("No stdin")?;
        let stdout = child.stdout.take().context("No stdout")?;
        
        let client = Self {
            _child: child,
            stdin: Arc::new(Mutex::new(stdin)),
            pending_requests: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(1)),
        };

        let pending_requests = Arc::clone(&client.pending_requests);
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if let Ok(res) = serde_json::from_str::<McpResponse>(&line) {
                    if let Some(id) = res.id {
                        let mut pending = pending_requests.lock().await;
                        if let Some(tx) = pending.remove(&id) {
                            let _ = tx.send(res);
                        }
                    }
                }
            }
        });

        // Initialize connection
        client.call("initialize", json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "Ralph-Nano", "version": "0.3.0" }
        })).await?;

        Ok(client)
    }

    pub async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let id = {
            let mut id_gen = self.next_id.lock().await;
            let id = *id_gen;
            *id_gen += 1;
            id
        };

        let req = McpRequest {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
            id,
        };

        let (tx, rx) = tokio::sync::oneshot::channel();
        {
            let mut pending = self.pending_requests.lock().await;
            pending.insert(id, tx);
        }

        let req_json = serde_json::to_string(&req)? + "\n";
        {
            let mut stdin = self.stdin.lock().await;
            stdin.write_all(req_json.as_bytes()).await?;
            stdin.flush().await?;
        }

        let res = rx.await.context("MCP response channel closed")?;
        if let Some(err) = res.error {
            return Err(anyhow::anyhow!("MCP Error: {}", err));
        }
        
        Ok(res.result.unwrap_or(Value::Null))
    }

    pub async fn list_tools(&self) -> Result<Vec<McpTool>> {
        let res = self.call("tools/list", json!({})).await?;
        let tools: Vec<McpTool> = serde_json::from_value(res["tools"].clone())?;
        Ok(tools)
    }

    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value> {
        self.call("tools/call", json!({
            "name": name,
            "arguments": arguments
        })).await
    }
}

pub struct McpManager {
    clients: Vec<(String, McpClient)>,
}

impl McpManager {
    pub fn new() -> Self {
        Self { clients: Vec::new() }
    }

    pub async fn add_server(&mut self, name: &str, command: &str, args: &[&str]) -> Result<()> {
        let client = McpClient::new(command, args).await?;
        self.clients.push((name.to_string(), client));
        Ok(())
    }

    pub async fn get_all_tools(&self) -> Result<Vec<(String, Vec<McpTool>)>> {
        let mut all_tools = Vec::new();
        for (name, client) in &self.clients {
            let tools = client.list_tools().await?;
            all_tools.push((name.clone(), tools));
        }
        Ok(all_tools)
    }

    pub async fn call_tool(&self, server_name: &str, tool_name: &str, arguments: Value) -> Result<Value> {
        for (name, client) in &self.clients {
            if name == server_name {
                return client.call_tool(tool_name, arguments).await;
            }
        }
        Err(anyhow::anyhow!("Server {} not found", server_name))
    }
}
