# Roadmap: Ralph-Nano

## 🚩 v0.1.0: The Foundation (✅ Completed)
*   **Core Agent Loop**: Input -> Context -> Prompt -> LLM -> Output.
*   **Context Cannon**: Recursive file reading with exclusion logic.
*   **Embedded Memory**: LanceDB integration with `fastembed`.
*   **Direct API**: Clean connection to Gemini 1.5 Pro.

## 🚩 v0.2.6: The Supervisor Release (✅ Completed)
*   **Bimodal Roles**: Switch between Supervisor (Architect) and Executor (Builder).
*   **Symbol Cannon**: Signature-level codebase awareness for token efficiency.
*   **Structured Directives**: JSON-based coordination protocol.
*   **Task Compression**: Automatic archival of long-running `TASKS.md` histories.
*   **Pro Consolidation**: Uniform use of `gemini-3-pro-preview`.

## 🚩 v0.3.0: Connectivity & Tooling (✅ Completed)
*   **MCP Support**: Full stdio client for Model Context Protocol servers.
*   **Web Toolkit**: Built-in `read_url` and `search_web` capabilities.
*   **Local LLM Fallback**: Triple-tier fallback with Ollama (`llama3.2`) support.

## 🚧 v0.5.0: Relational Intelligence (Next)
*   **Knowledge Graph (Graphiti)**: Move from vector search to a relational graph for cross-file dependency mapping.
*   **Infinite KV Caching**: Implement LLM-side context caching to handle 1M+ token codebases with zero latency.
*   **Haptic Telemetry**: Integration with macOS `say` and system notifications for headless progress tracking.

## 🚩 v1.0.0: The "Pro" Release
*   **TUI Dashboard**: Rich `ratatui` interface showing context hotness, memory hits, and loop status.
*   **Visual Auditor**: Multimodal analysis of UI and diagrams using screenshot capture via `host-scripting`.
*   **Multi-Agent Swarm**: Spawn parallel "Worker Nanos" to handle independent subsystems concurrently.
*   **Self-Evolution**: Dynamic tool generation via compiled `.wasm` plugins for specialized tasks.

## 🔭 Future Horizons
*   **Deep Simulation**: MCP-backed Dockerized testing for 100% safe autonomous refactors.
*   **Enterprise Sync**: Direct integration with Linear/GitHub/Jira via custom MCP adapters.
*   **Predictive Context**: Pre-fetching files based on architectural traversal before the LLM asks.
