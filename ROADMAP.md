# Roadmap: Ralph-Nano

## 🚩 v0.1.0: The Foundation (✅ Completed)
*   **Core Agent Loop**: Input -> Context -> Prompt -> LLM -> Output.
*   **Context Cannon**: Recursive file reading with exclusion logic.
*   **Embedded Memory**: CozoDB (RocksDB) integration with `candle`.
*   **Direct API**: Clean connection to Gemini 1.5 Pro.

## 🚩 v0.2.4: The Supervisor Release (✅ Completed)
*   **Bimodal Roles**: Switch between Supervisor (Architect) and Executor (Builder).
*   **Symbol Cannon**: Signature-level codebase awareness for token efficiency.
*   **Structured Directives**: JSON-based coordination protocol.
*   **Task Compression**: Automatic archival of long-running `TASKS.md` histories.

## 🚩 v0.3.5: The Auto-Didact Release (✅ Completed)
*   **Workspace Awareness**: Boot-time detection of `Cargo.toml`, `package.json`, and Python manifests.
*   **Recursive Self-Teaching**: Concurrent scraping of official documentation from `docs.rs`, `npmjs`, and `pypi`.
*   **Weighted Technical RAG**: Boosted vector search prioritizing code definitions over general text.
*   **Parallel Scaling**: High-performance ingestion using `rayon` and `futures` for 4x faster learning.
*   **MCP Support**: Full stdio client for Model Context Protocol servers.

## 🚩 v0.5.0: Relational Intelligence (✅ Completed)
*   **Knowledge Graph**: Relational-Graph-Vector hybrid mapping for cross-file dependency logic.
*   **Infinite KV Caching**: Leveraging Gemini context caching to handle 1M+ token codebases with zero latency.
*   **Haptic Telemetry**: Integration with macOS `say` and system notifications for headless progress tracking.
*   **Registry Monitoring**: Proactive notifications when ingested documentation libraries have upstream updates.

## 🚩 v1.0.0: The "Pro" Release (✅ Completed)
*   **Predictive Context**: Pre-fetching files based on architectural traversal before the LLM asks.
*   **Multi-Agent Swarm**: Spawn parallel "Worker Nanos" to handle independent subsystems concurrently.
*   **Self-Evolution**: Dynamic tool generation via compiled `.wasm` plugins for specialized tasks.
*   **Visual Auditor**: Multimodal analysis of UI and diagrams using screenshot capture via `host-scripting`.

## � v1.1.0: Context Hyper-Efficiency (✅ Completed)
*   **Differential Symbol Sync**: Only indexing changed lines in the Knowledge Graph for sub-100ms real-time workspace awareness.
*   **Compressed Semantic Hashes**: Representing entire modules in the prompt using ultra-compact architectural "fingerprints".
*   **Linguistic Pruning**: AI-guided stripping of redundant comments and boilerplate from context to maximize reasoning density.
*   **Local RAG Re-ranking**: Using the local `candle` engine to prune the Top-50 vector hits down to the Top-10 critical symbols.

## � v1.2.0: Verified Autonomy (✅ Completed)
*   **Self-Correcting "Debate" Loops**: Multi-persona LLM sessions to audit complex logic changes.
*   **Rollback Snapshots**: Zero-cost restoration of workspace state via `.ralph/snapshots` directory.
*   **Semantic Linting**: LLM-guided linting for catching logical API misuse that standard compilers miss.
*   **Multimodal Vision**: Screenshot capture passed to LLM for visual UI verification.

---
*Last Updated: January 20, 2026*
