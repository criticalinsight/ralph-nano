# Product Requirements Document (PRD): Ralph-Nano

## 1. Executive Summary
**Ralph-Nano** is a high-performance, single-binary autonomous coding agent designed for **Apple Silicon** hardware. 
**Philosophy**: "Maximum Intelligence, Minimum RAM."
Unlike bulky Python-based agents or Electron apps, Ralph-Nano provides a native, embedded, and highly efficient agent experience that integrates directly with the user's local environment.

## 2. Problem Statement
Current autonomous agents are often:
- **Resource Heavy**: Consuming GBs of RAM just to idle (Python/Electron/Docker overhead).
- **Slow Context Loading**: Struggling to traverse large codebases quickly.
- **Complex Deployment**: Requiring Docker, Python venvs, and multiple services.
- **Unsafe**: Executing code without sufficient critique or validation loops.
- **Episodic**: Exiting after one task, breaking flow for multi-step projects.

## 3. Product Vision
To build the "vim of agents"—lightweight, incredibly fast, and powerful. It should feel like a native Unix tool that enhances the developer's workflow without taking it over. It should be capable of **Persistent Autonomy**, waiting for and reacting to tasks as they appear.

## 4. Technical Requirements

### 4.1 Core Architecture
- **Language**: Rust (for memory safety, concurrency, and speed).
- **Distribution**: Single binary (no external runtime dependencies like Python or Node).
- **Platform**: Optimized for macOS (Apple Silicon).

### 4.2 AI & Reasoning
- **Primary Brain**: **Gemini 3.0 Flash-Preview** (Primary) with **Gemini 2.5 Flash** (Fallback).
  - Features: **Thinking Levels** (`High`/`Low`/`Auto`) for variable reasoning depth.
- **Context Window**: Leverage Gemini's large token window for "Context Cannon" mode.
- **Feedback Loop**: "Reflexion Engine" that critiques plans before execution.

### 4.3 Memory System
- **Vector Store**: Embedded CozoDB (Relational-Graph-Vector hybrid, RocksDB storage).
- **Embeddings**: `candle` (BERT) running locally on CPU/Metal.
- **Background Hygiene**: "Janitor" task to prune stale context and summarize lessons.

### 4.4 Inputs & Outputs
- **Input**: CLI (Readline) or Arguments.
- **Context**: Recursive file scanning (ignoring `.git`, `node_modules`).
- **Output**: Terminal streaming, file modifications, shell command execution.

## 5. Functional Specifications

### 5.1 The Context Cannon
- **Goal**: Instantly load the relevant parts (or all) of a codebase into the LLM's context.
- **Mechanism**: Parallel directory walking, intelligent file filtering, token estimation.

### 5.2 The Crucible (Safe Speculation)
- **Goal**: Prevent hallucinations and dangerous commands through sandboxed validation.
- **Mechanism**:
  1. **OverlayFS**: Edits are first applied to a transactional `OverlayFS` safety shield.
  2. **Cargo Check**: In Rust projects, `cargo check` is run against the overlay to verify compilation.
  3. **Universal Mode**: In non-Rust projects, compilation checks are gracefully skipped while maintaining the path-based sandboxing.
  3. **Path Sanitization**: Helper `is_safe_path` ensures all file operations remain within the workspace.
  4. **Interactive Diff**: Users see colored diffs before changes are committed to the real workspace.

### 5.3 Persistent Autonomy & Supervision (v0.2.4)
- **Goal**: Maintain continuous oversight and guide external executors with maximum token efficiency.
- **Mechanism**:
  - **Model**: Uses `gemini-3-flash-preview` with variable thinking.
  - **The Symbol Cannon**: Uses `scan_symbols` to provide high-level architectural context without implementational bloat.
  - **Structured Directives**: Mandates JSON output for clear coordination with executors.
  - **Task Compression**: Automatically archives completed tasks in `TASKS.md` after 5 completions.

### 5.4 The Auto-Didact Engine (Workspace Awareness)
- **Goal**: Automatically learn the project's tech stack and ingest official documentation to prevent LLM hallucination.
- **Mechanism**:
  - **Manifest Scanning**: Automatically detects dependencies in `Cargo.toml`, `package.json`, `requirements.txt`, and `pyproject.toml`.
  - **Recursive Ingestion**: Scrapes `docs.rs`, `npmjs.com`, and `pypi.org`, following module links for deep coverage.
  - **Semantic Chunking**: Splits documentation into logical blocks (definitions, overviews, examples).
  - **Weighted RAG**: Boosts the relevance of "definition" chunks during vector search for higher technical accuracy.

### 5.5 The Janitor
- **Goal**: Keep the agent's memory clean and efficient over long sessions.
- **Mechanism**:
  - Background `tokio` task running every 5 minutes.
  - Prunes memories > 24h old with low access counts.
  - Summarizes sessions > 20 turns into "Lesson" vectors.

### 5.6 Self-Evolution (WASM Plugins)
- **Goal**: Allow Ralph to extend its own toolset by generating and executing high-performance utility code.
- **Mechanism**:
  - **Secure Sandbox**: Uses `wasmtime` for isolated, cross-platform execution.
  - **Dynamic Compilation**: Ralph can generate Rust code, compile it to WASM, and call it as a native tool.
  - **Isolation**: Plugins are restricted to the `.ralph/plugins` environment.

### 5.7 Multi-Agent Swarm (Concurrency)
- **Goal**: Drastically speed up large-scale refactors by parallelizing independent subtasks.
- **Mechanism**:
  - **Worker Nanos**: Specialized, isolated Ralph instances spawned as background processes.
  - **Task Delegation**: The Supervisor partitions a complex objective into `task.json` files for workers.
  - **Status Reporting**: Workers provide real-time updates via a shared status registry.

### 5.8 Predictive Context (Active RAG)
- **Goal**: Proactively load related code symbols before the LLM explicitly requests them.
- **Mechanism**:
  - **Graph Neighborhood**: Detects "hot" files and traverses the Knowledge Graph for their immediate neighbors.
  - **Atomic Injection**: Injects related struct/function definitions into the prompt as `[PREDICTIVE_CONTEXT]`.

## 6. Success Metrics
- **Startup Time**: < 100ms.
- **Context Loading**: < 1s for 10MB codebase.
- **Memory Footprint**: < 100MB (idle).
- **Safety**: 0 accidental destructive commands executed in test suite thanks to OverlayFS.

## 7. Future Considerations (M2 Pro Performance)
- **Metal-Accelerated RAG**: Offloading context filtering to the GPU for sub-50ms retrieval.
- **Speculative Shadow-Verification**: Exploiting MacBook's multi-core performance to run parallel build/test checks.
- **macOS Native Integration**: Menu bar shortcuts and system-level hotkeys for high-speed interaction.
- **One-Click Rollbacks**: Differential state management for instant, zero-cost workspace restoration.
