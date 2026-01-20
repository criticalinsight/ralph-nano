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
- **Primary Brain**: Gemini 1.5 Pro (via direct API).
- **Context Window**: Leverage Gemini's 2M token window for "Context Cannon" mode.
- **Feedback Loop**: "Reflexion Engine" that critiques plans before execution.

### 4.3 Memory System
- **Vector Store**: Embedded LanceDB (no external database server).
- **Embeddings**: `fastembed-rs` (All-MiniLM-L6-v2) running locally on CPU/Metal.
- **Background Hygiene**: "Janitor" task to prune stale context and summarize lessons.

### 4.4 Inputs & Outputs
- **Input**: CLI (Readline) or Arguments.
- **Context**: Recursive file scanning (ignoring `.git`, `node_modules`).
- **Output**: Terminal streaming, file modifications, shell command execution.

## 5. Functional Specifications

### 5.1 The Context Cannon
- **Goal**: Instantly load the relevant parts (or all) of a codebase into the LLM's context.
- **Mechanism**: Parallel directory walking, intelligent file filtering, token estimation.

### 5.2 The Reflexion Engine (Safe Speculation)
- **Goal**: Prevent hallucinations and dangerous commands through sandboxed validation.
- **Mechanism**:
  1. **Shadow Workspace**: Edits are first applied to a `.ralph/shadow` directory.
  2. **Cargo Check**: In Rust projects, `cargo check` is run against the shadow workspace to verify compilation.
  3. **Universal Mode**: In non-Rust projects, compilation checks are gracefully skipped while maintaining the path-based sandboxing.
  4. **Interactive Diff**: Users see colored diffs before changes are committed to the real workspace.

### 5.3 Persistent Autonomy & Supervision (v0.2.6)
- **Goal**: Maintain continuous oversight and guide external executors with maximum token efficiency.
- **Mechanism**:
  - **Single Pro Model**: Uses `gemini-3-pro-preview` exclusively for all reasoning.
  - **The Symbol Cannon**: Uses `scan_symbols` to provide high-level architectural context without implementational bloat.
  - **Structured Directives**: Mandates JSON output for clear coordination with executors.
  - **Task Compression**: Automatically archives completed tasks in `TASKS.md` after 5 completions.

### 5.4 The Janitor
- **Goal**: Keep the agent's memory clean and efficient over long sessions.
- **Mechanism**:
  - Background `tokio` task running every 5 minutes.
  - Prunes memories > 24h old with low access counts.
  - Summarizes sessions > 20 turns into "Lesson" vectors.

## 6. Success Metrics
- **Startup Time**: < 100ms.
- **Context Loading**: < 1s for 10MB codebase.
- **Memory Footprint**: < 100MB (idle).
- **Safety**: 0 accidental destructive commands executed in test suite thanks to Shadow Workspace.

## 7. Future Considerations
- **Local LLM Support**: Integration with `llama.cpp` for fully offline operation.
- **MCP Integration**: Full client support for Model Context Protocol servers.
- **TUI**: A rich terminal user interface using `ratatui`.
