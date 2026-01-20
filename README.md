# Ralph-Nano 🚀

> **Maximum Intelligence, Minimum RAM**

A high-performance, single-binary autonomous coding agent optimized for Apple Silicon (M-series).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RALPH-NANO                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   MAIN LOOP (Context Cannon)              │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │   │
│  │  │  Codebase   │  │   Memory     │  │   Mega-Prompt   │  │   │
│  │  │   Reader    │──│  Retrieval   │──│   Constructor   │  │   │
│  │  └─────────────┘  └──────────────┘  └─────────────────┘  │   │
│  │         │                │                  │             │   │
│  │         └────────────────┴──────────────────┘             │   │
│  │                          │                                │   │
│  │                          ▼                                │   │
│  │  ┌──────────────────────────────────────────────────┐    │   │
│  │  │              REFLEXION ENGINE                     │    │   │
│  │  │  ┌─────────┐    ┌──────────┐    ┌────────────┐   │    │   │
│  │  │  │  Draft  │───▶│ Critique │───▶│  Approval  │   │    │   │
│  │  │  │  Plan   │    │  (LLM)   │    │ / Refine   │   │    │   │
│  │  │  └─────────┘    └──────────┘    └────────────┘   │    │   │
│  │  └──────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 JANITOR (Background Task)                 │   │
│  │  • Prune stale memories (every 5 min)                    │   │
│  │  • Summarize long sessions (>20 turns)                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     VECTOR STORE                          │   │
│  │              LanceDB (Embedded, Zero-Copy)                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Context Cannon**: Recursively loads your entire codebase into context
- **Reflexion Engine**: Self-critiques plans before execution for safety
- **Embedded Memory**: LanceDB vector store with semantic search
- **Background Janitor**: Automatic memory pruning and session summarization
- **Zero Dependencies**: Single binary, no Docker required

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Rust 2024 Edition |
| AI Framework | Rig (`rig-core`) |
| LLM | Gemini 1.5 Pro |
| Vector Store | LanceDB (embedded) |
| Embeddings | FastEmbed (all-MiniLM-L6-v2) |
| Runtime | Tokio (async) |

## Quick Start

### 1. Prerequisites

- Rust 1.82+ (2024 edition)
- Gemini API Key

### 2. Setup

```bash
# Clone and enter directory
cd mobiusnano

# Copy environment template
cp .env.example .env

# Add your Gemini API key
echo "GEMINI_API_KEY=your_key_here" >> .env
```

### 3. Build & Run

```bash
# Development build
cargo run

# Release build (optimized for Apple Silicon)
cargo build --release
./target/release/ralph-nano
```

## Usage

```
╔══════════════════════════════════════════════════════════════╗
║                     RALPH-NANO v0.1.0                        ║
║            Maximum Intelligence, Minimum RAM                 ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:                                                   ║
║    /exit    - Exit the agent                                 ║
║    /clear   - Clear session history                          ║
║    /status  - Show memory and session stats                  ║
║    /path    - Set codebase path                              ║
╚══════════════════════════════════════════════════════════════╝

ralph> Add a health check endpoint to the API
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `LANCEDB_PATH` | Path to vector database | `.ralph-nano/lancedb` |
| `RUST_LOG` | Log level | `info` |

## Performance

Optimized for Apple Silicon with:
- `opt-level = 3` (maximum optimization)
- `lto = "fat"` (link-time optimization)
- `codegen-units = 1` (better inlining)
- `strip = true` (smaller binary)

Typical memory usage: **< 100MB** for most codebases.

## Safety

The Reflexion Engine ensures:

1. **Draft Plan Extraction**: Parses LLM output for files and commands
2. **Risk Assessment**: Scores destructive operations
3. **LLM Critique**: Secondary validation pass
4. **User Confirmation**: Interactive approval for shell commands

## License

MIT
