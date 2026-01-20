# Architecture: Ralph-Nano

## High-Level System Design

Ralph-Nano operates as a single-binary, asynchronous Rust application. It leverages the Actor model pattern (via `tokio` tasks and `Arc<RwLock>`) to manage shared state between the main interaction loop and background maintenance tasks.

```mermaid
graph TD
    User[User Terminal] -->|Input| MainLoop[Context Cannon (Main Loop)]
    
    subgraph "Core Logic"
        MainLoop -->|Read| FS[File System]
        MainLoop -->|Query| VectorStore[LanceDB Vector Store]
        MainLoop -->|Prompt| Gemini[Gemini API]
        
        Gemini -->|Draft Plan| Reflexion[Reflexion Engine]
        Reflexion -->|Speculate| Shadow[Shadow Workspace]
        Shadow -->|cargo check| Verifier[Verifier]
        Verifier -->|Pass/Fail| Reflexion
        Reflexion -->|Approved Plan| Executor[Command Executor]
    end
    
    subgraph "Background Tasks"
        Janitor[Janitor Task] -->|Prune/Summarize| VectorStore
        Janitor -->|Read History| AppState
    end
    
    Executor -->|Modify| FS
    Executor -->|Run| Shell[System Shell]
    
    AppState[Shared State (History/Config)] -.-> MainLoop
    AppState -.-> Janitor
```

## Component Breakdown

### 1. `ContextCannon` (The Sensor)
- **Role**: Perception. It "sees" the codebase.
- **Implementation**: Uses `walkdir` to recursively scan the current directory.
- **Logic**:
  - Filtering: Ignores directories defined in `SKIP_DIRS` (e.g., `.git`, `node_modules`).
  - Filtering: Only includes files with extensions in `INCLUDE_EXTENSIONS`.
  - Formatting: Concatenates files into a standardized XML-like format for the LLM.

### 2. `ShadowWorkspace` (The Sandbox)
- **Role**: Safe Speculation.
- **Implementation**: A dedicated directory `.ralph/shadow` where edits are staged.
- **Logic**:
  - Copies `src/`, `Cargo.toml`, and `Cargo.lock` to a isolated environment.
  - Applies proposed LLM edits to the shadow files first.
  - Runs `cargo check` to verify that proposed changes don't break the build.
  - **Universal Mode**: Detection logic skips `cargo check` if it's not a Rust project, allowing the sandbox to be used for general file verification.

### 3. `VectorStore` (The Hippocampus)
- **Role**: Long-term memory.
- **Implementation**: Wrapper around `lancedb` + `fastembed`.
- **Schema**:
  - `id`: UUID
  - `content`: Text (Lesson/Rule/Fact)
  - `embedding`: 384-dim vector (All-MiniLM-L6-v2)
  - `created_at`, `last_accessed`, `access_count`: Metadata for eviction.
- **Operations**: `store_memory`, `query_memories`, `prune_stale_memories`.

### 4. `ReflexionEngine` (The Prefrontal Cortex)
- **Role**: Judgment and Safety.
- **Mechanism**:
  1. **Parses** response into `Action` variants (`WriteFile`, `ExecShell`).
  2. **Triggers Shadow Verification** for all file writes.
  3. **Generates Diffs** using the `similar` crate for visual user confirmation.
  4. **The Governor**: In autonomous mode, injects safety instructions to forbid `git push --force` and deletions outside shadow.

### 5. `Janitor` (The Glymphatic System)
- **Role**: Maintenance.
- **Implementation**: A detached `tokio::spawn` task.
- **Schedule**: Wakes up every 300s.
- **Tasks**:
  - **Pruning**: Deletes irrelevant memories.
  - **Consolidation**: Summarizes long histories into "Lessons".

### 6. `Main Loop` (v0.2.6)
- **Role**: Coordination and Oversight.
- **Implementation**: Infinite loop with Dual Role support and Pro-only inference.
- **Supervisor Features**:
    - **Symbol Cannon**: Replaces full workspace scan with a high-level symbol hierarchy (signatures).
    - **Structured Directives**: Enforces JSON communication (`{ "directives": [...] }`).
    - **Task Compression**: prunes `TASKS.md` to keep context focused.

## Data Flow

1. **User Turn**: Input is captured or polled from `TASKS.md`.
2. **Context Assembly**: `Codebase` + `Vector Memory` + `Session History`.
3. **Prompting**: Construction of the "Mega-Prompt".
4. **Inference**: Sent to Gemini 1.5 Pro.
5. **Speculation**: Proposed edits are written to `ShadowWorkspace`.
6. **Verification**: Build check (if applicable) and Diff generation.
7. **Execution**: If safe/approved, changes are committed to the real workspace.
8. **Memory Update**: Interaction is logged for the Janitor.
