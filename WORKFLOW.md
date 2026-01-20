# Ralph-Nano System Workflow

This document outlines the operational lifecycle of the Ralph-Nano agent, from initialization to the main interaction loop and background maintenance.

## 1. Boot Sequence (The "Wake Up")

When you execute `cargo run --release`, the following initialization steps occur:

1.  **Environment Loading**:
    *   Loads `.env` file to retrieve the `GEMINI_API_KEY`.
    *   *Failure Condition*: Panics if the key is missing.

2.  **Cortex Initialization**:
    *   Instantiates the `GeminiClient`, setting up the direct HTTP connection to Google's Generative Language API.

3.  **Memory Mounting**:
    *   Connects to the embedded **CozoDB** instance at `.ralph/cozo.db`.
    *   **Schema Check**: Creates `library`, `nodes`, `edges`, and `cache` tables.
    *   **Embedding Model**: Initializes `candle` (BERT) for local execution on CPU/Metal.
    *   *Note*: This ensures zero-latency access to long-term memory and official documentation.

4.  **Auto-Didact Engine Scan**:
    *   **Manifest Detection**: Scans for `Cargo.toml`, `package.json`, `requirements.txt`, and `pyproject.toml`.
    *   **Learning Phase**: Compares detected libraries against the `library` table.
    *   **Ingestion**: If new libraries are found, Ralph performs a concurrent recursive scrape of `docs.rs`, `npmjs.com`, or `pypi.org`.
    *   **Context Priming**: Vectorizes and stores API definitions for RAG injection.

5.  **Janitor Spawn**:
    *   Launches the `janitor_task` as a detached `tokio` background thread to handle maintenance independently of the main loop.

## 2. The Main Loop (Persistent Autonomy)

Ralph-Nano v0.2.4 features **Persistent Autonomy**. The agent no longer exits after completing a task; it enters a smart polling state.

### Step A: Perception (The Polling Wait)
- **Task Detection**: Ralph checks `TASKS.md` for unchecked boxes `[ ]`.
- **Idle State**: If all tasks are `[x]`, Ralph sleeps for 5 seconds and performs one final **Secure Git Sync**.
- **Resume**: As soon as a new task appears, the **Context Cannon** fires.

### Step B: The Context Cannon
*   **Recursive Scan**: Ralph walks the project directory.
*   **Filtering**: It filters out noise (directories like `.git`, `node_modules`, `target`) to ensure only relevant source code is read.
*   **Token Optimization**: Files are concatenated into a structured format, providing the LLM with the *entire* current state of the project.

### Step C: Cognition (The Mega-Prompt)
*   A comprehensive prompt is constructed:
    ```
    [SYSTEM_RULES]       <-- Behavioral directives & Safety Governor
    [OFFICIAL_DOCS]      <-- Recursive context from the Auto-Didact library
    [RETRIEVED_LESSONS]  <-- Vector context from CozoDB
    [FULL_CODEBASE]      <-- The raw source code
    [USER_INPUT]         <-- Task prompt or Polling status
    ```
*   This prompt is sent to the **Gemini 1.5 Pro** API.

### Step D: Safe Speculation (The Shadow Workspace)
Before modifying your files, Ralph speculative executes:
1.  **Stage**: Edits are written to `.ralph/shadow`.
2.  **Verify**: 
    - **Rust Projects**: Runs `cargo check` on the shadow files.
    - **Universal Mode**: Skips compilation check if no `Cargo.toml` is found.
3.  **Visual Confirmation**: Generates a unified diff using the `similar` crate.
4.  **The Governor**: Safety logic forbids `git push --force` and deletions outside the shadow root during autonomous mode.

### Step E: Execution
1.  **Approval**: In manual mode, waits for user `[y/N]`. In autonomous mode, consumes the "Governor" approval automatically.
2.  **Commit**: Verified changes are moved from the shadow workspace to your real files.
3.  **Git Persistence**: Automatically runs `git add`, `git commit`, and `git push` (after security scanning for leaked keys).

## 3. Background Hygiene (The Janitor)

The Janitor task runs concurrently, waking up every 5 minutes:

*   **Pruning**: Deletes irrelevant memories to keep the index efficient.
*   **Consolidation**: Checks if session history exceeds 20 turns. If so, it generates a "Lesson" summary, stores it in CozoDB, and clears the RAM buffer.

---

**Summary**: This workflow ensures Ralph-Nano remains fast (local vector search), smart (full codebase context), safe (shadow workspace verification), and persistent (continuous polling autonomy).
