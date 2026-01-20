# Ralph-Nano 🦀

> **Maximum Intelligence, Minimum RAM. Persistent Autonomy.**

Ralph-Nano is a high-performance, single-binary autonomous coding agent written in **Rust**. It is designed specifically for **Apple Silicon** to provide a lightning-fast, native AI pair programmer experience without the bloat of Python, Node.js, or Docker.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![Platform](https://img.shields.io/badge/platform-macos-lightgrey.svg)](https://www.apple.com/macos/)

## 🚀 Key Features (v0.2.6)

*   **Bimodal Roleplay**: Toggle between **Executor** (default) and **Supervisor** mode for high-level architectural guidance.
*   **The Symbol Cannon**: Scans codebase for hierarchical symbols (signatures) to provide architectural context without implementational bloat.
*   **Structured Directives**: Mandates JSON communication (`{ "directives": [...] }`) for zero-ambiguity coordination.
*   **Task Compression**: Automatically archives completed tasks in `TASKS.md` to keep the context window lean.
*   **Pro-Only Intelligence**: Consolidates architecture on `gemini-3-pro-preview` for maximum reasoning depth.
*   **Persistent Autonomy**: The agent polls `TASKS.md` for new work, allowing for continuous integration.
*   **Shadow Workspace**: Proposals are safely staged and verified in a sandbox before touching your source code.

## 🛠️ Quick Start

### 1. Installation
```bash
git clone https://github.com/yourusername/ralph-nano.git
cd ralph-nano
cargo build --release
```

### 2. Initialize a Workspace
Go to your project directory and run:
```bash
/path/to/ralph-nano init
```
This creates the isolated `.ralph` directory and a default `config.toml`.

### 3. Setup Environment
Ensure your `.env` file in the project directory has your API key:
```bash
GEMINI_API_KEY=your_key_here
```

### 4. Ignite the Brain
```bash
/path/to/ralph-nano
```

## 🎮 Usage

Once running, you maintain a conversation with Ralph or add tasks to `TASKS.md`.

### Configuration (`.ralph/config.toml`)
```toml
project_name = "my-awesome-app"
autonomous_mode = true          # Set to true for headless operation
max_autonomous_loops = 50       # Circuit breaker for API spend
primary_model = "gemini-3-pro-preview"
```

### Slash Commands
- `/path <path>`: Change the target codebase directory.
- `/status`: Show memory usage, session turns, and token counts.
- `/clear`: Wipe the current session history (RAM only).
- `/exit`: Quit the agent.

## 🤝 Contributing

Contributions are welcome! Please check the [ROADMAP.md](ROADMAP.md) for current goals.

## ⚠️ Safety Notice

Ralph-Nano can execute shell commands and modify files.
- The **Shadow Workspace** attempts to catch broken builds.
- **The Governor** prevents dangerous autonomous actions like `git push --force`.
- **Always** commit your work before letting an agent modify your codebase.

---
*Built with ❤️ in Rust*
