---
description: managing Ralph-Nano's Auto-Didact engine
---

Use this to manage the documentation library and autonomous learning.

1. **Trigger Learning Phase**
   Learning occurs automatically on boot, but you can force a re-scan by restarting the agent.
   // turbo
   ```bash
   cargo run
   ```

2. **Toggle Autonomous Learning**
   To let Ralph ingest all documentation without prompting you, update `.ralph/config.toml`:
   ```toml
   autonomous_mode = true
   ```

3. **Check Known Libraries**
   Ralph will announce detected libraries on startup. You can also inspect the `library` table in CozoDB if you have the Cozo CLI installed.

---
> [!NOTE]
> Ralph-Nano currently supports Rust (`Cargo.toml`), Node.js (`package.json`), and Python (`requirements.txt`, `pyproject.toml`).
