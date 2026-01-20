---
description: Testing Ralph-Nano autonomy and project status
---

1. **Verify Project Health**
   Run the speculative built-in verification.
   // turbo
   ```bash
   cargo check
   ```

2. **Run Autonomous Test**
   Execute with a small loop cap to verify task detection.
   // turbo
   ```bash
   cargo run
   ```

3. **Check Mission Status**
   Inspect active tasks.
   ```bash
   cat TASKS.md
   ```
