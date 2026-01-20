---
description: Refactoring code with Ralph-Nano autonomy
---

1. **Add Refactor Task**
   Append your refactor goal to `TASKS.md`.
   ```bash
   echo "- [ ] Refactor [module] to improve [metric]" >> TASKS.md
   ```

2. **Trigger Autonomous Refactor**
   Ensure `autonomous_mode = true` in `.ralph/config.toml` and start the agent.
   // turbo
   ```bash
   cargo run
   ```

3. **Monitor Diff**
   Review the speculative changes generated in the terminal.
