---
description: start the Ralph-Nano autonomous loop
---

To initialize and run your Ralph-Nano agent:

1. **Initialize DNA**
   Setup the isolated workspace and configuration.
   // turbo
   ```bash
   cargo run -- init
   ```

2. **Configure Environment**
   Ensure your `.env` file contains your `GEMINI_API_KEY`.
   ```bash
   echo "GEMINI_API_KEY=your_key_here" > .env
   ```

3. **Ignite the Brain**
   Start the interactive (or autonomous) loop.
   // turbo
   ```bash
   cargo run
   ```

---
> [!TIP]
> To enable **Headless Mode**, set `autonomous_mode = true` in `.ralph/config.toml`.
