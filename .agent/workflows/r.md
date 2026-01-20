---
description: run Ralph-Nano autonomous commands
---

Use this to interact with Ralph-Nano.

1. **Run Once**
   ```bash
   cargo run
   ```

2. **Run with autonomous instruction**
   If `autonomous_mode` is enabled in `.ralph/config.toml`, simply running `cargo run` will begin the polling/execution loop.

3. **Initialize workspace**
   ```bash
   cargo run -- init
   ```
