// src/wasm_runtime.rs - Secure WASM Execution Environment for Ralph-Nano v1.0.0 (using wasmtime v29.0)

use anyhow::{Context, Result};
use wasmtime::*;
use wasmtime_wasi::preview1::{add_to_linker_sync, WasiP1Ctx};
use wasmtime_wasi::WasiCtxBuilder;

#[allow(dead_code)]
pub struct WasmRuntime {
    engine: Engine,
}

struct HostState {
    ctx: WasiP1Ctx,
}

impl WasmRuntime {
    pub fn new() -> Self {
        Self {
            engine: Engine::default(),
        }
    }

    pub fn execute(&mut self, wasm_bytes: &[u8], func_name: &str, args: Vec<i32>) -> Result<i32> {
        let mut linker: Linker<HostState> = Linker::new(&self.engine);
        add_to_linker_sync(&mut linker, |state| &mut state.ctx)?;

        let wasi_ctx = WasiCtxBuilder::new()
            .inherit_stdout()
            .inherit_stderr()
            .build_p1();

        let mut store = Store::new(&self.engine, HostState { ctx: wasi_ctx });

        let module = Module::from_binary(&self.engine, wasm_bytes)?;
        let instance = linker.instantiate(&mut store, &module)?;

        let func = instance
            .get_func(&mut store, func_name)
            .context(format!("Function {} not found", func_name))?;

        let wasm_args: Vec<Val> = args.into_iter().map(Val::I32).collect();
        let mut results = vec![Val::I32(0)];

        func.call(&mut store, &wasm_args, &mut results)?;

        if let Some(Val::I32(res)) = results.first() {
            Ok(*res)
        } else {
            Ok(0)
        }
    }
}
