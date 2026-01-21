pub mod state;
pub mod r#loop;
pub mod cortex;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Pure Value representation of an external event (Temporal Decoupling).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpulseValue {
    UserInput(String),
    FileChange(PathBuf),
    SystemSignal(String),
}

/// Pure Value representation of a decided action (Decomplection).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionValue {
    Think(String),
    Index(PathBuf),
    Notify(String),
}
