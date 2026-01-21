use tokio::sync::mpsc::Receiver;
use crate::io::events::Impulse;
use crate::core::state::GlobalState;
use std::sync::Arc;
use colored::*;

pub async fn cortex_loop(
    mut priority_rx: Receiver<Impulse>,
    mut background_rx: Receiver<Impulse>,
    state: Arc<GlobalState>
) {
    println!("{}", "🧠 Cortex Event Bus: ONLINE (QoS Enabled)".green().bold());
    
    loop {
        // "biased" mode ensures priority_rx is polled first.
        tokio::select! {
            biased;

            Some(impulse) = priority_rx.recv() => {
                handle_impulse(impulse, "FAST", &state).await;
            }
            Some(impulse) = background_rx.recv() => {
                // Background tasks yield if priority tasks are waiting in next tick
                handle_impulse(impulse, "SLOW", &state).await;
            }
            else => break, // All channels closed
        }
    }
}

async fn handle_impulse(impulse: Impulse, lane: &str, state: &GlobalState) {
    let lane_tag = if lane == "FAST" { "⚡".yellow() } else { "🐢".blue() };
    
    // Convert to Data-Oriented "Value" where possible (Temporal Decoupling)
    match impulse {
        Impulse::UserInput(input) => {
            println!("{} [UserInput] {}", lane_tag, input);
            let cortex = state.cortex.clone();
            tokio::spawn(async move {
                match cortex.generate(&input, crate::core::cortex::ThinkingLevel::High).await {
                    Ok(response) => {
                        println!("{}\n{}", "🤖 Ralph-Nano Response:".green().bold(), response);
                    }
                    Err(e) => eprintln!("Cortex error: {}", e),
                }
            });
        }
        Impulse::FileSystemEvent(event) => {
            // MOVE KnowledgeEngine work into a background spawn to prevent temporal braiding
            let memory = state.memory_raw.clone();
            tokio::spawn(async move {
                let engine = crate::knowledge::KnowledgeEngine::new(memory);
                for path in event.paths {
                    if let Err(e) = engine.ingest_file(&path).await {
                        eprintln!("Ingestion error for {:?}: {}", path, e);
                    }
                }
            });
        }
        Impulse::SwarmSignal(id, status) => {
            println!("{} [Swarm] Worker {} reported: {}", lane_tag, id, status);
        }
        Impulse::JanitorWakeup => {
            println!("{} [Janitor] Waking up...", lane_tag);
        }
        Impulse::SystemInterrupt => {
            println!("{}", "🛑 System Interrupt received. Shutting down...".red());
            std::process::exit(0);
        }
        Impulse::Error(e) => {
            eprintln!("{} [Error] {}", lane_tag, e.red());
        }
    }
}
