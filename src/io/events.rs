use notify::Event;

#[derive(Debug)]
pub enum Impulse {
    UserInput(String),
    FileSystemEvent(Event),
    SwarmSignal(u64, String), // WorkerId, Status
    JanitorWakeup,
    SystemInterrupt,
    Error(String),
}
