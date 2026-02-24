use thiserror::Error;

pub type Result<T> = std::result::Result<T, AgentError>;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Memory error: {0}")]
    Memory(String),
    #[error("Planner error: {0}")]
    Planner(String),
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    #[error("Tool error: {0}")]
    Tool(String),
    #[error("Model error: {0}")]
    Model(String),
}
