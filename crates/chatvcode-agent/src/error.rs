use thiserror::Error;

use crate::types::AgentState;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("LLM inference failed: {0}")]
    LlmError(String),

    #[error("Tool execution failed: {tool_name}: {message}")]
    ToolError { tool_name: String, message: String },

    #[error("Agent timed out after {0} seconds")]
    Timeout(u64),

    #[error("Maximum steps ({0}) reached")]
    MaxStepsReached(usize),

    #[error("Invalid state transition from {from:?} to {to:?}")]
    InvalidStateTransition { from: AgentState, to: AgentState },

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Token budget exceeded: used {used}, budget {budget}")]
    TokenBudgetExceeded { used: usize, budget: usize },

    #[error("Agent cancelled")]
    Cancelled,

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type AgentResult<T> = Result<T, AgentError>;
