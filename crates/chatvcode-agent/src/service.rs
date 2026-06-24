use std::sync::mpsc;

use crate::error::{AgentError, AgentResult};
use crate::types::{AgentEvent, AgentResponse};

pub trait AgentService: Send + Sync {
    fn run(&mut self, query: &str) -> AgentResult<AgentResponse>;

    fn run_stream(&mut self, query: &str) -> Result<mpsc::Receiver<AgentEvent>, AgentError>;

    fn cancel(&self);

    fn continue_execution(&mut self) -> AgentResult<AgentResponse>;

    fn retry(&mut self) -> AgentResult<AgentResponse>;
}
