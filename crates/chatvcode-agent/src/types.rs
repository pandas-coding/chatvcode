use std::collections::HashMap;
use std::path::PathBuf;

use chatvcode_llm::{ChatTemplate, GenerationParams, ToolCall, ToolResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentState {
    Thinking,
    Acting,
    Done,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkingPhase {
    Planning,
    Observing,
    Concluding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStopReason {
    Completed,
    MaxSteps,
    Timeout,
    UserCancel,
    LoopDetected,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    pub step_number: usize,
    pub state: AgentState,
    pub thinking_phase: Option<ThinkingPhase>,
    pub thought: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
    pub duration_ms: u64,
    pub token_usage: TokenUsage,
}

#[derive(Debug, Clone)]
pub struct TokenBudgetConfig {
    pub total_budget: usize,
    pub system_prompt_reserve: usize,
    pub tool_result_max: usize,
    pub history_budget: usize,
    pub response_reserve: usize,
}

impl Default for TokenBudgetConfig {
    fn default() -> Self {
        Self {
            total_budget: 8192,
            system_prompt_reserve: 1500,
            tool_result_max: 2000,
            history_budget: 4000,
            response_reserve: 1500,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolRetryConfig {
    pub max_retries: usize,
    pub retry_on_timeout: bool,
    pub retry_on_transient_error: bool,
    pub backoff_ms: u64,
}

impl Default for ToolRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            retry_on_timeout: true,
            retry_on_transient_error: true,
            backoff_ms: 100,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_steps: usize,
    pub timeout_secs: u64,
    pub max_tool_calls_per_step: usize,
    pub allowed_tools: Vec<String>,
    pub project_path: PathBuf,
    pub verbose: bool,
    pub generation_params: GenerationParams,
    pub system_prompt: Option<String>,
    pub chat_template: ChatTemplate,
    pub token_budget: TokenBudgetConfig,
    pub tool_retry: ToolRetryConfig,
    pub require_plan_confirmation: bool,
    pub auto_approve_threshold: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: 10,
            timeout_secs: 120,
            max_tool_calls_per_step: 5,
            allowed_tools: vec![],
            project_path: PathBuf::from("."),
            verbose: false,
            generation_params: GenerationParams::default(),
            system_prompt: None,
            chat_template: ChatTemplate::Auto,
            token_budget: TokenBudgetConfig::default(),
            tool_retry: ToolRetryConfig::default(),
            require_plan_confirmation: false,
            auto_approve_threshold: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceReference {
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub symbol_name: Option<String>,
    pub relevance: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub total_steps: usize,
    pub tool_calls_by_name: HashMap<String, usize>,
    pub tool_success_rate: f64,
    pub avg_tool_latency_ms: u64,
    pub total_input_tokens: usize,
    pub total_output_tokens: usize,
    pub thinking_time_ms: u64,
    pub acting_time_ms: u64,
    pub loop_detection_triggered: usize,
    pub tool_cache_hits: usize,
    pub tool_retries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub answer: String,
    pub sources: Vec<SourceReference>,
    pub steps: Vec<AgentStep>,
    pub total_token_usage: TokenUsage,
    pub total_duration_ms: u64,
    pub total_tool_calls: usize,
    pub stop_reason: AgentStopReason,
    pub metrics: AgentMetrics,
}

#[derive(Debug, Clone)]
pub enum AgentEvent {
    StateChanged { from: AgentState, to: AgentState },
    ThinkingPhaseChanged { phase: ThinkingPhase },
    Thinking { text: String },
    ToolCallStarted { name: String, arguments: serde_json::Value },
    ToolCallCompleted { name: String, result: ToolResult, cached: bool },
    ToolCallFailed { name: String, error: String, will_retry: bool },
    ToolCallRetrying { name: String, attempt: usize, max_attempts: usize },
    StepCompleted { step: AgentStep },
    LoopDetected { detection_type: serde_json::Value },
    AnswerStarted,
    AnswerToken { text: String },
    AnswerCompleted { response: AgentResponse },
    Error { message: String },
    TokenBudgetWarning { used: usize, remaining: usize },
}
