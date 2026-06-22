//! Data model types for chatvcode-llm.
//!
//! Defines the configuration, model metadata, generation parameters, and
//! response types used across the LLM inference pipeline.

use std::path::PathBuf;
use std::time::Duration;


// Re-export chat types for backward compatibility
pub use crate::chat::{
    ChatMessage, ChatPromptBuilder, ChatSession, ChatTemplate,
    token_estimate, token_estimate_messages,
};

// ---------------------------------------------------------------------------
// Model configuration
// ---------------------------------------------------------------------------

/// Configuration for loading and running an LLM model.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Path to the GGUF model file.
    pub model_path: PathBuf,

    /// Context size (maximum token length). 0 = use model default.
    pub n_ctx: u32,

    /// Maximum batch size for prompt processing.
    pub n_batch: u32,

    /// Physical micro-batch size.
    pub n_ubatch: u32,

    /// Number of threads for single-token generation.
    pub n_threads: i32,

    /// Number of threads for batch/prompt processing.
    pub n_threads_batch: i32,

    /// Number of GPU layers to offload. -1 = all, 0 = CPU only.
    pub n_gpu_layers: i32,

    /// Whether to use memory-mapped I/O for model loading.
    pub use_mmap: bool,

    /// Whether to lock model pages in RAM.
    pub use_mlock: bool,

    /// Chat template override. `None` means auto-detect from GGUF metadata.
    pub chat_template: Option<String>,

    /// Show verbose llama.cpp/ggml log output (model loading details, etc.).
    /// When `false` (default), only warnings and errors from the C backend are shown.
    pub verbose_log: bool,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            n_ctx: 8192,
            n_batch: 8192,
            n_ubatch: 512,
            n_threads: num_cpus::get() as i32,
            n_threads_batch: num_cpus::get() as i32,
            n_gpu_layers: 0,
            use_mmap: true,
            use_mlock: false,
            chat_template: None,
            verbose_log: false,
        }
    }
}

impl LlmConfig {
    /// Create a new `LlmConfig` with the given model path and defaults.
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self { model_path: model_path.into(), ..Self::default() }
    }

    /// Set the context size.
    #[must_use]
    pub const fn with_n_ctx(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        // Ensure n_batch >= n_ctx so that llama_decode can process the full
        // context window in a single batch.  n_ubatch stays at a moderate size
        // (512) to keep memory usage reasonable during micro-batched processing.
        if self.n_batch < n_ctx {
            self.n_batch = n_ctx;
        }
        self
    }

    /// Set the batch size.
    ///
    /// Note: `n_batch` must be >= `n_ctx` for correct llama.cpp operation.
    /// If you set a value smaller than `n_ctx`, it will be silently upgraded.
    #[must_use]
    pub const fn with_n_batch(mut self, n_batch: u32) -> Self {
        self.n_batch = if n_batch < self.n_ctx { self.n_ctx } else { n_batch };
        self
    }

    /// Set the number of threads.
    #[must_use]
    pub const fn with_n_threads(mut self, n_threads: i32) -> Self {
        self.n_threads = n_threads;
        self.n_threads_batch = n_threads;
        self
    }

    /// Set the number of GPU layers to offload.
    #[must_use]
    pub const fn with_n_gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    /// Set whether to use mmap.
    #[must_use]
    pub const fn with_mmap(mut self, use_mmap: bool) -> Self {
        self.use_mmap = use_mmap;
        self
    }

    /// Override the chat template.
    pub fn with_chat_template(mut self, template: impl Into<String>) -> Self {
        self.chat_template = Some(template.into());
        self
    }

    /// Enable or disable verbose llama.cpp/ggml logging.
    ///
    /// When enabled, all model loading details (tensor creation, KV metadata,
    /// backend registration, etc.) are forwarded to Rust's log output.
    /// When disabled (default), only warnings and errors are shown.
    #[must_use]
    pub const fn with_verbose_log(mut self, verbose: bool) -> Self {
        self.verbose_log = verbose;
        self
    }
}

// ---------------------------------------------------------------------------
// Model metadata (informational)
// ---------------------------------------------------------------------------

/// Discovered metadata about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Human-readable model description (architecture + size).
    pub description: String,

    /// Model architecture name (e.g. "llama", "mistral", "gemma").
    pub architecture: String,

    /// Total number of parameters.
    pub n_params: u64,

    /// Model file size in bytes.
    pub size_bytes: u64,

    /// Context size the model was trained with.
    pub n_ctx_train: i32,

    /// Embedding dimension.
    pub n_embd: i32,

    /// Number of transformer layers.
    pub n_layer: i32,

    /// Number of attention heads.
    pub n_head: i32,

    /// Number of key/value heads.
    pub n_head_kv: i32,

    /// Vocabulary size.
    pub n_vocab: i32,

    /// Vocabulary type.
    pub vocab_type: String,

    /// GGUF file type (quantization level).
    pub ftype: String,

    /// Available chat template name or "none".
    pub chat_template_available: bool,

    /// `RoPE` type.
    pub rope_type: String,

    /// Whether the model has an encoder (encoder-decoder models).
    pub has_encoder: bool,

    /// Whether the model has a decoder.
    pub has_decoder: bool,
}

// ---------------------------------------------------------------------------
// Generation parameters
// ---------------------------------------------------------------------------

/// Parameters controlling the text generation process.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    /// Temperature for sampling (0.0 = greedy).
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold. 1.0 = disabled.
    pub top_p: f32,

    /// Top-k sampling. 0 = disabled.
    pub top_k: i32,

    /// Min-p sampling threshold. 0.0 = disabled.
    pub min_p: f32,

    /// Repeat penalty. 1.0 = disabled.
    pub repeat_penalty: f32,

    /// Number of last tokens to consider for repeat penalty.
    pub repeat_last_n: i32,

    /// Maximum number of tokens to generate.
    pub max_tokens: i32,

    /// Stop strings. Generation stops if any of these appear.
    pub stop_strings: Vec<String>,

    /// Random seed. Use `u32::MAX` for random.
    pub seed: u32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            min_p: 0.0,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            max_tokens: 2048,
            stop_strings: Vec::new(),
            seed: u32::MAX,
        }
    }
}

impl GenerationParams {
    /// Create params for greedy decoding (deterministic).
    #[must_use]
    pub fn greedy() -> Self {
        Self { temperature: 0.0, top_p: 1.0, top_k: 1, min_p: 0.0, ..Self::default() }
    }

    /// Set temperature.
    #[must_use]
    pub const fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Set top-p.
    #[must_use]
    pub const fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Set top-k.
    #[must_use]
    pub const fn with_top_k(mut self, k: i32) -> Self {
        self.top_k = k;
        self
    }

    /// Set max tokens, clamped to `[1, 65536]`.
    #[must_use]
    pub const fn with_max_tokens(mut self, n: i32) -> Self {
        self.max_tokens = if n < 1 { 1 } else if n > 65536 { 65536 } else { n };
        self
    }

    /// Set seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }
}

// ---------------------------------------------------------------------------
// Token usage
// ---------------------------------------------------------------------------

/// Token usage statistics for an inference call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenUsage {
    /// Number of prompt (input) tokens.
    pub prompt_tokens: i32,

    /// Number of completion (output) tokens.
    pub completion_tokens: i32,

    /// Total tokens (prompt + completion).
    pub total_tokens: i32,
}

impl TokenUsage {
    /// Create with prompt and completion counts.
    #[must_use]
    pub const fn new(prompt_tokens: i32, completion_tokens: i32) -> Self {
        Self { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens }
    }
}

// ---------------------------------------------------------------------------
// Stop reason
// ---------------------------------------------------------------------------

/// Reason why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// End-of-sentence token generated.
    Eos,
    /// Maximum token count reached.
    MaxTokens,
    /// A stop string was matched.
    StopString(String),
    /// User cancelled generation.
    Cancelled,
    /// An error occurred during generation.
    Error(String),
}

// ---------------------------------------------------------------------------
// Stream events
// ---------------------------------------------------------------------------

/// Events emitted during streaming inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// Generation has started.
    Started,

    /// A single token of generated text.
    Token(String),

    /// Generation completed successfully.
    Completed,

    /// Generation was cancelled by the user.
    Cancelled,

    /// An error occurred during generation.
    Error(String),
}

impl StreamEvent {
    /// Returns `true` if this is a terminal event (Completed, Cancelled, or Error).
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Error(_))
    }

    /// Returns `true` if this is a Token event.
    #[must_use]
    pub const fn is_token(&self) -> bool {
        matches!(self, Self::Token(_))
    }

    /// Extract the token text if this is a Token event.
    #[must_use]
    pub fn as_token(&self) -> Option<&str> {
        match self {
            Self::Token(text) => Some(text),
            _ => None,
        }
    }

    /// Extract the error message if this is an Error event.
    #[must_use]
    pub fn as_error(&self) -> Option<&str> {
        match self {
            Self::Error(msg) => Some(msg),
            _ => None,
        }
    }

    /// Returns `true` if this event indicates the generation was successful.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Started | Self::Token(_) | Self::Completed)
    }
}

// ---------------------------------------------------------------------------
// Inference response
// ---------------------------------------------------------------------------

/// Result of a (non-streaming) inference call.
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    /// The generated text.
    pub text: String,

    /// Why generation stopped.
    pub stop_reason: StopReason,

    /// Token usage statistics.
    pub token_usage: TokenUsage,

    /// Total generation duration.
    pub duration: Duration,

    /// Time to first token (only meaningful for streaming, but kept for API symmetry).
    pub time_to_first_token: Option<Duration>,

    /// Tokens-per-second.
    pub tokens_per_second: f64,
}

// ---------------------------------------------------------------------------
// Chat message
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(unused_imports)]
    use insta::assert_snapshot;

    #[test]
    fn test_llm_config_defaults() {
        let config = LlmConfig::default();
        assert_eq!(config.n_ctx, 8192);
        assert_eq!(config.n_batch, 8192);
        assert_eq!(config.n_gpu_layers, 0);
        assert!(config.use_mmap);
        assert!(!config.use_mlock);
        assert!(config.chat_template.is_none());
    }

    #[test]
    fn test_llm_config_builder() {
        let config = LlmConfig::new("models/test.gguf")
            .with_n_ctx(4096)
            .with_n_threads(8)
            .with_n_gpu_layers(32)
            .with_chat_template("llama3");

        assert_eq!(config.model_path.to_string_lossy(), "models/test.gguf");
        assert_eq!(config.n_ctx, 4096);
        assert_eq!(config.n_threads, 8);
        assert_eq!(config.n_gpu_layers, 32);
        assert_eq!(config.chat_template.as_deref(), Some("llama3"));
    }

    #[test]
    fn test_generation_params_default() {
        let params = GenerationParams::default();
        assert!((params.temperature - 0.7).abs() < f32::EPSILON);
        assert!((params.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(params.top_k, 40);
        assert_eq!(params.max_tokens, 2048);
        assert_eq!(params.seed, u32::MAX);
    }

    #[test]
    fn test_generation_params_max_tokens_clamping() {
        assert_eq!(GenerationParams::default().with_max_tokens(100).max_tokens, 100);
        assert_eq!(GenerationParams::default().with_max_tokens(0).max_tokens, 1);
        assert_eq!(GenerationParams::default().with_max_tokens(-1).max_tokens, 1);
        assert_eq!(GenerationParams::default().with_max_tokens(100_000).max_tokens, 65536);
        assert_eq!(GenerationParams::default().with_max_tokens(65536).max_tokens, 65536);
    }

    #[test]
    fn test_greedy_params() {
        let params = GenerationParams::greedy();
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.top_k, 1);
    }

    #[test]
    fn test_n_ctx_upgrades_n_batch() {
        // Default n_batch=8192, setting n_ctx=4096 should NOT reduce n_batch
        let config = LlmConfig::new("test.gguf").with_n_ctx(4096);
        assert_eq!(config.n_ctx, 4096);
        assert_eq!(config.n_batch, 8192); // n_batch stays at its default

        // Setting n_ctx larger than n_batch should upgrade n_batch
        let config = LlmConfig::new("test.gguf")
            .with_n_batch(512)
            .with_n_ctx(8192);
        assert_eq!(config.n_ctx, 8192);
        assert_eq!(config.n_batch, 8192); // auto-upgraded to match n_ctx

        // Setting n_batch smaller than n_ctx should be silently upgraded
        let config = LlmConfig::new("test.gguf")
            .with_n_ctx(8192)
            .with_n_batch(512);
        assert_eq!(config.n_ctx, 8192);
        assert_eq!(config.n_batch, 8192); // n_batch upgraded to match n_ctx
    }

    #[test]
    fn test_token_usage() {
        let usage = TokenUsage::new(10, 20);
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn test_chat_message_helpers() {
        let sys = ChatMessage::system("You are helpful.");
        assert_eq!(sys.role, "system");

        let user = ChatMessage::user("Hello");
        assert_eq!(user.role, "user");

        let assistant = ChatMessage::assistant("Hi there!");
        assert_eq!(assistant.role, "assistant");
    }

    #[test]
    fn test_chat_template_variants() {
        assert_eq!(ChatTemplate::Auto.template_name(), None);
        assert_eq!(ChatTemplate::Raw.template_name(), Some("raw"));
        assert_eq!(ChatTemplate::ChatML.template_name(), Some("chatml"));
        assert_eq!(ChatTemplate::Llama3.template_name(), Some("llama3"));
        assert_eq!(ChatTemplate::DeepSeek.template_name(), Some("deepseek"));
        assert_eq!(
            ChatTemplate::Custom("{{ bos_token }}".into()).custom_template(),
            Some("{{ bos_token }}")
        );
    }

    #[test]
    fn test_chat_template_format_raw() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let prompt = ChatTemplate::Raw.format(&messages, true).unwrap();
        assert_eq!(prompt, "You are helpful.\n\nHello");
    }

    #[test]
    fn test_chat_template_format_chatml() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let prompt = ChatTemplate::ChatML.format(&messages, true).unwrap();
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_chat_template_format_llama3() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let prompt = ChatTemplate::Llama3.format(&messages, true).unwrap();
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>\n\nYou are helpful."));
        assert!(prompt.contains("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_chat_template_format_deepseek() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let prompt = ChatTemplate::DeepSeek.format(&messages, true).unwrap();
        assert!(prompt.starts_with("<｜begin▁of▁sentence｜>"));
        assert!(prompt.contains("<｜User｜>Hello"));
        assert!(prompt.ends_with("<｜Assistant｜>"));
    }

    #[test]
    fn test_chat_template_format_deepseek_with_assistant() {
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
        ];
        let prompt = ChatTemplate::DeepSeek.format(&messages, false).unwrap();
        assert!(prompt.contains("<｜User｜>Hello"));
        assert!(prompt.contains("<｜Assistant｜>Hi there!<｜end▁of▁sentence｜>"));
        assert!(!prompt.ends_with("<｜Assistant｜>"));
    }

    #[test]
    fn test_chat_template_format_auto_fallback_to_chatml() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let auto = ChatTemplate::Auto.format(&messages, true).unwrap();
        let chatml = ChatTemplate::ChatML.format(&messages, true).unwrap();
        assert_eq!(auto, chatml);
    }

    #[test]
    fn test_chat_template_format_custom_errors() {
        let messages = vec![ChatMessage::user("Hello")];
        let result = ChatTemplate::Custom("{{ messages }}".into()).format(&messages, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_stream_event_is_terminal() {
        assert!(!StreamEvent::Started.is_terminal());
        assert!(!StreamEvent::Token(String::new()).is_terminal());
        assert!(StreamEvent::Completed.is_terminal());
        assert!(StreamEvent::Cancelled.is_terminal());
        assert!(StreamEvent::Error(String::new()).is_terminal());
    }

    #[test]
    fn test_stream_event_is_token() {
        assert!(!StreamEvent::Started.is_token());
        assert!(StreamEvent::Token("test".into()).is_token());
        assert!(!StreamEvent::Completed.is_token());
        assert!(!StreamEvent::Cancelled.is_token());
        assert!(!StreamEvent::Error(String::new()).is_token());
    }

    #[test]
    fn test_stream_event_as_token() {
        assert_eq!(StreamEvent::Token("hello".into()).as_token(), Some("hello"));
        assert_eq!(StreamEvent::Started.as_token(), None);
        assert_eq!(StreamEvent::Completed.as_token(), None);
        assert_eq!(StreamEvent::Cancelled.as_token(), None);
        assert_eq!(StreamEvent::Error(String::new()).as_token(), None);
    }

    #[test]
    fn test_stream_event_as_error() {
        assert_eq!(StreamEvent::Error("test err".into()).as_error(), Some("test err"));
        assert_eq!(StreamEvent::Started.as_error(), None);
        assert_eq!(StreamEvent::Token(String::new()).as_error(), None);
        assert_eq!(StreamEvent::Completed.as_error(), None);
        assert_eq!(StreamEvent::Cancelled.as_error(), None);
    }

    #[test]
    fn test_stream_event_is_success() {
        assert!(StreamEvent::Started.is_success());
        assert!(StreamEvent::Token(String::new()).is_success());
        assert!(StreamEvent::Completed.is_success());
        assert!(!StreamEvent::Cancelled.is_success());
        assert!(!StreamEvent::Error(String::new()).is_success());
    }

    #[test]
    fn test_stream_event_equality() {
        assert_eq!(StreamEvent::Started, StreamEvent::Started);
        assert_eq!(StreamEvent::Token("a".into()), StreamEvent::Token("a".into()));
        assert_ne!(StreamEvent::Token("a".into()), StreamEvent::Token("b".into()));
        assert_eq!(StreamEvent::Completed, StreamEvent::Completed);
        assert_eq!(StreamEvent::Cancelled, StreamEvent::Cancelled);
        assert_eq!(StreamEvent::Error("a".into()), StreamEvent::Error("a".into()));
        assert_ne!(StreamEvent::Started, StreamEvent::Completed);
    }

    #[test]
    fn test_token_estimate_empty() {
        assert_eq!(token_estimate(""), 0);
    }

    #[test]
    fn test_token_estimate_short_text() {
        assert_eq!(token_estimate("Hi"), 1);
        assert_eq!(token_estimate("Hello"), 2);
        assert_eq!(token_estimate("Hello world"), 3);
    }

    #[test]
    fn test_token_estimate_longer_text() {
        let text = "The quick brown fox jumps over the lazy dog";
        let est = token_estimate(text);
        assert!(est > 0);
        assert!(est <= text.len());
    }

    #[test]
    fn test_token_estimate_messages_overhead() {
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
        ];
        let est = token_estimate_messages(&messages);
        let individual_sum = token_estimate("Hello") + token_estimate("Hi there!");
        assert!(est > individual_sum);
    }

    #[test]
    fn test_token_estimate_messages_empty_vec() {
        assert_eq!(token_estimate_messages(&[]), 0);
    }

    #[test]
    fn test_chat_session_new_defaults() {
        let session = ChatSession::new(ChatTemplate::ChatML);
        assert!(session.is_empty());
        assert_eq!(session.len(), 0);
        assert_eq!(session.estimated_tokens(), 0);
        assert!(session.get_system_prompt().is_none());
        assert_eq!(session.turn_count(), 0);
    }

    #[test]
    fn test_chat_session_system_prompt_builder() {
        let session = ChatSession::new(ChatTemplate::ChatML)
            .system_prompt("You are a helpful assistant.");
        assert_eq!(session.get_system_prompt(), Some("You are a helpful assistant."));
    }

    #[test]
    fn test_chat_session_set_clear_system_prompt() {
        let mut session = ChatSession::new(ChatTemplate::ChatML);
        assert!(session.get_system_prompt().is_none());
        session.set_system_prompt(Some("New prompt".to_string()));
        assert_eq!(session.get_system_prompt(), Some("New prompt"));
        session.clear_system_prompt();
        assert!(session.get_system_prompt().is_none());
    }

    #[test]
    fn test_chat_session_add_messages() {
        let mut session = ChatSession::new(ChatTemplate::ChatML);
        session.add_message(ChatMessage::user("Hello"));
        assert_eq!(session.len(), 1);
        assert!(!session.is_empty());
        session.add_message(ChatMessage::assistant("Hi!"));
        assert_eq!(session.len(), 2);
        assert_eq!(session.turn_count(), 1);
    }

    #[test]
    fn test_chat_session_multiple_turns() {
        let mut session = ChatSession::new(ChatTemplate::ChatML);
        session.add_user_message("Question 1");
        session.add_assistant_message("Answer 1");
        session.add_user_message("Question 2");
        session.add_assistant_message("Answer 2");
        assert_eq!(session.len(), 4);
        assert_eq!(session.turn_count(), 2);
    }

    #[test]
    fn test_chat_session_turn_count_incremental() {
        let mut session = ChatSession::new(ChatTemplate::ChatML);
        assert_eq!(session.turn_count(), 0);
        session.add_user_message("Q1");
        assert_eq!(session.turn_count(), 0);
        session.add_assistant_message("A1");
        assert_eq!(session.turn_count(), 1);
        session.add_user_message("Q2");
        assert_eq!(session.turn_count(), 1);
        session.add_assistant_message("A2");
        assert_eq!(session.turn_count(), 2);
    }

    #[test]
    fn test_chat_session_clear_keeps_system() {
        let mut session = ChatSession::new(ChatTemplate::ChatML)
            .system_prompt("System prompt");
        session.add_user_message("Hello");
        session.add_assistant_message("Hi!");
        assert_eq!(session.len(), 2);
        session.clear();
        assert_eq!(session.len(), 0);
        assert!(session.is_empty());
        assert!(session.get_system_prompt().is_some());
    }

    #[test]
    fn test_chat_session_reset_clears_all() {
        let mut session = ChatSession::new(ChatTemplate::ChatML)
            .system_prompt("System prompt");
        session.add_user_message("Hello");
        session.reset();
        assert_eq!(session.len(), 0);
        assert!(session.get_system_prompt().is_none());
    }

    #[test]
    fn test_chat_session_build_prompt_chatml() {
        let mut session = ChatSession::new(ChatTemplate::ChatML)
            .system_prompt("You are helpful.");
        session.add_user_message("Hello");
        session.add_assistant_message("Hi!");
        let prompt = session.build_prompt().unwrap();
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("Hello"));
        assert!(prompt.contains("Hi!"));
        assert!(prompt.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_kv_cache_state_new_session_zero() {
        let session = ChatSession::new(ChatTemplate::ChatML);
        assert_eq!(session.kv_cache_state(), 0);
    }

    #[test]
    fn test_kv_cache_state_clear_resets() {
        let mut session = ChatSession::new(ChatTemplate::ChatML);
        session.add_user_message("Hello");
        // Simulate a successful turn that set cache state
        let mut session_test = session.clone();
        session_test.kv_cache_state = 42;

        session_test.clear();
        assert_eq!(session_test.kv_cache_state(), 0);
    }

    #[test]
    fn test_kv_cache_state_reset_resets() {
        let mut session = ChatSession::new(ChatTemplate::ChatML);
        session.kv_cache_state = 42;
        session.reset();
        assert_eq!(session.kv_cache_state(), 0);
    }

    #[test]
    fn test_system_prompt_change_resets_kv_cache() {
        let mut session = ChatSession::new(ChatTemplate::ChatML);
        session.kv_cache_state = 42;
        session.set_system_prompt(Some("New prompt".into()));
        assert_eq!(session.kv_cache_state(), 0);
    }

    #[test]
    fn test_clear_system_prompt_resets_kv_cache() {
        let mut session = ChatSession::new(ChatTemplate::ChatML)
            .system_prompt("Initial");
        session.kv_cache_state = 42;
        session.clear_system_prompt();
        assert_eq!(session.kv_cache_state(), 0);
    }

    #[test]
    fn test_invalidate_kv_cache() {
        let mut session = ChatSession::new(ChatTemplate::ChatML);
        session.kv_cache_state = 99;
        session.invalidate_kv_cache();
        assert_eq!(session.kv_cache_state(), 0);
    }

    #[test]
    fn test_chat_session_build_prompt_with() {
        let mut session = ChatSession::new(ChatTemplate::ChatML)
            .system_prompt("You are helpful.");
        session.add_user_message("Hello");
        session.add_assistant_message("Hi!");

        let prompt = session.build_prompt_with("What's next?").unwrap();
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("Hello"));
        assert!(prompt.contains("What's next?"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));

        // build_prompt_with should NOT modify the session
        assert_eq!(session.len(), 2);
    }

    #[test]
    fn test_chat_session_max_history_turns() {
        let session = ChatSession::new(ChatTemplate::ChatML).max_history_turns(2);

        let mut s = session.clone();
        // Add 3 turns (6 messages)
        for i in 0..3 {
            s.add_user_message(format!("Q{i}"));
            s.add_assistant_message(format!("A{i}"));
        }
        assert_eq!(s.len(), 6);

        // The prompt should only include the last 2 turns (4 messages)
        let prompt = s.build_prompt_with("Q3").unwrap();
        assert!(!prompt.contains("Q0"));
        assert!(prompt.contains("Q1"));
        assert!(prompt.contains("Q2"));
    }

    #[test]
    fn test_chat_session_token_estimation_after_adding() {
        let mut session = ChatSession::new(ChatTemplate::ChatML);
        assert_eq!(session.estimated_tokens(), 0);

        session.add_user_message("Hello world");
        assert!(session.estimated_tokens() > 0);

        session.add_assistant_message("Hi there!");
        assert!(session.estimated_tokens() > token_estimate("Hello world") + token_estimate("Hi there!"));
    }

    #[test]
    fn test_chat_session_to_json_from_json_roundtrip() {
        let mut session = ChatSession::new(ChatTemplate::ChatML)
            .system_prompt("System");
        session.add_user_message("Hello");
        session.add_assistant_message("Hi!");
        session.kv_cache_state = 42;

        let json = session.to_json().unwrap();

        let restored = ChatSession::from_json(&json, ChatTemplate::ChatML).unwrap();
        assert_eq!(restored.session_id(), session.session_id());
        assert_eq!(restored.get_system_prompt(), session.get_system_prompt());
        assert_eq!(restored.len(), session.len());
        assert_eq!(restored.messages(), session.messages());
        // KV cache state is NOT persisted (it's runtime state)
        assert_eq!(restored.kv_cache_state(), 0);
    }

    // -----------------------------------------------------------------------
    // Chat template snapshot tests
    // -----------------------------------------------------------------------

    fn chat_template_fixture_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage::system("You are a helpful coding assistant."),
            ChatMessage::user("What does the `main` function do?"),
            ChatMessage::assistant("It prints a greeting."),
        ]
    }

    #[test]
    fn snapshot_chat_template_chatml() {
        let prompt = ChatTemplate::ChatML
            .format(&chat_template_fixture_messages(), true)
            .unwrap();
        assert_snapshot!(prompt);
    }

    #[test]
    fn snapshot_chat_template_llama3() {
        let prompt = ChatTemplate::Llama3
            .format(&chat_template_fixture_messages(), true)
            .unwrap();
        assert_snapshot!(prompt);
    }

    #[test]
    fn snapshot_chat_template_deepseek() {
        let prompt = ChatTemplate::DeepSeek
            .format(&chat_template_fixture_messages(), true)
            .unwrap();
        assert_snapshot!(prompt);
    }

    #[test]
    fn snapshot_chat_template_raw() {
        let prompt = ChatTemplate::Raw
            .format(&chat_template_fixture_messages(), true)
            .unwrap();
        assert_snapshot!(prompt);
    }

    #[test]
    fn snapshot_chat_template_auto_fallback() {
        let prompt = ChatTemplate::Auto
            .format(&chat_template_fixture_messages(), true)
            .unwrap();
        assert_snapshot!(prompt);
    }
}
