//! Data model types for atlas-llm.
//!
//! Defines the configuration, model metadata, generation parameters, and
//! response types used across the LLM inference pipeline.

use std::path::PathBuf;
use std::time::Duration;

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
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            n_ctx: 2048,
            n_batch: 512,
            n_ubatch: 512,
            n_threads: num_cpus::get() as i32,
            n_threads_batch: num_cpus::get() as i32,
            n_gpu_layers: 0,
            use_mmap: true,
            use_mlock: false,
            chat_template: None,
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
        self
    }

    /// Set the batch size.
    #[must_use]
    pub const fn with_n_batch(mut self, n_batch: u32) -> Self {
        self.n_batch = n_batch;
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
            max_tokens: 512,
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

    /// Set max tokens.
    #[must_use]
    pub const fn with_max_tokens(mut self, n: i32) -> Self {
        self.max_tokens = n;
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

/// A single message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    /// Role of the speaker (e.g., "system", "user", "assistant").
    pub role: String,

    /// Message content.
    pub content: String,
}

impl ChatMessage {
    /// Create a new chat message.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into() }
    }

    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

// ---------------------------------------------------------------------------
// Chat template
// ---------------------------------------------------------------------------

/// Supported chat template variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Auto-detect from model metadata (falls back to `ChatML` if not found).
    Auto,

    /// Raw text, no template formatting applied.
    Raw,

    /// `ChatML` format (used by many models).
    /// Format: `<|im_start|>role\ncontent<|im_end|>\n`
    ChatML,

    /// Llama 3 format.
    /// Format: `<|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
    Llama3,

    /// Custom jinja template string.
    Custom(String),
}

impl ChatTemplate {
    /// Returns the template name used for `llama_chat_apply_template`.
    #[must_use]
    pub const fn template_name(&self) -> Option<&str> {
        match self {
            Self::Auto => None, // use model default
            Self::Raw => Some("raw"),
            Self::ChatML => Some("chatml"),
            Self::Llama3 => Some("llama3"),
            Self::Custom(_) => None, // handled separately
        }
    }

    /// Returns the custom jinja template string, if any.
    #[must_use]
    pub const fn custom_template(&self) -> Option<&str> {
        match self {
            Self::Custom(tmpl) => Some(tmpl.as_str()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_config_defaults() {
        let config = LlmConfig::default();
        assert_eq!(config.n_ctx, 2048);
        assert_eq!(config.n_batch, 512);
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
        assert_eq!(params.max_tokens, 512);
        assert_eq!(params.seed, u32::MAX);
    }

    #[test]
    fn test_greedy_params() {
        let params = GenerationParams::greedy();
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.top_k, 1);
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
        assert_eq!(
            ChatTemplate::Custom("{{ bos_token }}".into()).custom_template(),
            Some("{{ bos_token }}")
        );
    }

    #[test]
    fn test_stream_event_is_terminal() {
        assert!(!StreamEvent::Started.is_terminal());
        assert!(!StreamEvent::Token("".into()).is_terminal());
        assert!(StreamEvent::Completed.is_terminal());
        assert!(StreamEvent::Cancelled.is_terminal());
        assert!(StreamEvent::Error("".into()).is_terminal());
    }

    #[test]
    fn test_stream_event_is_token() {
        assert!(!StreamEvent::Started.is_token());
        assert!(StreamEvent::Token("test".into()).is_token());
        assert!(!StreamEvent::Completed.is_token());
        assert!(!StreamEvent::Cancelled.is_token());
        assert!(!StreamEvent::Error("".into()).is_token());
    }

    #[test]
    fn test_stream_event_as_token() {
        assert_eq!(StreamEvent::Token("hello".into()).as_token(), Some("hello"));
        assert_eq!(StreamEvent::Started.as_token(), None);
        assert_eq!(StreamEvent::Completed.as_token(), None);
        assert_eq!(StreamEvent::Cancelled.as_token(), None);
        assert_eq!(StreamEvent::Error("".into()).as_token(), None);
    }

    #[test]
    fn test_stream_event_as_error() {
        assert_eq!(StreamEvent::Error("test err".into()).as_error(), Some("test err"));
        assert_eq!(StreamEvent::Started.as_error(), None);
        assert_eq!(StreamEvent::Token("".into()).as_error(), None);
        assert_eq!(StreamEvent::Completed.as_error(), None);
        assert_eq!(StreamEvent::Cancelled.as_error(), None);
    }

    #[test]
    fn test_stream_event_is_success() {
        assert!(StreamEvent::Started.is_success());
        assert!(StreamEvent::Token("".into()).is_success());
        assert!(StreamEvent::Completed.is_success());
        assert!(!StreamEvent::Cancelled.is_success());
        assert!(!StreamEvent::Error("".into()).is_success());
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
}
