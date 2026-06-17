//! Data model types for chatvcode-llm.
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
    ///
    /// When no model metadata is available, `Auto` behaves identically to
    /// `ChatML` to maximize portability. Callers that need a different
    /// fallback can explicitly select [`Self::Raw`] or another template.
    Auto,

    /// Raw text, no template formatting applied.
    Raw,

    /// `ChatML` format (used by many models).
    /// Format: `<|im_start|>role\ncontent<|im_end|>\n`
    ChatML,

    /// Llama 3 format.
    /// Format: `<|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
    Llama3,

    /// DeepSeek 2.5 / V3 format.
    /// Format: `<｜begin▁of▁sentence｜>system\n\n<｜User｜>content<｜Assistant｜>`
    DeepSeek,

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
            Self::DeepSeek => Some("deepseek"),
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

    /// Format a list of chat messages into a prompt string using this template.
    ///
    /// This is a pure-Rust implementation that does not require `llama.cpp` FFI.
    /// It supports `Auto` (falls back to `ChatML`), `Raw`, `ChatML`, `Llama3`,
    /// and `DeepSeek`.
    ///
    /// For `Custom` templates, this returns an error — callers should use
    /// the FFI-based `LlamaService::format_prompt` instead.
    ///
    /// # Arguments
    ///
    /// * `messages` — Ordered list of chat messages (system, user, assistant).
    /// * `add_generation_prompt` — If `true`, appends the assistant prefix
    ///   so the model continues generating from there.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::InvalidParameter`] for `Custom` templates (which
    /// require jinja evaluation via `llama.cpp`).
    pub fn format(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, crate::error::LlmError> {
        match self {
            Self::Raw => {
                // Raw: concatenate message contents with newlines
                let prompt: String = messages
                    .iter()
                    .map(|m| m.content.as_str())
                    .collect::<Vec<&str>>()
                    .join("\n\n");
                Ok(prompt)
            }
            Self::ChatML => Ok(Self::format_chatml(messages, add_generation_prompt)),
            Self::Llama3 => Ok(Self::format_llama3(messages, add_generation_prompt)),
            Self::DeepSeek => Ok(Self::format_deepseek(messages, add_generation_prompt)),
            Self::Auto => {
                // Auto falls back to ChatML — in production use, the model's
                // own template (discovered via GGUF metadata) is preferred,
                // but this provides a safe default. Users needing a different
                // fallback can select `Raw` or another template explicitly.
                Ok(Self::format_chatml(messages, add_generation_prompt))
            }
            Self::Custom(tmpl) => {
                // Custom jinja templates require llama.cpp's template engine.
                // We can't evaluate jinja in pure Rust, so return an error.
                Err(crate::error::LlmError::InvalidParameter(format!(
                    "Custom jinja templates require the llama.cpp template engine. \
                     Template: {tmpl}"
                )))
            }
        }
    }

    /// Format messages using ChatML template.
    ///
    /// ChatML format:
    /// ```text
    /// <|im_start|>system
    /// {content}<|im_end|>
    /// <|im_start|>user
    /// {content}<|im_end|>
    /// <|im_start|>assistant
    /// {content}<|im_end|>
    /// <|im_start|>assistant
    /// ```
    fn format_chatml(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let mut prompt = String::new();

        for msg in messages {
            prompt.push_str("<|im_start|>");
            prompt.push_str(&msg.role);
            prompt.push('\n');
            prompt.push_str(&msg.content);
            prompt.push_str("<|im_end|>\n");
        }

        if add_generation_prompt {
            prompt.push_str("<|im_start|>assistant\n");
        }

        prompt
    }

    /// Format messages using Llama 3 template.
    ///
    /// Llama 3 format:
    /// ```text
    /// <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    ///
    /// {content}<|eot_id|><|start_header_id|>user<|end_header_id|>
    ///
    /// {content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    ///
    /// {content}<|eot_id|>
    /// ```
    fn format_llama3(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let mut prompt = String::from("<|begin_of_text|>");

        for msg in messages {
            prompt.push_str("<|start_header_id|>");
            prompt.push_str(&msg.role);
            prompt.push_str("<|end_header_id|>\n\n");
            prompt.push_str(&msg.content);
            prompt.push_str("<|eot_id|>");
        }

        if add_generation_prompt {
            prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }

        prompt
    }

    /// Format messages using DeepSeek 2.5 / V3 template.
    ///
    /// This mirrors the `deepseek3` template used by `llama.cpp`:
    /// ```text
    /// <｜begin▁of▁sentence｜>system content
    ///
    /// <｜User｜>user content<｜Assistant｜>assistant content<｜end▁of▁sentence｜>
    /// <｜Assistant｜>
    /// ```
    fn format_deepseek(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let mut prompt = String::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    prompt.push_str("<｜begin▁of▁sentence｜>");
                    prompt.push_str(&msg.content);
                    prompt.push_str("\n\n");
                }
                "user" => {
                    prompt.push_str("<｜User｜>");
                    prompt.push_str(&msg.content);
                }
                _ => {
                    // assistant and any other role
                    prompt.push_str("<｜Assistant｜>");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<｜end▁of▁sentence｜>");
                }
            }
        }

        if add_generation_prompt {
            prompt.push_str("<｜Assistant｜>");
        }

        prompt
    }
}

// ---------------------------------------------------------------------------
// Chat prompt builder (single-turn, RAG-aware)
// ---------------------------------------------------------------------------

/// Builder for constructing chat prompts with optional system prompt and
/// retrieved context (RAG).
///
/// This is the primary interface for building prompts for single-turn
/// question-answering. It supports:
/// - System prompt injection
/// - Context injection from code retrieval results
/// - Token budget management for context snippets
/// - Multiple chat template formats
///
/// # Example
///
/// ```ignore
/// use chatvcode_llm::{ChatPromptBuilder, ChatTemplate};
///
/// let prompt = ChatPromptBuilder::new(ChatTemplate::ChatML)
///     .system_prompt("You are a helpful coding assistant.")
///     .user_question("What does the `main` function do?")
///     .context("fn main() { println!(\"hello\"); }")
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ChatPromptBuilder {
    /// Chat template to use for formatting.
    template: ChatTemplate,
    /// System prompt (prepended as a system message).
    system_prompt: Option<String>,
    /// User question.
    user_question: Option<String>,
    /// Retrieved context snippets to inject before the question.
    context: Vec<String>,
    /// Maximum token budget for context. 0 = unlimited.
    context_token_budget: usize,
    /// Whether to add the assistant generation prompt at the end.
    add_generation_prompt: bool,
    /// Additional messages to include (for future multi-turn support).
    /// In M3, this is reserved but not actively used.
    history: Vec<ChatMessage>,
}

impl ChatPromptBuilder {
    /// Create a new prompt builder with the given template.
    ///
    /// Defaults:
    /// - No system prompt
    /// - No context
    /// - No token budget (unlimited)
    /// - `add_generation_prompt` = `true`
    #[must_use]
    pub fn new(template: ChatTemplate) -> Self {
        Self {
            template,
            system_prompt: None,
            user_question: None,
            context: Vec::new(),
            context_token_budget: 0,
            add_generation_prompt: true,
            history: Vec::new(),
        }
    }

    /// Set the system prompt.
    ///
    /// The system prompt is prepended as a `system` role message,
    /// establishing the model's behavior and persona.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the user question.
    ///
    /// This is the main question the user is asking.
    pub fn user_question(mut self, question: impl Into<String>) -> Self {
        self.user_question = Some(question.into());
        self
    }

    /// Add a context snippet.
    ///
    /// Context snippets are injected into the prompt to provide
    /// retrieved code or other reference material (RAG).
    pub fn context(mut self, ctx: impl Into<String>) -> Self {
        self.context.push(ctx.into());
        self
    }

    /// Set multiple context snippets at once.
    pub fn context_snippets(mut self, snippets: Vec<String>) -> Self {
        self.context = snippets;
        self
    }

    /// Set the token budget for context snippets.
    ///
    /// When set to a non-zero value, context snippets are truncated
    /// to fit within approximately this many tokens. A rough heuristic
    /// of 4 characters per token is used for estimation.
    ///
    /// Set to 0 for unlimited context (default).
    #[must_use]
    pub const fn context_token_budget(mut self, budget: usize) -> Self {
        self.context_token_budget = budget;
        self
    }

    /// Set whether to add the assistant generation prompt.
    ///
    /// When `true` (default), the formatted prompt ends with the
    /// assistant prefix, ready for the model to continue.
    #[must_use]
    pub const fn add_generation_prompt(mut self, add: bool) -> Self {
        self.add_generation_prompt = add;
        self
    }

    /// Add a message to the conversation history.
    ///
    /// **Reserved for future multi-turn support.** In M3, this is
    /// not actively used but provides an extension point.
    pub fn message(mut self, msg: ChatMessage) -> Self {
        self.history.push(msg);
        self
    }

    /// Build the formatted prompt string.
    ///
    /// This constructs the full message sequence:
    /// 1. System prompt (if set)
    /// 2. Conversation history (reserved for multi-turn, currently empty)
    /// 3. Context-injected user question
    ///
    /// The user question is formatted with context as:
    /// ```text
    /// [Retrieved Context]
    /// --- snippet 1 ---
    /// {context_1}
    /// --- snippet 2 ---
    /// {context_2}
    /// ---
    ///
    /// {user_question}
    /// ```
    ///
    /// If no context is provided, the user question is used directly.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::InvalidParameter`] if no user question is set,
    /// or if the template cannot format the messages (e.g., Custom jinja).
    pub fn build(self) -> Result<String, crate::error::LlmError> {
        let question = self.user_question.ok_or_else(|| {
            crate::error::LlmError::InvalidParameter(
                "user question is required for ChatPromptBuilder".into(),
            )
        })?;

        // Construct the user message content, with optional context injection
        let user_content = if self.context.is_empty() {
            question
        } else {
            let mut content = String::new();
            content.push_str("[Retrieved Context]\n");

            // Apply token budget if set
            let mut remaining_budget = if self.context_token_budget > 0 {
                // Rough heuristic: ~4 chars per token
                self.context_token_budget.saturating_mul(4)
            } else {
                usize::MAX
            };

            for (i, snippet) in self.context.iter().enumerate() {
                let header = format!("--- snippet {} ---\n", i + 1);
                let budget_for_snippet = remaining_budget.saturating_sub(header.len());
                let truncated = if snippet.len() > budget_for_snippet {
                    let end = snippet.floor_char_boundary(budget_for_snippet);
                    &snippet[..end]
                } else {
                    snippet.as_str()
                };

                content.push_str(&header);
                content.push_str(truncated);
                content.push('\n');

                remaining_budget = remaining_budget.saturating_sub(header.len() + truncated.len());
                if remaining_budget == 0 {
                    content.push_str("... (context truncated due to token budget)\n");
                    break;
                }
            }

            content.push_str("---\n\n");
            content.push_str(&question);
            content
        };

        // Build message list
        let mut messages = Vec::new();

        if let Some(sys) = &self.system_prompt {
            messages.push(ChatMessage::system(sys.as_str()));
        }

        messages.extend(self.history);
        messages.push(ChatMessage::user(user_content));

        self.template.format(&messages, self.add_generation_prompt)
    }
}

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/// Estimate the number of tokens in a text string.
///
/// Uses a heuristic of ~4 characters per token, which is a reasonable
/// approximation for most Latin-script text. CJK characters and special
/// tokens may use fewer characters per token.
///
/// This is useful for:
/// - Pre-checking whether a prompt fits within a context window
/// - Deciding when to trim conversation history
/// - Budget management for RAG context injection
#[must_use]
pub fn token_estimate(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let total_chars = text.len();
    (total_chars + 3) / 4
}

/// Estimate the number of tokens for a list of chat messages.
///
/// Includes a small overhead (~4 tokens per message) for role markers
/// and template formatting overhead.
#[must_use]
pub fn token_estimate_messages(messages: &[ChatMessage]) -> usize {
    const OVERHEAD_PER_MSG: usize = 4;
    messages
        .iter()
        .map(|m| token_estimate(&m.content) + OVERHEAD_PER_MSG)
        .sum()
}

// ---------------------------------------------------------------------------
// Chat session (multi-turn conversation support)
// ---------------------------------------------------------------------------

/// A chat session that maintains conversation history across multiple turns.
///
/// Supports:
/// - Multi-turn conversation with context accumulation
/// - KV cache reuse for efficient multi-turn inference
/// - History trimming for context window management
/// - System prompt management
/// - Both synchronous and streaming inference
///
/// # Example
///
/// ```ignore
/// use chatvcode_llm::{ChatSession, ChatTemplate, GenerationParams, LlmService};
///
/// let mut session = ChatSession::new(ChatTemplate::ChatML)
///     .system_prompt("You are a helpful coding assistant.")
///     .max_context_tokens(4096);
///
/// // First turn
/// let response = session.chat("What is Rust?", &llm, &params)?;
/// println!("{}", response.text);
///
/// // Second turn (context from first turn is preserved)
/// let response = session.chat("How do lifetimes work?", &llm, &params)?;
/// println!("{}", response.text);
/// ```
#[derive(Debug, Clone)]
pub struct ChatSession {
    session_id: String,
    system_prompt: Option<String>,
    messages: Vec<ChatMessage>,
    max_history_turns: usize,
    template: ChatTemplate,
    estimated_tokens: usize,
    max_context_tokens: usize,
    reserve_for_response: usize,
    /// Opaque KV cache state from the last inference call.
    /// Passed to `LlmService::infer_cached` for multi-turn efficiency.
    kv_cache_state: crate::service::KvCacheState,
}

impl ChatSession {
    /// Create a new chat session with the given template.
    #[must_use]
    pub fn new(template: ChatTemplate) -> Self {
        Self {
            session_id: uuid_like_id(),
            system_prompt: None,
            messages: Vec::new(),
            max_history_turns: 0,
            template,
            estimated_tokens: 0,
            max_context_tokens: 0,
            reserve_for_response: 512,
            kv_cache_state: 0,
        }
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the maximum number of history turns to keep.
    ///
    /// A "turn" is one user + one assistant message.
    /// Setting to 0 means unlimited history.
    #[must_use]
    pub const fn max_history_turns(mut self, turns: usize) -> Self {
        self.max_history_turns = turns;
        self
    }

    /// Set the maximum context token count.
    ///
    /// When the estimated token count exceeds this, older messages
    /// are trimmed from the history.
    #[must_use]
    pub const fn max_context_tokens(mut self, tokens: usize) -> Self {
        self.max_context_tokens = tokens;
        self
    }

    /// Set the number of tokens to reserve for the model's response.
    ///
    /// When trimming history, this amount is subtracted from the
    /// `max_context_tokens` budget to leave room for the generated response.
    /// Default: 512.
    #[must_use]
    pub const fn reserve_for_response(mut self, tokens: usize) -> Self {
        self.reserve_for_response = tokens;
        self
    }

    /// Set the system prompt at runtime.
    ///
    /// Replaces any existing system prompt. Pass `None` to remove it.
    /// Invalidates the KV cache since the context changed.
    pub fn set_system_prompt(&mut self, prompt: Option<String>) {
        self.system_prompt = prompt;
        self.recalculate_tokens();
        self.kv_cache_state = 0;
    }

    /// Get the current system prompt.
    #[must_use]
    pub fn get_system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    /// Clear the system prompt.
    ///
    /// Invalidates the KV cache since the context changed.
    pub fn clear_system_prompt(&mut self) {
        self.system_prompt = None;
        self.recalculate_tokens();
        self.kv_cache_state = 0;
    }

    /// Add a message to the session.
    pub fn add_message(&mut self, msg: ChatMessage) {
        self.messages.push(msg);
        self.recalculate_tokens();
    }

    /// Add a user message to the session.
    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.messages.push(ChatMessage::user(content));
        self.recalculate_tokens();
    }

    /// Add an assistant message to the session.
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.messages.push(ChatMessage::assistant(content));
        self.recalculate_tokens();
    }

    /// Clear all conversation history.
    ///
    /// The system prompt is preserved. Call `clear_system_prompt()` separately
    /// to also remove the system prompt.
    ///
    /// The KV cache state is invalidated since the context changed.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.estimated_tokens = 0;
        self.kv_cache_state = 0;
    }

    /// Reset the session entirely, including the system prompt.
    ///
    /// The KV cache state is invalidated.
    pub fn reset(&mut self) {
        self.messages.clear();
        self.system_prompt = None;
        self.estimated_tokens = 0;
        self.kv_cache_state = 0;
    }

    /// Run synchronous inference for the next user message.
    ///
    /// This method:
    /// 1. Adds the user message to history
    /// 2. Trims history if needed to fit within `max_context_tokens`
    /// 3. Builds the prompt from the full conversation
    /// 4. Runs inference via the provided `LlmService`
    /// 5. Adds the assistant response to history
    /// 6. Returns the inference response
    ///
    /// # Errors
    ///
    /// Returns an error if prompt building or inference fails.
    pub fn chat(
        &mut self,
        user_message: &str,
        llm: &dyn crate::service::LlmService,
        params: &GenerationParams,
    ) -> Result<InferenceResponse, crate::error::LlmError> {
        self.add_user_message(user_message);
        self.trim_history();

        let prompt = self.build_prompt()?;

        let cancel_flag = std::sync::atomic::AtomicBool::new(false);
        let (response, new_cache_state) =
            llm.infer_cached(&prompt, params, self.kv_cache_state, Some(&cancel_flag))?;

        // Update KV cache state. If the backend returned 0 (not supported),
        // we keep 0 — no caching benefit, but also no harm.
        self.kv_cache_state = new_cache_state;

        self.add_assistant_message(&response.text);

        Ok(response)
    }

    /// Run streaming inference for the next user message.
    ///
    /// Similar to [`chat`](Self::chat), but returns a receiver for streaming
    /// token events. The caller must collect the tokens and add the complete
    /// assistant response to the session via [`add_assistant_message`](Self::add_assistant_message).
    ///
    /// # Note
    ///
    /// Unlike `chat()`, this method does NOT automatically add the assistant
    /// response to history. The caller should collect all tokens, concatenate
    /// them, and call `add_assistant_message()` with the full response text.
    ///
    /// # Errors
    ///
    /// Returns an error if prompt building or inference initiation fails.
    pub fn chat_stream(
        &mut self,
        user_message: &str,
        llm: &dyn crate::service::LlmService,
        params: &GenerationParams,
    ) -> Result<std::sync::mpsc::Receiver<StreamEvent>, crate::error::LlmError> {
        self.add_user_message(user_message);
        self.trim_history();

        let prompt = self.build_prompt()?;

        let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (rx, new_cache_state) =
            llm.infer_stream_cached(&prompt, params, self.kv_cache_state, Some(cancel_flag))?;

        // For streaming backends, the cache state comes from a new context
        // each time, so it's typically 0. Still track it for consistency.
        self.kv_cache_state = new_cache_state;

        Ok(rx)
    }

    /// Build the prompt for the current session state.
    ///
    /// Constructs the full message list including system prompt,
    /// history, and the next assistant generation prompt.
    ///
    /// # Errors
    ///
    /// Returns an error if the template cannot format the messages.
    pub fn build_prompt(&self) -> Result<String, crate::error::LlmError> {
        let mut messages = Vec::new();

        if let Some(sys) = &self.system_prompt {
            messages.push(ChatMessage::system(sys.as_str()));
        }

        let messages_to_include = self.trimmed_messages();
        messages.extend_from_slice(messages_to_include);

        self.template.format(&messages, true)
    }

    /// Build the prompt including an explicit user message at the end.
    ///
    /// Unlike `build_prompt()`, this does NOT require the user message to
    /// already be in the session history. Useful for previewing what the
    /// prompt would look like without modifying the session.
    ///
    /// # Errors
    ///
    /// Returns an error if the template cannot format the messages.
    pub fn build_prompt_with(
        &self,
        user_message: &str,
    ) -> Result<String, crate::error::LlmError> {
        let mut messages = Vec::new();

        if let Some(sys) = &self.system_prompt {
            messages.push(ChatMessage::system(sys.as_str()));
        }

        let messages_to_include = self.trimmed_messages();
        messages.extend_from_slice(messages_to_include);
        messages.push(ChatMessage::user(user_message));

        self.template.format(&messages, true)
    }

    /// Get the messages that would be included in the prompt after trimming.
    fn trimmed_messages(&self) -> &[ChatMessage] {
        if self.max_history_turns > 0 {
            let max_msgs = self.max_history_turns * 2;
            if self.messages.len() > max_msgs {
                return &self.messages[self.messages.len() - max_msgs..];
            }
        }
        &self.messages[..]
    }

    /// Recalculate the estimated token count.
    fn recalculate_tokens(&mut self) {
        self.estimated_tokens = self.estimate_tokens();
    }

    /// Estimate the token count for current messages.
    ///
    /// Uses [`token_estimate`] per message plus a per-message overhead
    /// for role markers and template formatting.
    fn estimate_tokens(&self) -> usize {
        let mut total = token_estimate_messages(&self.messages);
        if let Some(sys) = &self.system_prompt {
            total += token_estimate(sys) + 4;
        }
        total
    }

    /// Trim history to fit within the token budget.
    ///
    /// Removes oldest user+assistant pairs until the estimated
    /// token count fits within `max_context_tokens - reserve_for_response`.
    ///
    /// The system prompt is always preserved (never trimmed).
    /// If `max_context_tokens` is 0, no trimming is performed.
    pub fn trim_history(&mut self) {
        if self.max_context_tokens == 0 {
            return;
        }

        let budget = self.max_context_tokens.saturating_sub(self.reserve_for_response);
        let system_tokens = self
            .system_prompt
            .as_ref()
            .map_or(0, |s| token_estimate(s) + 4);
        let history_budget = budget.saturating_sub(system_tokens);

        if history_budget == 0 {
            self.messages.clear();
            self.recalculate_tokens();
            return;
        }

        while self.messages.len() > 1 {
            let current = self.estimate_message_tokens(&self.messages);
            if current <= history_budget {
                break;
            }
            self.messages.remove(0);
        }

        self.recalculate_tokens();
    }

    /// Estimate tokens for a specific slice of messages.
    fn estimate_message_tokens(&self, messages: &[ChatMessage]) -> usize {
        token_estimate_messages(messages)
    }

    /// Returns the number of messages in the session (excluding system prompt).
    #[must_use]
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns whether the session has no messages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Returns the estimated token count.
    #[must_use]
    pub fn estimated_tokens(&self) -> usize {
        self.estimated_tokens
    }

    /// Returns a reference to the session messages.
    #[must_use]
    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Returns the session ID.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Returns the chat template.
    #[must_use]
    pub fn template(&self) -> &ChatTemplate {
        &self.template
    }

    /// Returns the max context token limit.
    #[must_use]
    pub const fn context_token_limit(&self) -> usize {
        self.max_context_tokens
    }

    /// Returns the number of tokens reserved for the model response.
    #[must_use]
    pub const fn response_token_reserve(&self) -> usize {
        self.reserve_for_response
    }

    /// Returns the number of complete turns (user+assistant pairs).
    #[must_use]
    pub fn turn_count(&self) -> usize {
        let user_count = self.messages.iter().filter(|m| m.role == "user").count();
        let assistant_count = self.messages.iter().filter(|m| m.role == "assistant").count();
        user_count.min(assistant_count)
    }

    /// Returns the current KV cache state for multi-turn inference.
    ///
    /// Zero means no cache is active. Non-zero values are opaque
    /// counters managed by the LLM backend.
    #[must_use]
    pub fn kv_cache_state(&self) -> crate::service::KvCacheState {
        self.kv_cache_state
    }

    /// Invalidate the KV cache state.
    ///
    /// Call this when the conversation context has been modified outside
    /// of `chat()` / `chat_stream()` (e.g., after adding or removing
    /// messages manually). The next inference call will start fresh.
    pub fn invalidate_kv_cache(&mut self) {
        self.kv_cache_state = 0;
    }

    /// Serialize the session state to JSON for persistence.
    ///
    /// Returns a JSON string containing the session ID, system prompt,
    /// and all messages. The chat template is NOT serialized (it should
    /// be re-specified when restoring).
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String, crate::error::LlmError> {
        let state = SessionState {
            session_id: self.session_id.clone(),
            system_prompt: self.system_prompt.clone(),
            messages: self.messages.clone(),
        };
        serde_json::to_string_pretty(&state).map_err(|e| {
            crate::error::LlmError::Internal(format!("Failed to serialize session: {e}"))
        })
    }

    /// Restore a session from a JSON string.
    ///
    /// The chat template must be provided separately since it is not
    /// serialized.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    pub fn from_json(
        json: &str,
        template: ChatTemplate,
    ) -> Result<Self, crate::error::LlmError> {
        let state: SessionState = serde_json::from_str(json).map_err(|e| {
            crate::error::LlmError::Internal(format!("Failed to deserialize session: {e}"))
        })?;

        let mut session = Self {
            session_id: state.session_id,
            system_prompt: state.system_prompt,
            messages: state.messages,
            max_history_turns: 0,
            template,
            estimated_tokens: 0,
            max_context_tokens: 0,
            reserve_for_response: 512,
            kv_cache_state: 0,
        };
        session.recalculate_tokens();
        Ok(session)
    }
}

/// Serializable session state for persistence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SessionState {
    session_id: String,
    system_prompt: Option<String>,
    messages: Vec<ChatMessage>,
}

/// Generate a simple unique ID for sessions.
///
/// NOT a full UUID implementation — just enough for uniqueness.
fn uuid_like_id() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("session-{:x}-{:x}", now.as_secs(), now.subsec_nanos())
}

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
