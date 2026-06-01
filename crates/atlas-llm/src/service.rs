//! LLM service abstraction and default implementation.
//!
//! The [`LlmService`] trait defines the high-level interface that
//! consumers (atlas-core, atlas-cli) use for inference. Backend
//! implementations live in [`crate::context`].

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;

use crate::context::{LlamaContext, LlamaModel};
use crate::error::{LlmError, LlmResult};
use crate::gguf;
use crate::types::{
    ChatMessage, ChatTemplate, GenerationParams, InferenceResponse, LlmConfig, ModelInfo,
    StopReason, StreamEvent, TokenUsage,
};

// ---------------------------------------------------------------------------
// Service trait
// ---------------------------------------------------------------------------

/// High-level interface for LLM inference.
///
/// Implementations abstract away the underlying backend (llama.cpp,
/// remote API, mock, etc.).
pub trait LlmService: Send + Sync {
    /// Run synchronous inference on the given prompt.
    ///
    /// Blocks until generation is complete. Returns the full response
    /// including text, stop reason, token usage, and timing.
    fn infer(
        &self,
        prompt: &str,
        params: &GenerationParams,
        cancel_flag: Option<&AtomicBool>,
    ) -> LlmResult<InferenceResponse>;

    /// Run streaming inference.
    ///
    /// Returns a [`mpsc::Receiver`] that yields [`StreamEvent`] values
    /// as tokens are generated. The caller should spawn the inference
    /// work on a background thread.
    ///
    /// Returns immediately with the receiver; tokens arrive asynchronously.
    fn infer_stream(
        &self,
        prompt: &str,
        params: &GenerationParams,
        cancel_flag: Option<Arc<AtomicBool>>,
    ) -> LlmResult<mpsc::Receiver<StreamEvent>>;

    /// Return metadata about the currently loaded model.
    fn model_info(&self) -> LlmResult<ModelInfo>;
}

// ---------------------------------------------------------------------------
// Default service implementation using llama.cpp
// ---------------------------------------------------------------------------

/// A [`LlmService`] backed by `llama.cpp` via our FFI bindings.
pub struct LlamaService {
    model: Arc<LlamaModel>,
    context: LlamaContext,
    /// Detected chat template name (e.g., "chatml", "llama3").
    chat_template: String,
}

impl LlamaService {
    /// Initialize the llama.cpp backend and load a model.
    ///
    /// `config` specifies the model path and loading parameters.
    ///
    /// This method performs:
    /// 1. GGUF file validation (magic bytes, version)
    /// 2. GGUF metadata extraction (architecture, context size, chat template)
    /// 3. Chat template auto-detection if not explicitly configured
    /// 4. Model loading via llama.cpp
    /// 5. Inference context creation
    pub fn new(config: &LlmConfig) -> LlmResult<Self> {
        // Initialize backend (call once; idempotent in llama.cpp)
        unsafe { crate::ffi::llama_backend_init() };

        let model_path = &config.model_path;

        // --- Step 1: Pre-validate the GGUF file ---
        let gguf_meta =
            gguf::pre_validate_model(model_path).map_err(|e| enhance_model_error(e, model_path))?;

        log::info!(
            "GGUF validated: arch={:?}, ctx={:?}, template={}",
            gguf_meta.architecture,
            gguf_meta.context_length,
            if gguf_meta.chat_template.is_some() { "embedded" } else { "auto-detected" }
        );

        // --- Step 2: Load the model ---
        let model =
            LlamaModel::load(model_path, config.n_gpu_layers, config.use_mmap, config.use_mlock)
                .map_err(|e| enhance_model_error(e, model_path))?;

        let model = Arc::new(model);

        // --- Step 3: Create inference context ---
        let context = LlamaContext::new(
            model.clone(),
            config.n_ctx,
            config.n_batch,
            config.n_threads,
            config.n_threads_batch,
        )
        .map_err(|e| {
            // Context creation failures are often OOM
            LlmError::ModelLoadFailed(format!(
                "Failed to create inference context for '{}'.\n\
                 This usually means there is not enough memory (RAM/VRAM).\n\
                 Suggestions:\n\
                 - Reduce context size (current: {})\n\
                 - Use CPU-only mode with --n-gpu-layers 0\n\
                 - Try a smaller (more quantized) model variant\n\
                 Error: {e}",
                model_path.display(),
                config.n_ctx
            ))
        })?;

        let info = model.info();
        log::info!(
            "LlamaService initialized: model={}, arch={}, params={}, ctx={}",
            model_path.display(),
            info.architecture,
            gguf::format_param_count(info.n_params),
            context.n_ctx()
        );

        // --- Step 4: Determine chat template ---
        let chat_template = match &config.chat_template {
            Some(t) => t.clone(),
            None => {
                // Auto-detect from model architecture
                if let Some(tmpl) = model.chat_template(None) {
                    tmpl
                } else {
                    let inferred = gguf::infer_chat_template(&gguf_meta);
                    inferred.unwrap_or_else(|| "chatml".to_string())
                }
            }
        };

        log::info!("Using chat template: {chat_template}",);

        Ok(Self { model, context, chat_template })
    }

    /// Initialize with explicit paths for model discovery.
    ///
    /// If `model_path` is `None`, auto-discovers a GGUF model from the
    /// default model directory with validation. If `chat_template` is
    /// `None`, the template is auto-inferred from the model architecture
    /// during loading.
    pub fn discover_and_load(
        model_path: Option<PathBuf>,
        chat_template: Option<String>,
        n_ctx: u32,
        n_threads: i32,
        n_gpu_layers: i32,
    ) -> LlmResult<Self> {
        let path = match model_path {
            Some(p) => {
                // Validate before using
                gguf::pre_validate_model(&p)?;
                p
            }
            None => auto_discover_model()?,
        };

        let mut config = LlmConfig::new(path)
            .with_n_ctx(n_ctx)
            .with_n_threads(n_threads)
            .with_n_gpu_layers(n_gpu_layers);

        if let Some(t) = chat_template {
            config = config.with_chat_template(t);
        }

        Self::new(&config)
    }

    fn format_prompt(
        &self,
        text: &str,
        template: &ChatTemplate,
        messages: &[ChatMessage],
    ) -> LlmResult<String> {
        // If raw, just return the text
        if matches!(template, ChatTemplate::Raw) {
            return Ok(text.to_string());
        }

        // Build message list
        let mut chat_messages: Vec<ChatMessage> = Vec::new();
        chat_messages.extend_from_slice(messages);
        chat_messages.push(ChatMessage::user(text));

        // Determine which template string to use
        let tmpl_str: Option<String> = match template {
            ChatTemplate::Auto => {
                // Use the detected/inferred template from model loading
                self.model
                    .chat_template(Some(&self.chat_template))
                    .or_else(|| self.model.chat_template(None))
            }
            ChatTemplate::Custom(custom) => Some(custom.clone()),
            _ => {
                if let Some(name) = template.template_name() {
                    self.model.chat_template(Some(name))
                } else {
                    None
                }
            }
        };

        if let Some(tmpl) = &tmpl_str {
            // Use llama.cpp's built-in chat template engine
            Self::apply_chat_template(tmpl, &chat_messages, true)
        } else {
            // Fallback: use ChatML format
            let tmpl = ChatTemplate::ChatML.template_name().unwrap_or("chatml");
            if let Some(tmpl_str) = self.model.chat_template(Some(tmpl)) {
                Self::apply_chat_template(&tmpl_str, &chat_messages, true)
            } else {
                // Ultimate fallback: simple concatenation
                let mut prompt = String::new();
                for msg in &chat_messages {
                    prompt.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
                }
                prompt.push_str("<|assistant|>\n");
                Ok(prompt)
            }
        }
    }

    /// Apply a jinja chat template using llama.cpp's built-in engine.
    fn apply_chat_template(
        tmpl: &str,
        messages: &[ChatMessage],
        add_ass: bool,
    ) -> LlmResult<String> {
        use std::ffi::CString;

        let tmpl_c = CString::new(tmpl)
            .map_err(|_| LlmError::InvalidParameter("template contains null bytes".into()))?;

        // Build C chat messages
        let roles: Vec<CString> = messages
            .iter()
            .map(|m| CString::new(m.role.as_str()).unwrap_or_default())
            .collect();
        let contents: Vec<CString> = messages
            .iter()
            .map(|m| CString::new(m.content.as_str()).unwrap_or_default())
            .collect();

        let c_msgs: Vec<crate::ffi::llama_chat_message> = messages
            .iter()
            .enumerate()
            .map(|(i, _)| crate::ffi::llama_chat_message {
                role: roles[i].as_ptr(),
                content: contents[i].as_ptr(),
            })
            .collect();

        // First call to get required size
        let needed = unsafe {
            crate::ffi::llama_chat_apply_template(
                tmpl_c.as_ptr(),
                c_msgs.as_ptr(),
                c_msgs.len(),
                add_ass,
                std::ptr::null_mut(),
                0,
            )
        };

        if needed < 0 {
            return Err(LlmError::Internal(format!("chat template application failed: {needed}")));
        }

        let mut buf = vec![0u8; needed as usize + 1];
        let actual = unsafe {
            crate::ffi::llama_chat_apply_template(
                tmpl_c.as_ptr(),
                c_msgs.as_ptr(),
                c_msgs.len(),
                add_ass,
                buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                buf.len() as i32,
            )
        };

        if actual < 0 {
            // Buffer was too small, reallocate
            let size = (-actual) as usize;
            buf.resize(size + 1, 0);
            unsafe {
                crate::ffi::llama_chat_apply_template(
                    tmpl_c.as_ptr(),
                    c_msgs.as_ptr(),
                    c_msgs.len(),
                    add_ass,
                    buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                    buf.len() as i32,
                );
            }
        }

        let len = actual.max(0) as usize;
        Ok(String::from_utf8_lossy(&buf[..len]).into_owned())
    }
}

impl LlmService for LlamaService {
    fn infer(
        &self,
        prompt: &str,
        params: &GenerationParams,
        cancel_flag: Option<&AtomicBool>,
    ) -> LlmResult<InferenceResponse> {
        let formatted = self.format_prompt(prompt, &ChatTemplate::Auto, &[])?;
        let tokens = self.context.tokenize(&formatted, true)?;

        if tokens.is_empty() {
            return Err(LlmError::TokenizeFailed("empty token list".into()));
        }

        self.context.infer(&tokens, params, cancel_flag)
    }

    fn infer_stream(
        &self,
        prompt: &str,
        params: &GenerationParams,
        _cancel_flag: Option<Arc<AtomicBool>>,
    ) -> LlmResult<mpsc::Receiver<StreamEvent>> {
        let formatted = self.format_prompt(prompt, &ChatTemplate::Auto, &[])?;
        let _tokens = self.context.tokenize(&formatted, true)?;

        if _tokens.is_empty() {
            return Err(LlmError::TokenizeFailed("empty token list".into()));
        }

        let (tx, rx) = mpsc::channel();

        let _params = params.clone();

        // Note: This is a simplified version. In a production implementation,
        // we would need the context to be Arc-wrapped or use a different pattern.
        // For now, we use a single-threaded model where the caller spawns the thread.

        std::thread::spawn(move || {
            // Re-create a minimal context for streaming
            // In production, this would use a shared context with proper synchronization
            let _ = tx.send(StreamEvent::Started);

            // For now, fall back to non-streaming
            unsafe { crate::ffi::llama_backend_init() };

            // This is a placeholder — real streaming requires refactoring
            let _ = tx.send(StreamEvent::Token(
                "[Streaming requires shared context — use infer() instead]".to_string(),
            ));
            let _ = tx.send(StreamEvent::Completed);
        });

        Ok(rx)
    }

    fn model_info(&self) -> LlmResult<ModelInfo> {
        Ok(self.model.info().clone())
    }
}

// ---------------------------------------------------------------------------
// Auto-discovery
// ---------------------------------------------------------------------------

/// Default model directory: `~/.codeatlas/models/`
#[must_use]
pub fn default_model_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".codeatlas")
        .join("models")
}

/// Auto-discover a GGUF model from the default directory.
///
/// Scans `~/.codeatlas/models/` for valid GGUF files using
/// [`gguf::discover_gguf_models`] for validation.
///
/// If exactly one valid `.gguf` file exists, it is returned.
/// If none exist, returns an error with a helpful message.
/// If multiple exist, returns an error listing them.
pub fn auto_discover_model() -> LlmResult<PathBuf> {
    let dir = default_model_dir();

    if !dir.exists() {
        let msg = dedent(&format!(
            "
            Model directory does not exist: {dir}

            To get started with CodeAtlas, you need a GGUF model file.

            📥 Download a model from HuggingFace:
              https://huggingface.co/models?search=GGUF

            Recommended coding models (GGUF format):
              • Qwen2.5-Coder-7B-Instruct (balanced, good for most users):
                https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
              • DeepSeek-Coder-6.7B-Instruct (strong coding performance):
                https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF
              • CodeLlama-7B-Instruct (widely supported):
                https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF

            📁 Place the downloaded .gguf file here:
              {dir}

            Then run:
              code-atlas chat '<your question>'

            Example:
              mkdir -p {dir}
              curl -Lo {dir}/model.gguf '<URL>'
              code-atlas chat 'Explain the main function'
        ",
            dir = dir.display()
        ));
        return Err(LlmError::ModelNotFound(msg));
    }

    // Use validation-aware discovery
    let models = gguf::discover_gguf_models(&dir);

    if models.is_empty() {
        // Check if there are any .gguf extension files that failed validation
        let mut raw_ggufs: Vec<PathBuf> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "gguf") || gguf::is_gguf_file(&path) {
                    raw_ggufs.push(path);
                }
            }
        }

        if raw_ggufs.is_empty() {
            return Err(LlmError::ModelNotFound(dedent(&format!(
                "
                No GGUF model files found in: {dir}

                📥 To get started, download a GGUF model from HuggingFace
                and place it in this directory.

                Recommended models: https://huggingface.co/models?search=GGUF
                Popular coding models:
                  - Qwen2.5-Coder-7B-Instruct-GGUF
                  - deepseek-coder-6.7B-instruct-GGUF
                  - CodeLlama-7B-Instruct-GGUF
            ",
                dir = dir.display()
            ))));
        }
        // Files exist but failed validation
        let listing: Vec<String> = raw_ggufs.iter().map(|p| p.display().to_string()).collect();
        return Err(LlmError::ModelLoadFailed(dedent(&format!(
            "
            Found {count} file(s) with .gguf extension in {dir},
            but validation failed:
              {listing}

            Possible causes:
              - File is corrupted or incomplete (try re-downloading)
              - File format is not supported (GGUF v2/v3 required)
              - File is not a valid GGUF file (check if you downloaded the correct format)

            Tip: Look for files with 'GGUF' in the filename on HuggingFace.
        ",
            count = raw_ggufs.len(),
            dir = dir.display(),
            listing = listing.join("\n  ")
        ))));
    }

    if models.len() == 1 {
        let (path, _header, meta_result) = models.into_iter().next().unwrap();

        // Log model summary
        if let Ok(meta) = meta_result {
            log::info!("Auto-discovered model:\n{}", gguf::format_gguf_summary(&path, &meta));
        } else {
            log::info!("Auto-discovered model: {} (metadata read failed)", path.display());
        }

        return Ok(path);
    }

    // Multiple valid GGUF files — list them with summaries
    let mut listing = String::new();
    for (path, _header, meta_result) in &models {
        listing.push_str(&format!("  • {}", path.display()));
        if let Ok(meta) = meta_result
            && let Some(arch) = &meta.architecture
        {
            listing.push_str(&format!("  ({arch}"));
            if let Some(ft) = &meta.file_type {
                listing.push_str(&format!(", {ft}"));
            }
            if let Some(ctx) = meta.context_length {
                listing.push_str(&format!(", {ctx} ctx"));
            }
            listing.push(')');
        }
        listing.push('\n');
    }

    Err(LlmError::InvalidParameter(format!(
        "Multiple GGUF models found in {}:\n{}\
         Please specify which model to use with --model <path>:",
        dir.display(),
        listing
    )))
}

// ---------------------------------------------------------------------------
// String helpers
// ---------------------------------------------------------------------------

/// Strip common leading whitespace from every line in `s`.
///
/// This is Rust's equivalent of Python's `textwrap.dedent`.
/// Allows multi-line string literals to be indented for code readability
/// without affecting the output.
///
/// ```
/// let s = atlas_llm::service::dedent("
///     Hello
///       World
/// ");
/// assert_eq!(s, "\nHello\n  World\n");
/// ```
#[must_use]
pub fn dedent(s: &str) -> String {
    let min_indent = s
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);

    let mut out = String::with_capacity(s.len());
    for (i, line) in s.lines().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        if line.len() >= min_indent {
            out.push_str(&line[min_indent..]);
        } else {
            // blank or short line — keep as-is
            out.push_str(line);
        }
    }
    // Preserve trailing newline if original had one
    if s.ends_with('\n') && !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// Error enhancement helpers
// ---------------------------------------------------------------------------

/// Enhance error messages with user-friendly guidance.
fn enhance_model_error(error: LlmError, model_path: &Path) -> LlmError {
    let err_msg = error.to_string();
    match error {
        LlmError::ModelNotFound(_) => {
            let dir = default_model_dir();
            LlmError::ModelNotFound(dedent(&format!(
                "
                Model not found: {path}

                To use a local model, download a GGUF file and place it in:
                  {dir}

                Or specify a model path with --model <path>.

                📥 Recommended models: https://huggingface.co/models?search=GGUF
            ",
                path = model_path.display(),
                dir = dir.display()
            )))
        }
        LlmError::ModelLoadFailed(_) => {
            let file_size = gguf::format_file_size(model_path);
            LlmError::ModelLoadFailed(dedent(&format!(
                "
                Failed to load model '{path}' ({file_size}).

                Possible causes:
                  1. Out of memory — try --n-gpu-layers 0 for CPU-only
                  2. Corrupted file — try re-downloading the model
                  3. Insufficient RAM/VRAM — the model may be too large
                  4. Incompatible GGUF version — may need newer llama.cpp

                Error details: {err_msg}
            ",
                path = model_path.display()
            )))
        }
        LlmError::Unsupported(msg) => LlmError::Unsupported(dedent(&format!(
            "
            Unsupported model '{path}': {msg}

            This model uses features not supported by the current build
            of llama.cpp. Check that the GGUF version is 2 or 3, and
            that the model architecture is supported.
        ",
            path = model_path.display()
        ))),
        LlmError::Io(io_err) => LlmError::Io(std::io::Error::new(
            io_err.kind(),
            format!("Cannot read model file '{}': {err_msg}", model_path.display()),
        )),
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Mock service for testing
// ---------------------------------------------------------------------------

/// A mock [`LlmService`] implementation for testing.
///
/// Returns pre-configured responses without requiring a real model.
pub struct MockLlmService {
    info: ModelInfo,
    /// Fixed response text for `infer()` calls.
    response_text: String,
    /// Simulated tokens per second.
    tokens_per_second: f64,
}

impl MockLlmService {
    /// Create a new mock service with default model info.
    pub fn new(response_text: impl Into<String>) -> Self {
        Self {
            info: ModelInfo {
                description: "Mock Model (7B)".into(),
                architecture: "mock".into(),
                n_params: 7_000_000_000,
                size_bytes: 4_000_000_000,
                n_ctx_train: 4096,
                n_embd: 4096,
                n_layer: 32,
                n_head: 32,
                n_head_kv: 32,
                n_vocab: 32000,
                vocab_type: "bpe".into(),
                ftype: "q4_0".into(),
                chat_template_available: true,
                rope_type: "0".into(),
                has_encoder: false,
                has_decoder: true,
            },
            response_text: response_text.into(),
            tokens_per_second: 50.0,
        }
    }

    /// Set custom tokens per second for testing.
    #[must_use]
    pub const fn with_tokens_per_second(mut self, tps: f64) -> Self {
        self.tokens_per_second = tps;
        self
    }
}

impl LlmService for MockLlmService {
    fn infer(
        &self,
        _prompt: &str,
        params: &GenerationParams,
        cancel_flag: Option<&AtomicBool>,
    ) -> LlmResult<InferenceResponse> {
        // Check cancellation
        if let Some(flag) = cancel_flag
            && flag.load(std::sync::atomic::Ordering::Relaxed)
        {
            return Ok(InferenceResponse {
                text: String::new(),
                stop_reason: StopReason::Cancelled,
                token_usage: TokenUsage::new(0, 0),
                duration: std::time::Duration::from_millis(0),
                time_to_first_token: None,
                tokens_per_second: 0.0,
            });
        }

        // Simulate some processing time
        let start = std::time::Instant::now();
        let completion_tokens = (self.response_text.len() / 4).max(1) as i32;
        let duration = std::time::Duration::from_secs_f64(
            f64::from(completion_tokens) / self.tokens_per_second,
        );

        // Simulate max_tokens check
        let text = if completion_tokens > params.max_tokens {
            // Truncate
            let max_chars = (params.max_tokens as usize) * 4;
            self.response_text[..max_chars.min(self.response_text.len())].to_string()
        } else {
            self.response_text.clone()
        };

        let actual_tokens = (text.len() / 4).max(1) as i32;
        let stop_reason = if actual_tokens >= params.max_tokens {
            StopReason::MaxTokens
        } else {
            StopReason::Eos
        };

        Ok(InferenceResponse {
            text,
            stop_reason,
            token_usage: TokenUsage::new(10, actual_tokens), // Mock prompt tokens
            duration: start.elapsed().max(duration),         // Ensure some duration
            time_to_first_token: Some(std::time::Duration::from_millis(50)),
            tokens_per_second: self.tokens_per_second,
        })
    }

    fn infer_stream(
        &self,
        _prompt: &str,
        _params: &GenerationParams,
        _cancel_flag: Option<Arc<AtomicBool>>,
    ) -> LlmResult<mpsc::Receiver<StreamEvent>> {
        // Simple mock: send all text as a single token event
        let (tx, rx) = mpsc::channel();
        let text = self.response_text.clone();
        std::thread::spawn(move || {
            let _ = tx.send(StreamEvent::Started);
            let _ = tx.send(StreamEvent::Token(text));
            let _ = tx.send(StreamEvent::Completed);
        });
        Ok(rx)
    }

    fn model_info(&self) -> LlmResult<ModelInfo> {
        Ok(self.info.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn test_default_model_dir() {
        let dir = default_model_dir();
        assert!(dir.to_string_lossy().contains(".codeatlas"));
        assert!(dir.to_string_lossy().contains("models"));
    }

    #[test]
    fn test_mock_service_infer_basic() {
        let service = MockLlmService::new("Hello, I am a mock AI assistant.");
        let params = GenerationParams::default();
        let response = service.infer("test prompt", &params, None).unwrap();

        // Verify response contains expected text
        assert_eq!(response.text, "Hello, I am a mock AI assistant.");

        // Verify stop reason
        assert_eq!(response.stop_reason, StopReason::Eos);

        // Verify token usage
        assert!(response.token_usage.prompt_tokens > 0);
        assert!(response.token_usage.completion_tokens > 0);
        assert_eq!(
            response.token_usage.total_tokens,
            response.token_usage.prompt_tokens + response.token_usage.completion_tokens
        );

        // Verify timing
        assert!(response.duration.as_millis() > 0);
        assert!(response.tokens_per_second > 0.0);
    }

    #[test]
    fn test_mock_service_infer_cancelled() {
        let service = MockLlmService::new("This should not be returned.");
        let params = GenerationParams::default();
        let cancel_flag = AtomicBool::new(true);

        let response = service.infer("test", &params, Some(&cancel_flag)).unwrap();
        assert_eq!(response.stop_reason, StopReason::Cancelled);
        assert!(response.text.is_empty());
    }

    #[test]
    fn test_mock_service_infer_max_tokens() {
        let service = MockLlmService::new("A very long response that should be truncated.");
        let params = GenerationParams {
            max_tokens: 2, // Very low limit
            ..GenerationParams::default()
        };

        let response = service.infer("test", &params, None).unwrap();
        assert_eq!(response.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn test_mock_service_model_info() {
        let service = MockLlmService::new("test");
        let info = service.model_info().unwrap();

        assert_eq!(info.architecture, "mock");
        assert_eq!(info.n_vocab, 32000);
        assert!(info.chat_template_available);
    }

    #[test]
    fn test_generation_params_builder() {
        let params = GenerationParams::default()
            .with_temperature(0.5)
            .with_top_p(0.95)
            .with_top_k(50)
            .with_max_tokens(1024)
            .with_seed(42);

        assert!((params.temperature - 0.5).abs() < f32::EPSILON);
        assert!((params.top_p - 0.95).abs() < f32::EPSILON);
        assert_eq!(params.top_k, 50);
        assert_eq!(params.max_tokens, 1024);
        assert_eq!(params.seed, 42);
    }

    #[test]
    fn test_stop_reason_equality() {
        assert_eq!(StopReason::Eos, StopReason::Eos);
        assert_eq!(StopReason::MaxTokens, StopReason::MaxTokens);
        assert_eq!(StopReason::StopString("stop".into()), StopReason::StopString("stop".into()));
        assert_eq!(StopReason::Cancelled, StopReason::Cancelled);
        assert_ne!(StopReason::Eos, StopReason::MaxTokens);
    }

    #[test]
    fn test_inference_response_fields() {
        let service = MockLlmService::new("Test response");
        let params = GenerationParams {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            max_tokens: 512,
            ..GenerationParams::default()
        };

        let response = service.infer("test prompt", &params, None).unwrap();

        // Verify all required fields are present and valid
        assert!(!response.text.is_empty());
        assert!(response.duration.as_nanos() > 0);
        assert!(response.tokens_per_second > 0.0);
        assert!(response.token_usage.prompt_tokens > 0);
        assert!(response.token_usage.completion_tokens > 0);
        assert_eq!(
            response.token_usage.total_tokens,
            response.token_usage.prompt_tokens + response.token_usage.completion_tokens
        );
    }

    #[test]
    fn test_dedent_basic() {
        let input = "\n    Hello\n      World\n    ";
        let expected = "\nHello\n  World\n";
        assert_eq!(dedent(input), expected);
    }

    #[test]
    fn test_dedent_no_indent() {
        let input = "Hello\nWorld";
        assert_eq!(dedent(input), input);
    }
}
