//! LLM service abstraction and default implementation.
//!
//! The [`LlmService`] trait defines the high-level interface that
//! consumers (chatvcode-core, chatvcode-cli) use for inference. Backend
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

/// Session state key for tracking KV cache between multi-turn calls.
///
/// An opaque token returned by [`LlmService::infer_cached`] that identifies
/// the cached state for a specific session. Backends that do not support
/// KV cache reuse return zero, indicating a fresh state each time.
pub type KvCacheState = i32;

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

    /// Run synchronous inference reusing the KV cache from a previous turn.
    ///
    /// `cache_state` contains opaque state from a previous `infer_cached` call
    /// (or zero for the first turn). The returned tuple contains the inference
    /// response and a new cache state to pass to the next call.
    ///
    /// The default implementation ignores the cache and delegates to
    /// [`infer`](LlmService::infer). Backends with KV cache support
    /// (e.g., [`LlamaService`]) override this for efficient multi-turn
    /// inference.
    fn infer_cached(
        &self,
        prompt: &str,
        params: &GenerationParams,
        _cache_state: KvCacheState,
        cancel_flag: Option<&AtomicBool>,
    ) -> LlmResult<(InferenceResponse, KvCacheState)> {
        // Default: ignore cache, return zero state
        self.infer(prompt, params, cancel_flag).map(|r| (r, 0))
    }

    /// Run streaming inference reusing the KV cache from a previous turn.
    ///
    /// Like [`infer_cached`](LlmService::infer_cached), but for streaming.
    /// The returned tuple includes the receiver and a new cache state.
    ///
    /// The default implementation ignores the cache and delegates to
    /// [`infer_stream`](LlmService::infer_stream).
    fn infer_stream_cached(
        &self,
        prompt: &str,
        params: &GenerationParams,
        _cache_state: KvCacheState,
        cancel_flag: Option<Arc<AtomicBool>>,
    ) -> LlmResult<(mpsc::Receiver<StreamEvent>, KvCacheState)> {
        // Default: ignore cache, return zero state
        self.infer_stream(prompt, params, cancel_flag).map(|r| (r, 0))
    }

    /// Run batch inference on multiple prompts.
    ///
    /// Processes each prompt sequentially (or in parallel if the backend
    /// supports it) and returns a vector of responses. The default
    /// implementation processes prompts sequentially using [`infer`].
    ///
    /// # Arguments
    ///
    /// * `prompts` — Slice of prompt strings to process.
    /// * `params` — Generation parameters applied to all prompts.
    /// * `cancel_flag` — Optional cancellation flag checked between prompts.
    fn infer_batch(
        &self,
        prompts: &[&str],
        params: &GenerationParams,
        cancel_flag: Option<&AtomicBool>,
    ) -> LlmResult<Vec<InferenceResponse>> {
        let mut results = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            if let Some(flag) = cancel_flag
                && flag.load(std::sync::atomic::Ordering::Relaxed)
            {
                break;
            }
            results.push(self.infer(prompt, params, cancel_flag)?);
        }
        Ok(results)
    }

    /// Return metadata about the currently loaded model.
    fn model_info(&self) -> LlmResult<ModelInfo>;
}

// ---------------------------------------------------------------------------
// Default service implementation using llama.cpp
// ---------------------------------------------------------------------------

/// Parameters needed to create a new `LlamaContext`.
///
/// Stored to allow creating fresh contexts for streaming inference.
#[derive(Debug, Clone)]
struct ContextParams {
    n_ctx: u32,
    n_batch: u32,
    n_threads: i32,
    n_threads_batch: i32,
}

/// A [`LlmService`] backed by `llama.cpp` via our FFI bindings.
///
/// This is the primary production implementation of the [`LlmService`] trait.
/// It handles:
///
/// - GGUF file validation and metadata extraction
/// - Model loading with configurable GPU offload and memory mapping
/// - Chat template auto-detection from model metadata
/// - Synchronous and streaming inference
/// - KV cache reuse for multi-turn conversations
///
/// # Thread Safety
///
/// `LlamaService` is `Send + Sync`. However, inference methods take `&self`
/// and internally use the single `LlamaContext`, so concurrent inference
/// calls will block each other. For parallel inference, create multiple
/// service instances.
///
/// # Example
///
/// ```ignore
/// use chatvcode_llm::{LlamaService, LlmConfig, LlmService, GenerationParams};
///
/// let config = LlmConfig::new("model.gguf").with_n_ctx(8192);
/// let service = LlamaService::new(&config)?;
///
/// let response = service.infer("Hello", &GenerationParams::default(), None)?;
/// println!("{} ({:.1} tok/s)", response.text, response.tokens_per_second);
/// ```
pub struct LlamaService {
    model: Arc<LlamaModel>,
    context: LlamaContext,
    /// Parameters used to create the context (for streaming).
    ctx_params: ContextParams,
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
        // Suppress verbose llama.cpp/ggml log output unless explicitly enabled.
        // This must be called before backend_init to capture all C-level log output.
        crate::log::setup_ggml_logging(config.verbose_log);

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
                    inferred.unwrap_or_else(|| {
                        log::info!(
                            "Chat template auto-detection failed; falling back to ChatML"
                        );
                        "chatml".to_string()
                    })
                }
            }
        };

        log::info!("Using chat template: {chat_template}",);

        let ctx_params = ContextParams {
            n_ctx: config.n_ctx,
            n_batch: config.n_batch,
            n_threads: config.n_threads,
            n_threads_batch: config.n_threads_batch,
        };

        Ok(Self { model, context, ctx_params, chat_template })
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
        cancel_flag: Option<Arc<AtomicBool>>,
    ) -> LlmResult<mpsc::Receiver<StreamEvent>> {
        let formatted = self.format_prompt(prompt, &ChatTemplate::Auto, &[])?;
        let tokens = self.context.tokenize(&formatted, true)?;

        if tokens.is_empty() {
            return Err(LlmError::TokenizeFailed("empty token list".into()));
        }

        let (tx, rx) = mpsc::channel();
        let params = params.clone();
        let model = self.model.clone();
        let ctx_params = self.ctx_params.clone();

        // Spawn a dedicated thread for streaming inference.
        // We create a new context for the streaming thread since LlamaContext
        // is not Sync (it contains UnsafeCell for the sampler).
        std::thread::Builder::new()
            .name("llm-stream".into())
            .spawn(move || {
                // Catch panics to avoid undefined behavior
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // Create a new context for this streaming session
                    let ctx = match LlamaContext::new(
                        model,
                        ctx_params.n_ctx,
                        ctx_params.n_batch,
                        ctx_params.n_threads,
                        ctx_params.n_threads_batch,
                    ) {
                        Ok(ctx) => ctx,
                        Err(e) => {
                            let _ = tx.send(StreamEvent::Error(format!(
                                "Failed to create inference context: {e}"
                            )));
                            return;
                        }
                    };

                    // Run streaming inference
                    match ctx.infer_stream(&tokens, &params, tx.clone(), cancel_flag) {
                        Ok(_usage) => {
                            // Completed successfully, Completed event already sent
                        }
                        Err(e) => {
                            let _ = tx.send(StreamEvent::Error(format!("Inference error: {e}")));
                        }
                    }
                }));

                // Handle thread panic
                if let Err(panic) = result {
                    let msg = if let Some(s) = panic.downcast_ref::<String>() {
                        s.clone()
                    } else if let Some(s) = panic.downcast_ref::<&str>() {
                        s.to_string()
                    } else {
                        "Inference thread panicked".to_string()
                    };
                    let _ = tx.send(StreamEvent::Error(msg));
                }
            })
            .map_err(|e| LlmError::Internal(format!("Failed to spawn inference thread: {e}")))?;

        Ok(rx)
    }

    fn model_info(&self) -> LlmResult<ModelInfo> {
        Ok(self.model.info().clone())
    }

    fn infer_cached(
        &self,
        prompt: &str,
        params: &GenerationParams,
        cache_state: KvCacheState,
        cancel_flag: Option<&AtomicBool>,
    ) -> LlmResult<(InferenceResponse, KvCacheState)> {
        let formatted = self.format_prompt(prompt, &ChatTemplate::Auto, &[])?;
        let tokens = self.context.tokenize(&formatted, true)?;

        if tokens.is_empty() {
            return Err(LlmError::TokenizeFailed("empty token list".into()));
        }

        let (resp, new_cache) =
            self.context
                .infer_incremental(&tokens, cache_state, params, cancel_flag)?;

        Ok((resp, new_cache))
    }

    fn infer_stream_cached(
        &self,
        prompt: &str,
        params: &GenerationParams,
        _cache_state: KvCacheState,
        cancel_flag: Option<Arc<AtomicBool>>,
    ) -> LlmResult<(mpsc::Receiver<StreamEvent>, KvCacheState)> {
        let formatted = self.format_prompt(prompt, &ChatTemplate::Auto, &[])?;
        let tokens = self.context.tokenize(&formatted, true)?;

        if tokens.is_empty() {
            return Err(LlmError::TokenizeFailed("empty token list".into()));
        }

        let (tx, rx) = mpsc::channel();
        let params = params.clone();
        let model = self.model.clone();
        let ctx_params = self.ctx_params.clone();

        // For streaming multi-turn, we create a new context with the KV cache
        // approach: we pass the cache_state so the new context knows how many
        // tokens are already cached from previous turns.
        std::thread::Builder::new()
            .name("llm-stream-cached".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let ctx = match LlamaContext::new(
                        model,
                        ctx_params.n_ctx,
                        ctx_params.n_batch,
                        ctx_params.n_threads,
                        ctx_params.n_threads_batch,
                    ) {
                        Ok(ctx) => ctx,
                        Err(e) => {
                            let _ = tx.send(StreamEvent::Error(format!(
                                "Failed to create inference context: {e}"
                            )));
                            return;
                        }
                    };

                    // For streaming with a new context each time, cache is always 0
                    match ctx.infer_stream_incremental(&tokens, 0, &params, tx.clone(), cancel_flag) {
                        Ok(_) => {}
                        Err(e) => {
                            let _ = tx.send(StreamEvent::Error(format!("Inference error: {e}")));
                        }
                    }
                }));

                if let Err(panic) = result {
                    let msg = if let Some(s) = panic.downcast_ref::<String>() {
                        s.clone()
                    } else if let Some(s) = panic.downcast_ref::<&str>() {
                        s.to_string()
                    } else {
                        "Inference thread panicked".to_string()
                    };
                    let _ = tx.send(StreamEvent::Error(msg));
                }
            })
            .map_err(|e| LlmError::Internal(format!("Failed to spawn inference thread: {e}")))?;

        // For streaming, KV cache state is always 0 since we spawn a new context
        Ok((rx, 0))
    }
}

// ---------------------------------------------------------------------------
// Auto-discovery
// ---------------------------------------------------------------------------

/// Default model directory: `~/.chatvcode/models/`
#[must_use]
pub fn default_model_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".chatvcode")
        .join("models")
}

/// Auto-discover a GGUF model from the default directory.
///
/// Scans `~/.chatvcode/models/` for valid GGUF files using
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

            To get started with ChatVCode, you need a GGUF model file.

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
              chatvcode chat '<your question>'

            Example:
              mkdir -p {dir}
              curl -Lo {dir}/model.gguf '<URL>'
              chatvcode chat 'Explain the main function'
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
/// let s = chatvcode_llm::service::dedent("
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
/// Useful for unit tests and integration tests where model loading
/// is not needed.
///
/// # Example
///
/// ```ignore
/// use chatvcode_llm::{MockLlmService, LlmService, GenerationParams};
///
/// let mock = MockLlmService::new("Hello, world!")
///     .with_tokens_per_second(100.0);
///
/// let response = mock.infer("test", &GenerationParams::default(), None)?;
/// assert_eq!(response.text, "Hello, world!");
/// ```
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
        params: &GenerationParams,
        cancel_flag: Option<Arc<AtomicBool>>,
    ) -> LlmResult<mpsc::Receiver<StreamEvent>> {
        let (tx, rx) = mpsc::channel();
        let text = self.response_text.clone();
        let tokens_per_second = self.tokens_per_second;
        let max_tokens = params.max_tokens;

        std::thread::Builder::new()
            .name("mock-stream".into())
            .spawn(move || {
                let _ = tx.send(StreamEvent::Started);

                // Split text into words and simulate token-by-token generation
                let words: Vec<&str> = text.split_whitespace().collect();
                let token_delay = std::time::Duration::from_secs_f64(1.0 / tokens_per_second);

                for (generated_count, word) in words.into_iter().enumerate() {
                    // Check cancellation
                    if let Some(ref flag) = cancel_flag
                        && flag.load(std::sync::atomic::Ordering::Relaxed)
                    {
                        let _ = tx.send(StreamEvent::Cancelled);
                        return;
                    }

                    // Check max_tokens
                    if generated_count as i32 >= max_tokens {
                        break;
                    }

                    let token_text =
                        if generated_count == 0 { word.to_string() } else { format!(" {word}") };

                    if tx.send(StreamEvent::Token(token_text)).is_err() {
                        // Receiver dropped
                        return;
                    }

                    std::thread::sleep(token_delay);
                }

                let _ = tx.send(StreamEvent::Completed);
            })
            .map_err(|e| LlmError::Internal(format!("Failed to spawn stream thread: {e}")))?;

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
        assert!(dir.to_string_lossy().contains(".chatvcode"));
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

    #[test]
    fn test_mock_service_infer_stream_basic() {
        let service = MockLlmService::new("Hello world");
        let params = GenerationParams::default();

        let rx = service.infer_stream("test", &params, None).unwrap();

        // Collect events
        let mut events = Vec::new();
        while let Ok(event) = rx.recv_timeout(std::time::Duration::from_secs(5)) {
            events.push(event);
        }

        assert!(!events.is_empty());
        assert_eq!(events.first(), Some(&StreamEvent::Started));
        assert_eq!(events.last(), Some(&StreamEvent::Completed));

        // Check tokens
        let tokens: Vec<&str> = events.iter().filter_map(|e| e.as_token()).collect();
        assert_eq!(tokens.join(""), "Hello world");
    }

    #[test]
    fn test_mock_service_infer_stream_cancelled() {
        let service = MockLlmService::new("Should be cancelled");
        let params = GenerationParams::default();
        let cancel = Arc::new(AtomicBool::new(true));

        let rx = service.infer_stream("test", &params, Some(cancel)).unwrap();

        let mut events = Vec::new();
        while let Ok(event) = rx.recv_timeout(std::time::Duration::from_secs(5)) {
            events.push(event);
        }

        assert_eq!(events.first(), Some(&StreamEvent::Started));
        assert!(events.contains(&StreamEvent::Cancelled));
    }

    #[test]
    fn test_mock_service_infer_stream_max_tokens() {
        let service = MockLlmService::new("word1 word2 word3 word4 word5");
        let params = GenerationParams { max_tokens: 2, ..GenerationParams::default() };

        let rx = service.infer_stream("test", &params, None).unwrap();

        let mut token_count = 0;
        while let Ok(event) = rx.recv_timeout(std::time::Duration::from_secs(5)) {
            if event.is_token() {
                token_count += 1;
            }
        }

        assert!(token_count <= 2, "Should not exceed max_tokens");
    }

    #[test]
    fn test_mock_service_infer_batch() {
        let service = MockLlmService::new("Batch response");
        let params = GenerationParams::default();
        let prompts = vec!["prompt1", "prompt2", "prompt3"];

        let results = service.infer_batch(&prompts, &params, None).unwrap();
        assert_eq!(results.len(), 3);
        for resp in &results {
            assert_eq!(resp.text, "Batch response");
        }
    }

    #[test]
    fn test_mock_service_infer_batch_cancelled() {
        let service = MockLlmService::new("Should not complete");
        let params = GenerationParams::default();
        let prompts = vec!["p1", "p2", "p3"];
        let cancel = AtomicBool::new(true);

        let results = service.infer_batch(&prompts, &params, Some(&cancel)).unwrap();
        for resp in &results {
            assert_eq!(resp.stop_reason, StopReason::Cancelled);
        }
    }

    #[test]
    fn test_mock_service_infer_batch_empty() {
        let service = MockLlmService::new("test");
        let params = GenerationParams::default();
        let prompts: Vec<&str> = vec![];

        let results = service.infer_batch(&prompts, &params, None).unwrap();
        assert!(results.is_empty());
    }

    // --- LlamaEmbeddingService unit tests ---

    #[test]
    fn test_mock_service_infer_cached_default() {
        let service = MockLlmService::new("Cached inference test");
        let params = GenerationParams::default();

        // The default infer_cached should delegate to infer and return cache_state=0
        let (resp, cache_state) = service.infer_cached("test", &params, 0, None).unwrap();
        assert_eq!(resp.text, "Cached inference test");
        assert_eq!(cache_state, 0); // Mock doesn't support KV cache
    }

    #[test]
    fn test_mock_service_infer_stream_cached_default() {
        let service = MockLlmService::new("Stream cached test");
        let params = GenerationParams::default();

        let (rx, cache_state) =
            service.infer_stream_cached("test", &params, 0, None).unwrap();
        assert_eq!(cache_state, 0); // Mock doesn't support KV cache

        let mut events = Vec::new();
        while let Ok(event) = rx.recv_timeout(std::time::Duration::from_secs(5)) {
            events.push(event);
        }
        assert!(!events.is_empty());
        assert_eq!(events.first(), Some(&StreamEvent::Started));
        assert_eq!(events.last(), Some(&StreamEvent::Completed));
    }

    #[test]
    fn test_chat_session_with_mock_multi_turn() {
        use crate::types::ChatSession;

        let mock = MockLlmService::new("Multi-turn response");
        let params = GenerationParams::default();

        let mut session = ChatSession::new(ChatTemplate::ChatML)
            .system_prompt("You are helpful.");

        // First turn
        assert_eq!(session.kv_cache_state(), 0);
        let resp1 = session.chat("What is Rust?", &mock, &params).unwrap();
        assert!(!resp1.text.is_empty());
        assert_eq!(session.len(), 2); // user + assistant
        assert_eq!(session.turn_count(), 1);
        // Mock returns cache_state=0
        assert_eq!(session.kv_cache_state(), 0);

        // Second turn (history preserved)
        let resp2 = session.chat("How do lifetimes work?", &mock, &params).unwrap();
        assert!(!resp2.text.is_empty());
        assert_eq!(session.len(), 4);
        assert_eq!(session.turn_count(), 2);

        // The prompt for the second turn includes history
        let prompt = session.build_prompt_with("Third question").unwrap();
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("How do lifetimes work?"));
        assert!(prompt.contains("Third question"));
    }

    // --- LlamaEmbeddingService unit tests ---

    #[test]
    fn test_llama_embedding_service_from_path_nonexistent() {
        // Loading from a nonexistent path should fail gracefully
        let result =
            LlamaEmbeddingService::from_path(Path::new("/nonexistent/model.gguf"), 512, 4, 0, false);
        assert!(result.is_err(), "Expected error for nonexistent model path");
    }
}

// ---------------------------------------------------------------------------
// Llama-based Embedding Service
// ---------------------------------------------------------------------------

/// A [`chatvcode_vdb::EmbeddingService`] implementation backed by a GGUF model via `llama.cpp`.
///
/// Uses the same GGUF model file as the LLM for inference, but creates
/// a separate context with `embeddings = true` to extract embedding vectors.
/// This eliminates the need for a separate ONNX embedding model.
///
/// # Example
///
/// ```ignore
/// let model = LlamaModel::load("qwen2.5-coder-7b.gguf", 0, true, false)?;
/// let service = LlamaEmbeddingService::new(Arc::new(model), 512, 4)?;
/// let vectors = service.embed(&["Hello, world!"])?;
/// println!("Embedding dim: {}", service.dimension()); // e.g. 3584
/// ```
pub struct LlamaEmbeddingService {
    embed_ctx: std::sync::Mutex<crate::context::LlamaEmbeddingContext>,
    dimension: usize,
}

impl LlamaEmbeddingService {
    /// Create a new embedding service from a loaded model.
    ///
    /// # Arguments
    ///
    /// * `model` — Shared reference to a loaded `LlamaModel`
    /// * `n_ctx` — Context window for embedding (512 is usually sufficient)
    /// * `n_threads` — Number of threads for embedding computation
    pub fn new(model: Arc<LlamaModel>, n_ctx: u32, n_threads: i32) -> LlmResult<Self> {
        let embed_ctx = crate::context::LlamaEmbeddingContext::new(model, n_ctx, n_threads)?;
        let dim = embed_ctx.dimension();

        log::info!(
            "LlamaEmbeddingService created: dimension={dim}, n_ctx={n_ctx}, threads={n_threads}"
        );

        Ok(Self { embed_ctx: std::sync::Mutex::new(embed_ctx), dimension: dim })
    }

    /// Create a new embedding service by loading a model from disk.
    ///
    /// Convenience method that loads the model and creates the embedding context.
    pub fn from_path(
        model_path: &Path,
        n_ctx: u32,
        n_threads: i32,
        n_gpu_layers: i32,
        verbose: bool,
    ) -> LlmResult<Self> {
        // Ensure logging is set up before model load (idempotent).
        crate::log::setup_ggml_logging(verbose);
        let model = LlamaModel::load(model_path, n_gpu_layers, true, false)?;
        Self::new(Arc::new(model), n_ctx, n_threads)
    }

    /// Get the embedding dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Embed a batch of texts, returning L2-normalized vectors.
    ///
    /// Each text is embedded independently using the GGUF model's
    /// hidden state output. The vectors are L2-normalized for
    /// cosine similarity search.
    pub fn embed(&self, texts: &[&str]) -> LlmResult<Vec<Vec<f32>>> {
        let mut ctx = self
            .embed_ctx
            .lock()
            .map_err(|e| LlmError::Internal(format!("Embedding context lock poisoned: {e}")))?;
        ctx.embed_batch(texts)
    }
}
