//! Safe Rust wrapper around llama.cpp FFI bindings.
//!
//! Provides [`LlamaModel`] for model loading/inspection and
//! [`LlamaContext`] for inference. These are the building blocks
//! for the higher-level [`crate::LlmService`] trait.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::error::{LlmError, LlmResult};
use crate::ffi;
use crate::types::{GenerationParams, InferenceResponse, ModelInfo, StopReason, TokenUsage};

// ---------------------------------------------------------------------------
// LlamaModel
// ---------------------------------------------------------------------------

/// A loaded GGUF model.
///
/// Wraps a `*mut llama_model` and provides safe access to model metadata.
pub struct LlamaModel {
    ptr: *mut ffi::llama_model,
    info: ModelInfo,
}

// SAFETY: llama_model is thread-safe for read-only access according to llama.cpp docs.
unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}

impl LlamaModel {
    /// Load a GGUF model from disk.
    pub fn load(
        path: &Path,
        n_gpu_layers: i32,
        use_mmap: bool,
        use_mlock: bool,
    ) -> LlmResult<Self> {
        if !path.exists() {
            return Err(LlmError::ModelNotFound(path.display().to_string()));
        }

        let path_c = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| LlmError::InvalidParameter("model path contains null bytes".into()))?;

        let mut params = unsafe { ffi::llama_model_default_params() };
        params.n_gpu_layers = n_gpu_layers;
        params.use_mmap = use_mmap;
        params.use_mlock = use_mlock;

        let ptr = unsafe { ffi::llama_model_load_from_file(path_c.as_ptr(), params) };
        if ptr.is_null() {
            return Err(LlmError::ModelLoadFailed(format!(
                "failed to load model from {}",
                path.display()
            )));
        }

        let info = Self::read_info(ptr);

        Ok(Self { ptr, info })
    }

    /// Load a GGUF model using a full [`crate::types::LlmConfig`].
    pub fn load_with_config(config: &crate::types::LlmConfig) -> LlmResult<Self> {
        Self::load(&config.model_path, config.n_gpu_layers, config.use_mmap, config.use_mlock)
    }

    /// Returns model metadata.
    #[must_use]
    pub const fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// Returns the embedding dimension of the model.
    #[must_use]
    pub fn n_embd(&self) -> i32 {
        unsafe { ffi::llama_model_n_embd(self.ptr) }
    }

    /// Returns the output embedding dimension of the model.
    ///
    /// For dedicated embedding models, this may differ from `n_embd()` when the
    /// model projects hidden states to a different output dimension.
    #[must_use]
    pub fn n_embd_out(&self) -> i32 {
        unsafe { ffi::llama_model_n_embd_out(self.ptr) }
    }

    /// Returns the raw C pointer (for use by `LlamaContext` and `LlamaEmbeddingContext`).
    pub(crate) const fn as_ptr(&self) -> *mut ffi::llama_model {
        self.ptr
    }

    /// Get the model's chat template, or the default if not specified.
    #[must_use]
    pub fn chat_template(&self, name: Option<&str>) -> Option<String> {
        let name_c = name.and_then(|n| CString::new(n).ok());
        let name_ptr = name_c.as_ref().map_or(ptr::null(), |c| c.as_ptr());

        unsafe {
            let tmpl_ptr = ffi::llama_model_chat_template(self.ptr, name_ptr);
            crate::error::cstr_to_string(tmpl_ptr)
        }
    }

    /// Read all metadata key-value pairs from the model.
    #[must_use]
    pub fn metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        let count = unsafe { ffi::llama_model_meta_count(self.ptr) };

        for i in 0..count {
            let mut key_buf = vec![0u8; 256];
            let mut val_buf = vec![0u8; 4096];

            let key_len = unsafe {
                ffi::llama_model_meta_key_by_index(
                    self.ptr,
                    i,
                    key_buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                    key_buf.len(),
                )
            };
            let val_len = unsafe {
                ffi::llama_model_meta_val_str_by_index(
                    self.ptr,
                    i,
                    val_buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                    val_buf.len(),
                )
            };

            // Handle negative val_len (indicates required buffer size)
            let actual_val_len = if val_len < 0 {
                let required = (-val_len) as usize;
                val_buf.resize(required + 1, 0);
                let retry_len = unsafe {
                    ffi::llama_model_meta_val_str_by_index(
                        self.ptr,
                        i,
                        val_buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                        val_buf.len(),
                    )
                };
                retry_len.max(0) as usize
            } else {
                val_len.max(0) as usize
            };

            if key_len > 0 && actual_val_len > 0 {
                let key = String::from_utf8_lossy(&key_buf[..key_len as usize]).into_owned();
                let val = String::from_utf8_lossy(&val_buf[..actual_val_len]).into_owned();
                meta.insert(key, val);
            }
        }

        meta
    }

    /// Read model metadata into a `ModelInfo` struct.
    fn read_info(ptr: *mut ffi::llama_model) -> ModelInfo {
        let mut desc_buf = vec![0u8; 1024]; // Larger buffer for model description
        let desc_len = unsafe {
            ffi::llama_model_desc(
                ptr,
                desc_buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                desc_buf.len(),
            )
        };

        // Handle negative desc_len (indicates required buffer size)
        if desc_len < 0 {
            let required = (-desc_len) as usize;
            desc_buf.resize(required + 1, 0);
            unsafe {
                ffi::llama_model_desc(
                    ptr,
                    desc_buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                    desc_buf.len(),
                )
            };
        }
        let actual_desc_len = desc_len.max(0) as usize;

        let description = String::from_utf8_lossy(&desc_buf[..actual_desc_len])
            .trim_end_matches('\0')
            .to_string();

        let vocab = unsafe { ffi::llama_model_get_vocab(ptr) };
        let n_vocab = if vocab.is_null() { 0 } else { unsafe { ffi::llama_vocab_n_tokens(vocab) } };
        let vocab_type = if vocab.is_null() {
            "none".to_string()
        } else {
            let vt = unsafe { ffi::llama_vocab_type(vocab) };
            match ffi::LlamaVocabType::from_i32(vt) {
                Some(ffi::LlamaVocabType::Spm) => "spm",
                Some(ffi::LlamaVocabType::Bpe) => "bpe",
                Some(ffi::LlamaVocabType::None) => "none",
                Some(ffi::LlamaVocabType::Wpm) => "wpm",
                Some(ffi::LlamaVocabType::Ugm) => "ugm",
                Some(ffi::LlamaVocabType::Rwkv) => "rwkv",
                Some(ffi::LlamaVocabType::Plamo2) => "plamo2",
                None => "other",
            }
            .to_string()
        };

        let n_ctx_train = unsafe { ffi::llama_model_n_ctx_train(ptr) };
        let n_embd = unsafe { ffi::llama_model_n_embd(ptr) };
        let n_layer = unsafe { ffi::llama_model_n_layer(ptr) };
        let n_head = unsafe { ffi::llama_model_n_head(ptr) };
        let n_head_kv = unsafe { ffi::llama_model_n_head_kv(ptr) };
        let size_bytes = unsafe { ffi::llama_model_size(ptr) };
        let n_params = unsafe { ffi::llama_model_n_params(ptr) };
        let has_encoder = unsafe { ffi::llama_model_has_encoder(ptr) };
        let has_decoder = unsafe { ffi::llama_model_has_decoder(ptr) };

        let chat_template_available =
            unsafe { !ffi::llama_model_chat_template(ptr, ptr::null()).is_null() };

        // Try to extract architecture from metadata
        let meta = {
            let mut m = HashMap::new();
            let count = unsafe { ffi::llama_model_meta_count(ptr) };
            for i in 0..count {
                let mut key_buf = vec![0u8; 256];
                let mut val_buf = vec![0u8; 4096]; // Larger buffer for chat templates
                let key_len = unsafe {
                    ffi::llama_model_meta_key_by_index(
                        ptr,
                        i,
                        key_buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                        key_buf.len(),
                    )
                };
                let val_len = unsafe {
                    ffi::llama_model_meta_val_str_by_index(
                        ptr,
                        i,
                        val_buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                        val_buf.len(),
                    )
                };

                // Handle negative val_len (indicates required buffer size)
                let actual_val_len = if val_len < 0 {
                    let required = (-val_len) as usize;
                    val_buf.resize(required + 1, 0);
                    let retry_len = unsafe {
                        ffi::llama_model_meta_val_str_by_index(
                            ptr,
                            i,
                            val_buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                            val_buf.len(),
                        )
                    };
                    retry_len.max(0) as usize
                } else {
                    val_len.max(0) as usize
                };

                if key_len > 0 && actual_val_len > 0 {
                    let key = String::from_utf8_lossy(&key_buf[..key_len as usize]).into_owned();
                    let val = String::from_utf8_lossy(&val_buf[..actual_val_len]).into_owned();
                    m.insert(key, val);
                }
            }
            m
        };

        let architecture = meta
            .get("general.architecture")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        let ftype = meta.get("general.file_type").cloned().unwrap_or_else(|| {
            meta.get("general.quantization_version")
                .map_or_else(|| "unknown".to_string(), |v| format!("q{v}"))
        });

        let rope_type_str = {
            let rt = unsafe { ffi::llama_model_rope_type(ptr) };
            format!("{rt}")
        };

        ModelInfo {
            description,
            architecture,
            n_params,
            size_bytes,
            n_ctx_train,
            n_embd,
            n_layer,
            n_head,
            n_head_kv,
            n_vocab,
            vocab_type,
            ftype,
            chat_template_available,
            rope_type: rope_type_str,
            has_encoder,
            has_decoder,
        }
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::llama_model_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// LlamaContext
// ---------------------------------------------------------------------------

/// Inference context for a loaded model.
///
/// Holds the KV cache, tokenizer state, and sampler chain.
/// Uses `UnsafeCell` for the sampler to allow rebuilding during inference.
pub struct LlamaContext {
    ctx: *mut ffi::llama_context,
    model: Arc<LlamaModel>,
    sampler: UnsafeCell<*mut ffi::llama_sampler>,
    n_ctx: u32,
}

// SAFETY: llama_context is not thread-safe, but we guard access at a higher level.
unsafe impl Send for LlamaContext {}
unsafe impl Sync for LlamaContext {}

impl LlamaContext {
    /// Create a new inference context from a loaded model.
    pub fn new(
        model: Arc<LlamaModel>,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
        n_threads_batch: i32,
    ) -> LlmResult<Self> {
        let mut params = unsafe { ffi::llama_context_default_params() };
        params.n_ctx = n_ctx;
        params.n_batch = n_batch;
        // n_ubatch controls the physical micro-batch size.  Keep it at a
        // moderate size (512) to limit peak memory; only raise it to n_batch
        // if n_batch itself is very small.
        params.n_ubatch = n_batch.min(512);
        // We set the rest to default; they will be auto-adjusted after init
        params.embeddings = false;
        params.no_perf = false;

        let ctx = unsafe { ffi::llama_init_from_model(model.as_ptr(), params) };
        if ctx.is_null() {
            return Err(LlmError::ModelLoadFailed("failed to create inference context".into()));
        }

        // Read actual values after init (may differ from requested)
        let actual_n_ctx = unsafe { ffi::llama_n_ctx(ctx) };
        let actual_n_batch = unsafe { ffi::llama_n_batch(ctx) };

        // Set thread counts
        unsafe { ffi::llama_set_n_threads(ctx, n_threads, n_threads_batch) };

        // Create sampler chain
        let sampler = Self::create_default_sampler();

        log::info!(
            "LlamaContext created: n_ctx={actual_n_ctx}, n_batch={actual_n_batch}, threads={n_threads}/{n_threads_batch}"
        );

        Ok(Self { ctx, model, sampler: UnsafeCell::new(sampler), n_ctx: actual_n_ctx })
    }

    /// Returns the context size.
    pub const fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    /// Returns the model reference.
    pub const fn model(&self) -> &Arc<LlamaModel> {
        &self.model
    }

    /// Get the vocab pointer.
    fn vocab(&self) -> *const ffi::llama_vocab {
        unsafe { ffi::llama_model_get_vocab(self.model.as_ptr()) }
    }

    /// Tokenize a text string.
    pub fn tokenize(&self, text: &str, add_special: bool) -> LlmResult<Vec<ffi::llama_token>> {
        let text_c = CString::new(text)
            .map_err(|_| LlmError::TokenizeFailed("text contains null bytes".into()))?;

        // First call to get required buffer size
        // llama_tokenize returns:
        //   - positive number: required token count
        //   - negative number: need abs(n) space (old behavior)
        let n = unsafe {
            ffi::llama_tokenize(
                self.vocab(),
                text_c.as_ptr(),
                text.len() as i32,
                ptr::null_mut(),
                0,
                add_special,
                false,
            )
        };

        if n == 0 {
            return Ok(Vec::new());
        }

        // Determine required size
        let size = if n < 0 { (-n) as usize } else { n as usize };
        let mut tokens = vec![0i32; size];

        // Second call to actually tokenize
        let actual = unsafe {
            ffi::llama_tokenize(
                self.vocab(),
                text_c.as_ptr(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                size as i32,
                add_special,
                false,
            )
        };

        if actual < 0 {
            return Err(LlmError::TokenizeFailed(format!("tokenization returned {actual}")));
        }

        tokens.truncate(actual.max(0) as usize);
        Ok(tokens)
    }

    /// Detokenize a token into its text representation.
    pub fn token_to_piece(&self, token: ffi::llama_token) -> String {
        let mut buf = vec![0u8; 256]; // Larger buffer for multi-byte chars
        let len = unsafe {
            ffi::llama_token_to_piece(
                self.vocab(),
                token,
                buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                buf.len() as i32,
                0,
                true,
            )
        };

        let actual_len = if len < 0 {
            // Need larger buffer
            let required_size = (-len) as usize;
            buf.resize(required_size + 1, 0);
            let retry_len = unsafe {
                ffi::llama_token_to_piece(
                    self.vocab(),
                    token,
                    buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                    buf.len() as i32,
                    0,
                    true,
                )
            };
            retry_len.max(0) as usize
        } else {
            len.max(0) as usize
        };

        String::from_utf8_lossy(
            &buf[..actual_len]
                .iter()
                .take_while(|&&b| b != 0)
                .copied()
                .collect::<Vec<u8>>(),
        )
        .into_owned()
    }

    /// Check if a token is end-of-generation.
    pub fn is_eog(&self, token: ffi::llama_token) -> bool {
        unsafe { ffi::llama_vocab_is_eog(self.vocab(), token) }
    }

    /// Get the EOS token id.
    pub fn eos_token(&self) -> ffi::llama_token {
        unsafe { ffi::llama_vocab_eos(self.vocab()) }
    }

    /// Decode a batch and return the id of the last (or -1 indexed) logit.
    fn decode_batch(&self, tokens: &[ffi::llama_token]) -> LlmResult<()> {
        if tokens.len() > self.n_ctx as usize {
            return Err(LlmError::ContextOverflow {
                n_ctx: self.n_ctx,
                n_tokens: tokens.len() as i32,
            });
        }

        let n_tokens = tokens.len() as i32;
        if n_tokens == 0 {
            return Ok(());
        }

        // Use llama_batch_get_one for simplicity with single sequence
        let mut tokens_mut = tokens.to_vec();
        let batch = unsafe { ffi::llama_batch_get_one(tokens_mut.as_mut_ptr(), n_tokens) };

        let ret = unsafe { ffi::llama_decode(self.ctx, batch) };
        match ret {
            0 => Ok(()),
            1 => Err(LlmError::ContextOverflow { n_ctx: self.n_ctx, n_tokens }),
            -1 => Err(LlmError::InferenceFailed("invalid input batch".into())),
            code => Err(LlmError::InferenceFailed(format!("decode returned {code}"))),
        }
    }

    /// Returns the current sampler pointer.
    fn get_sampler(&self) -> *mut ffi::llama_sampler {
        unsafe { *self.sampler.get() }
    }

    /// Sample the next token from the logits at position `idx`.
    ///
    /// Also calls `llama_sampler_accept()` to update sampler state
    /// (required for `repeat_penalty`, etc.).
    fn sample_and_accept(&self, idx: i32) -> ffi::llama_token {
        let token = unsafe { ffi::llama_sampler_sample(self.get_sampler(), self.ctx, idx) };
        unsafe { ffi::llama_sampler_accept(self.get_sampler(), token) };
        token
    }

    /// Run inference synchronously.
    ///
    /// Returns the generated text and stats.
    pub fn infer(
        &self,
        prompt_tokens: &[ffi::llama_token],
        params: &GenerationParams,
        cancel_flag: Option<&AtomicBool>,
    ) -> LlmResult<InferenceResponse> {
        let start_time = Instant::now();
        let prompt_n = prompt_tokens.len() as i32;

        // ---- Configure sampler based on params ----
        self.rebuild_sampler(params);

        // ---- Evaluate the prompt ----
        self.decode_batch(prompt_tokens)?;

        // ---- Generate tokens ----
        let mut generated_tokens: Vec<ffi::llama_token> = Vec::new();
        let eos = self.eos_token();

        for _i in 0..params.max_tokens {
            // Check cancellation
            if let Some(flag) = cancel_flag
                && flag.load(Ordering::Relaxed)
            {
                return Ok(InferenceResponse {
                    text: self.detokenize_all(&generated_tokens),
                    stop_reason: StopReason::Cancelled,
                    token_usage: TokenUsage::new(prompt_n, generated_tokens.len() as i32),
                    duration: start_time.elapsed(),
                    time_to_first_token: None,
                    tokens_per_second: 0.0,
                });
            }

            // Sample the next token and update sampler state
            let next_token = self.sample_and_accept(-1);

            // Check for EOS
            if next_token == eos || self.is_eog(next_token) {
                break;
            }

            generated_tokens.push(next_token);

            // Check stop strings
            if !params.stop_strings.is_empty() {
                let text = self.detokenize_all(&generated_tokens);
                for stop_str in &params.stop_strings {
                    if text.contains(stop_str.as_str()) {
                        return Ok(InferenceResponse {
                            text,
                            stop_reason: StopReason::StopString(stop_str.clone()),
                            token_usage: TokenUsage::new(prompt_n, generated_tokens.len() as i32),
                            duration: start_time.elapsed(),
                            time_to_first_token: None,
                            tokens_per_second: 0.0,
                        });
                    }
                }
            }

            // Decode the single new token
            self.decode_batch(&[next_token])?;
        }

        let text = self.detokenize_all(&generated_tokens);
        let duration = start_time.elapsed();
        let completion_tokens = generated_tokens.len() as i32;
        let tps = if duration.as_secs_f64() > 0.0 {
            f64::from(completion_tokens) / duration.as_secs_f64()
        } else {
            0.0
        };

        let stop_reason = if completion_tokens >= params.max_tokens {
            StopReason::MaxTokens
        } else {
            StopReason::Eos
        };

        Ok(InferenceResponse {
            text,
            stop_reason,
            token_usage: TokenUsage::new(prompt_n, completion_tokens),
            duration,
            time_to_first_token: None,
            tokens_per_second: tps,
        })
    }

    /// Run streaming inference, sending events through the provided channel.
    ///
    /// This function runs the generation loop and sends [`StreamEvent`]s
    /// through the `sender`. The caller should spawn this on a dedicated
    /// thread and read from the receiver.
    pub fn infer_stream(
        &self,
        prompt_tokens: &[ffi::llama_token],
        params: &GenerationParams,
        sender: std::sync::mpsc::Sender<crate::types::StreamEvent>,
        cancel_flag: Option<Arc<AtomicBool>>,
    ) -> LlmResult<TokenUsage> {
        let prompt_n = prompt_tokens.len() as i32;

        // ---- Configure sampler ----
        self.rebuild_sampler(params);

        // ---- Evaluate the prompt ----
        self.decode_batch(prompt_tokens)?;

        let _ = sender.send(crate::types::StreamEvent::Started);

        let mut generated_tokens: Vec<ffi::llama_token> = Vec::new();
        let eos = self.eos_token();

        for _i in 0..params.max_tokens {
            // Check cancellation
            if let Some(ref flag) = cancel_flag
                && flag.load(Ordering::Relaxed)
            {
                let _ = sender.send(crate::types::StreamEvent::Cancelled);
                return Ok(TokenUsage::new(prompt_n, generated_tokens.len() as i32));
            }

            let next_token = self.sample_and_accept(-1);

            if next_token == eos || self.is_eog(next_token) {
                break;
            }

            generated_tokens.push(next_token);

            // Emit token
            let piece = self.token_to_piece(next_token);
            if !piece.is_empty()
                && sender
                    .send(crate::types::StreamEvent::Token(piece))
                    .is_err()
            {
                // Receiver dropped → cancelled
                return Ok(TokenUsage::new(prompt_n, generated_tokens.len() as i32));
            }

            // Check stop strings
            if !params.stop_strings.is_empty() {
                let text = self.detokenize_all(&generated_tokens);
                for stop_str in &params.stop_strings {
                    if text.contains(stop_str.as_str()) {
                        let _ = sender.send(crate::types::StreamEvent::Completed);
                        return Ok(TokenUsage::new(prompt_n, generated_tokens.len() as i32));
                    }
                }
            }

            self.decode_batch(&[next_token])?;
        }

        let _ = sender.send(crate::types::StreamEvent::Completed);
        Ok(TokenUsage::new(prompt_n, generated_tokens.len() as i32))
    }

    /// Detokenize a list of tokens into a string.
    fn detokenize_all(&self, tokens: &[ffi::llama_token]) -> String {
        if tokens.is_empty() {
            return String::new();
        }

        // Use a larger initial buffer to avoid frequent reallocations
        // Each token can expand to multiple bytes (especially for CJK characters)
        let mut buf = vec![0u8; tokens.len() * 64 + 256]; // generous estimate

        let len = unsafe {
            ffi::llama_detokenize(
                self.vocab(),
                tokens.as_ptr(),
                tokens.len() as i32,
                buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                buf.len() as i32,
                false,
                true,
            )
        };

        // If len is negative, it indicates the required buffer size
        let actual_len = if len < 0 {
            let required_size = (-len) as usize;
            buf.resize(required_size + 1, 0); // +1 for null terminator
            let retry_len = unsafe {
                ffi::llama_detokenize(
                    self.vocab(),
                    tokens.as_ptr(),
                    tokens.len() as i32,
                    buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                    buf.len() as i32,
                    false,
                    true,
                )
            };
            retry_len.max(0) as usize
        } else {
            len.max(0) as usize
        };

        // Convert to string, handling potential null bytes
        String::from_utf8_lossy(
            &buf[..actual_len]
                .iter()
                .take_while(|&&b| b != 0)
                .copied()
                .collect::<Vec<u8>>(),
        )
        .into_owned()
    }

    /// Rebuild the sampler chain based on generation params.
    fn rebuild_sampler(&self, params: &GenerationParams) {
        // Free the old sampler
        let old_sampler = self.get_sampler();
        if !old_sampler.is_null() {
            unsafe { ffi::llama_sampler_free(old_sampler) };
        }

        // Create chain
        let chain = unsafe {
            ffi::llama_sampler_chain_init(ffi::llama_sampler_chain_params { no_perf: false })
        };

        // Add samplers in order
        if params.repeat_penalty != 1.0 {
            let penalties = unsafe {
                ffi::llama_sampler_init_penalties(
                    params.repeat_last_n,
                    params.repeat_penalty,
                    0.0, // freq penalty
                    0.0, // presence penalty
                )
            };
            unsafe { ffi::llama_sampler_chain_add(chain, penalties) };
        }

        if params.top_k > 0 {
            let top_k = unsafe { ffi::llama_sampler_init_top_k(params.top_k) };
            unsafe { ffi::llama_sampler_chain_add(chain, top_k) };
        }

        if params.top_p < 1.0 {
            let top_p = unsafe { ffi::llama_sampler_init_top_p(params.top_p, 1) };
            unsafe { ffi::llama_sampler_chain_add(chain, top_p) };
        }

        if params.min_p > 0.0 {
            let min_p = unsafe { ffi::llama_sampler_init_min_p(params.min_p, 1) };
            unsafe { ffi::llama_sampler_chain_add(chain, min_p) };
        }

        if params.temperature <= 0.0 {
            // Greedy
            let greedy = unsafe { ffi::llama_sampler_init_greedy() };
            unsafe { ffi::llama_sampler_chain_add(chain, greedy) };
        } else {
            let temp = unsafe { ffi::llama_sampler_init_temp(params.temperature) };
            unsafe { ffi::llama_sampler_chain_add(chain, temp) };

            let dist = unsafe { ffi::llama_sampler_init_dist(params.seed) };
            unsafe { ffi::llama_sampler_chain_add(chain, dist) };
        }

        // Update the pointer via UnsafeCell
        unsafe {
            *self.sampler.get() = chain;
        }
    }

    /// Create a default sampler chain.
    fn create_default_sampler() -> *mut ffi::llama_sampler {
        unsafe {
            let chain =
                ffi::llama_sampler_chain_init(ffi::llama_sampler_chain_params { no_perf: false });
            // Default: top-k=40, top-p=0.9, temp=0.7, dist
            let top_k = ffi::llama_sampler_init_top_k(40);
            ffi::llama_sampler_chain_add(chain, top_k);
            let top_p = ffi::llama_sampler_init_top_p(0.9, 1);
            ffi::llama_sampler_chain_add(chain, top_p);
            let temp = ffi::llama_sampler_init_temp(0.7);
            ffi::llama_sampler_chain_add(chain, temp);
            let dist = ffi::llama_sampler_init_dist(u32::MAX);
            ffi::llama_sampler_chain_add(chain, dist);
            chain
        }
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        let sampler = unsafe { *self.sampler.get() };
        if !sampler.is_null() {
            unsafe { ffi::llama_sampler_free(sampler) };
        }
        if !self.ctx.is_null() {
            unsafe { ffi::llama_free(self.ctx) };
            self.ctx = ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// Embedding Context
// ---------------------------------------------------------------------------

/// A context for generating text embeddings from a loaded GGUF model.
///
/// Unlike [`LlamaContext`] which is used for text generation,
/// this context uses `embeddings = true` in the context parameters,
/// allowing the model to output embedding vectors via
/// `llama_get_embeddings()`.
///
/// # Usage
///
/// ```ignore
/// let model = LlamaModel::load("model.gguf", 0, true, false)?;
/// let embed_ctx = LlamaEmbeddingContext::new(Arc::new(model), 512, 4)?;
/// let embedding = embed_ctx.embed("Hello, world!")?;
/// println!("Embedding dim: {}", embedding.len());
/// ```
pub struct LlamaEmbeddingContext {
    ctx: *mut ffi::llama_context,
    model: Arc<LlamaModel>,
    n_embd: usize,
    n_ctx: u32,
}

// SAFETY: We guard access with &mut self on embed(), ensuring exclusive access.
unsafe impl Send for LlamaEmbeddingContext {}
unsafe impl Sync for LlamaEmbeddingContext {}

impl LlamaEmbeddingContext {
    /// Create a new embedding context from a loaded model.
    ///
    /// # Arguments
    ///
    /// * `model` — Shared reference to a loaded `LlamaModel`
    /// * `n_ctx` — Context size for the embedding (512 is usually sufficient)
    /// * `n_threads` — Number of threads for embedding computation
    pub fn new(model: Arc<LlamaModel>, n_ctx: u32, n_threads: i32) -> LlmResult<Self> {
        let mut params = unsafe { ffi::llama_context_default_params() };
        params.n_ctx = n_ctx;
        params.n_batch = n_ctx;
        params.n_ubatch = n_ctx.min(512);
        params.embeddings = true;
        params.pooling_type = ffi::LlamaPoolingType::Mean as i32;
        params.no_perf = false;

        let ctx = unsafe { ffi::llama_init_from_model(model.as_ptr(), params) };
        if ctx.is_null() {
            return Err(LlmError::ModelLoadFailed(
                "Failed to create embedding context. This may be due to insufficient memory."
                    .into(),
            ));
        }

        let n_embd_out = model.n_embd_out();
        let n_embd = if n_embd_out > 0 { n_embd_out } else { model.n_embd() };

        unsafe { ffi::llama_set_n_threads(ctx, n_threads, n_threads) };

        log::info!(
            "LlamaEmbeddingContext created: n_ctx={n_ctx}, n_embd={n_embd}, n_embd_out={n_embd_out}, threads={n_threads}"
        );

        Ok(Self { ctx, model, n_embd: n_embd as usize, n_ctx })
    }

    /// Get the embedding dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.n_embd
    }

    /// Get the context size.
    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    /// Embed a single text string and return its embedding vector.
    ///
    /// The returned vector is L2-normalized and has `dimension()` elements.
    pub fn embed(&mut self, text: &str) -> LlmResult<Vec<f32>> {
        // Tokenize using the model's vocabulary (same as LlamaContext)
        let text_c = CString::new(text)
            .map_err(|_| LlmError::TokenizeFailed("text contains null bytes".into()))?;

        let vocab = unsafe { ffi::llama_model_get_vocab(self.model.ptr) };
        if vocab.is_null() {
            return Err(LlmError::TokenizeFailed("model vocabulary is null".into()));
        }

        // Determine required token buffer size
        let n = unsafe {
            ffi::llama_tokenize(
                vocab,
                text_c.as_ptr(),
                text.len() as i32,
                ptr::null_mut(),
                0,
                true, // add_bos
                true, // add_eos
            )
        };

        if n == 0 {
            return Err(LlmError::TokenizeFailed("no tokens produced for embedding input".into()));
        }

        let max_tokens = if n < 0 { (-n) as usize } else { n as usize };
        let mut tokens = vec![0i32; max_tokens + 1];
        let actual = unsafe {
            ffi::llama_tokenize(
                vocab,
                text_c.as_ptr(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                true, // add_bos
                true, // add_eos
            )
        };

        if actual < 0 {
            return Err(LlmError::TokenizeFailed(format!(
                "tokenization failed for embedding input (ret={actual})"
            )));
        }
        tokens.truncate(actual as usize);

        // Truncate tokens to fit within the context window.
        // If the tokenized text exceeds n_ctx, keep only the first n_ctx tokens
        // so that llama_decode won't fail with GGML_ASSERT(n_tokens_all <= n_batch).
        let max_tokens = self.n_ctx as usize;
        if tokens.len() > max_tokens {
            log::warn!(
                "Embedding input has {} tokens, truncating to n_ctx={}",
                tokens.len(),
                max_tokens
            );
            tokens.truncate(max_tokens);
        }

        // Create batch and compute embeddings
        let batch = unsafe { ffi::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };

        let ret = unsafe { ffi::llama_decode(self.ctx, batch) };
        if ret != 0 {
            return Err(LlmError::Internal(format!(
                "Embedding decode failed with return code {ret}"
            )));
        }

        let embd_ptr = unsafe { ffi::llama_get_embeddings_seq(self.ctx, 0) };
        if embd_ptr.is_null() {
            return Err(LlmError::Internal(
                "Failed to get sequence embeddings from context (null pointer). \
                 This may indicate the model does not support mean pooling."
                    .into(),
            ));
        }

        // Copy embeddings to Vec
        let mut embedding = vec![0.0f32; self.n_embd];
        unsafe {
            std::ptr::copy_nonoverlapping(embd_ptr, embedding.as_mut_ptr(), self.n_embd);
        }

        // L2-normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        // Clear KV cache for next embedding
        unsafe {
            let mem = ffi::llama_get_memory(self.ctx);
            ffi::llama_memory_clear(mem, true);
        }

        Ok(embedding)
    }

    /// Embed multiple texts in sequence.
    ///
    /// Returns embeddings in the same order as the input texts.
    /// Each embedding vector is L2-normalized.
    pub fn embed_batch(&mut self, texts: &[&str]) -> LlmResult<Vec<Vec<f32>>> {
        let total = texts.len();
        let mut results = Vec::with_capacity(total);
        for (i, text) in texts.iter().enumerate() {
            let embedding = self.embed(text)?;
            if total > 1 {
                // Emit per-text progress so user can see the batch is still working
                log::debug!("Embedded chunk {}/{} ({} tokens)", i + 1, total, embedding.len());
                eprint!(" {}/{}", i + 1, total);
            }
            results.push(embedding);
        }
        Ok(results)
    }
}

impl Drop for LlamaEmbeddingContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { ffi::llama_free(self.ctx) };
            self.ctx = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod embedding_context_tests {
    use super::*;

    #[test]
    fn test_llama_model_n_embd_method_exists() {
        // Verify the n_embd() method exists on LlamaModel
        fn _assert_n_embd_exists(model: &LlamaModel) -> i32 {
            model.n_embd()
        }
        // This test passes by compiling.
    }

    #[test]
    fn test_llama_model_load_nonexistent_fails() {
        let result = LlamaModel::load(Path::new("/nonexistent/model.gguf"), 0, true, false);
        assert!(result.is_err());
    }
}
