//! LLM service abstraction and default implementation.
//!
//! The [`LlmService`] trait defines the high-level interface that
//! consumers (atlas-core, atlas-cli) use for inference. Backend
//! implementations live in [`crate::context`].

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;

use crate::context::{LlamaContext, LlamaModel};
use crate::error::{LlmError, LlmResult};
use crate::types::*;

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
}

impl LlamaService {
    /// Initialize the llama.cpp backend and load a model.
    ///
    /// `config` specifies the model path and loading parameters.
    pub fn new(config: &LlmConfig) -> LlmResult<Self> {
        // Initialize backend (call once; idempotent in llama.cpp)
        unsafe { crate::ffi::llama_backend_init() };

        let model = LlamaModel::load_with_config(config)?;
        let model = Arc::new(model);

        let context = LlamaContext::new(
            model.clone(),
            config.n_ctx,
            config.n_batch,
            config.n_threads,
            config.n_threads_batch,
        )?;

        let info = model.info();
        log::info!(
            "LlamaService initialized: model={}, arch={}, params={}, ctx={}",
            config.model_path.display(),
            info.architecture,
            info.n_params,
            context.n_ctx()
        );

        Ok(Self { model, context })
    }

    /// Initialize with explicit paths for model discovery.
    pub fn discover_and_load(
        model_path: Option<PathBuf>,
        n_ctx: u32,
        n_threads: i32,
        n_gpu_layers: i32,
    ) -> LlmResult<Self> {
        let path = match model_path {
            Some(p) => p,
            None => auto_discover_model()?,
        };

        let config = LlmConfig::new(path)
            .with_n_ctx(n_ctx)
            .with_n_threads(n_threads)
            .with_n_gpu_layers(n_gpu_layers);

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

        // Try to use llama_chat_apply_template
        let tmpl_str: Option<String> = match template {
            ChatTemplate::Auto => self.model.chat_template(None),
            ChatTemplate::Custom(custom) => Some(custom.clone()),
            _ => {
                if let Some(name) = template.template_name() {
                    self.model.chat_template(Some(name))
                } else {
                    None
                }
            }
        };

        match &tmpl_str {
            Some(tmpl) => {
                // Use llama.cpp's built-in chat template engine
                Self::apply_chat_template(tmpl, &chat_messages, true)
            }
            None => {
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
            return Err(LlmError::Internal(format!(
                "chat template application failed: {}",
                needed
            )));
        }

        let mut buf = vec![0u8; needed as usize + 1];
        let actual = unsafe {
            crate::ffi::llama_chat_apply_template(
                tmpl_c.as_ptr(),
                c_msgs.as_ptr(),
                c_msgs.len(),
                add_ass,
                buf.as_mut_ptr() as *mut std::ffi::c_char,
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
                    buf.as_mut_ptr() as *mut std::ffi::c_char,
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
pub fn default_model_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".codeatlas")
        .join("models")
}

/// Auto-discover a GGUF model from the default directory.
///
/// If exactly one `.gguf` file exists, it is returned.
/// If none exist, returns an error with a helpful message.
/// If multiple exist, returns an error listing them.
pub fn auto_discover_model() -> LlmResult<PathBuf> {
    let dir = default_model_dir();

    if !dir.exists() {
        return Err(LlmError::ModelNotFound(format!(
            "Model directory does not exist: {}
Please create it and place a GGUF model file inside:
  mkdir -p {}
Then download a GGUF model from https://huggingface.co/models",
            dir.display(),
            dir.display()
        )));
    }

    let mut gguf_files: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "gguf") {
                gguf_files.push(path);
            }
        }
    }

    match gguf_files.len() {
        0 => Err(LlmError::ModelNotFound(format!(
            "No GGUF model found in: {}
Please download a GGUF model and place it in this directory.
Recommended models: https://huggingface.co/models?search=GGUF",
            dir.display()
        ))),
        1 => Ok(gguf_files.remove(0)),
        _ => {
            let listing: Vec<String> = gguf_files.iter().map(|p| p.display().to_string()).collect();
            Err(LlmError::InvalidParameter(format!(
                "Multiple GGUF models found in {}:\n  {}\nPlease specify one with --model",
                dir.display(),
                listing.join("\n  ")
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model_dir() {
        let dir = default_model_dir();
        assert!(dir.to_string_lossy().contains(".codeatlas"));
        assert!(dir.to_string_lossy().contains("models"));
    }
}
