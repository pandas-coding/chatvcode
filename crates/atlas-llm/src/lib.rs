//! # atlas-llm
//!
//! LLM inference engine using `llama.cpp` FFI bindings.
//!
//! This crate provides:
//! - Raw FFI bindings to the `llama.cpp` C API ([`ffi`])
//! - Safe Rust wrappers for model loading and inference ([`context`])
//! - High-level service abstraction ([`service`])
//! - Data model types for configuration, generation, and responses ([`types`])
//!
//! ## Quick Start
//!
//! ```ignore
//! use atlas_llm::{LlmConfig, LlamaService, LlmService as _};
//!
//! // Load a model
//! let config = LlmConfig::new("~/.codeatlas/models/codellama-7b.gguf");
//! let service = LlamaService::new(&config)?;
//!
//! // Sync inference
//! let response = service.infer(
//!     "Explain Rust lifetimes",
//!     &GenerationParams::default(),
//!     None,
//! )?;
//! println!("{}", response.text);
//! ```
//!
//! ## Backend Abstraction
//!
//! The [`LlmService`] trait defines the interface that consumers
//! (atlas-core, atlas-cli) use. Backend implementations (e.g.,
//! [`LlamaService`]) live in the [`service`] module.
//!
//! This decoupling ensures that upper layers never depend on
//! `llama.cpp` FFI details directly.

pub mod context;
pub mod error;
pub mod ffi;
pub mod gguf;
pub mod service;
pub mod types;

// Re-export key types for convenience
pub use context::{LlamaContext, LlamaModel};
pub use error::{LlmError, LlmResult};
pub use gguf::{
    GGUF_MAGIC, GgufHeader, GgufMetadata, SUPPORTED_VERSIONS, discover_gguf_models,
    format_file_size, format_gguf_summary, format_param_count, infer_chat_template, is_gguf_file,
    load_model_safe, pre_validate_model, read_gguf_metadata, scan_model, validate_gguf,
};
pub use service::{
    LlamaService, LlmService, MockLlmService, auto_discover_model, dedent, default_model_dir,
};
pub use types::{
    ChatMessage, ChatTemplate, GenerationParams, InferenceResponse, LlmConfig, ModelInfo,
    StopReason, StreamEvent, TokenUsage,
};

/// Registered backend information discovered from ggml/llama.cpp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendInfo {
    pub name: String,
    pub devices: Vec<String>,
}

/// Initialize the llama.cpp backend.
///
/// Must be called once before any other LLM operations.
/// It is safe to call this multiple times (idempotent).
pub fn init() {
    unsafe { ffi::llama_backend_init() };
}

/// Shut down the llama.cpp backend.
///
/// Should be called once when the program exits.
pub fn shutdown() {
    unsafe { ffi::llama_backend_free() };
}

/// Returns whether llama.cpp reports GPU offload support at runtime.
#[must_use]
pub fn supports_gpu_offload() -> bool {
    unsafe { ffi::llama_supports_gpu_offload() }
}

/// Enumerate registered ggml backends and their visible devices.
#[must_use]
pub fn available_backends() -> Vec<BackendInfo> {
    let mut backends = Vec::new();
    let count = unsafe { ffi::ggml_backend_reg_count() };

    for i in 0..count {
        let reg = unsafe { ffi::ggml_backend_reg_get(i) };
        if reg.is_null() {
            continue;
        }

        let name = unsafe { crate::error::cstr_to_string(ffi::ggml_backend_reg_name(reg)) }
            .unwrap_or_else(|| "unknown".to_string());

        let mut devices = Vec::new();
        let dev_count = unsafe { ffi::ggml_backend_reg_dev_count(reg) };
        for j in 0..dev_count {
            let dev = unsafe { ffi::ggml_backend_reg_dev_get(reg, j) };
            if dev.is_null() {
                continue;
            }

            let dev_name = unsafe { crate::error::cstr_to_string(ffi::ggml_backend_dev_name(dev)) }
                .unwrap_or_else(|| "unknown".to_string());
            let dev_desc =
                unsafe { crate::error::cstr_to_string(ffi::ggml_backend_dev_description(dev)) }
                    .unwrap_or_default();

            if dev_desc.is_empty() || dev_desc == dev_name {
                devices.push(dev_name);
            } else {
                devices.push(format!("{dev_name} ({dev_desc})"));
            }
        }

        backends.push(BackendInfo { name, devices });
    }

    backends
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_shutdown_idempotent() {
        // Multiple calls should be safe
        init();
        init();
        shutdown();
        shutdown();
    }

    #[test]
    fn test_module_structure() {
        // Verify key types are accessible
        let _config = LlmConfig::default();
        let _params = GenerationParams::default();
        let _msg = ChatMessage::user("test");
        let _template = ChatTemplate::Auto;
    }

    #[test]
    fn test_available_backends() {
        init();

        let backends = available_backends();
        assert!(!backends.is_empty(), "no ggml backends registered");
        assert!(backends.iter().any(|b| b.name.eq_ignore_ascii_case("cpu")));

        if std::env::var("LLAMA_CUDA").ok().as_deref() == Some("1") {
            assert!(
                backends.iter().any(|b| b.name.eq_ignore_ascii_case("cuda")),
                "CUDA requested but backend not registered: {backends:#?}"
            );
        }

        if std::env::var("LLAMA_VULKAN").ok().as_deref() == Some("1") {
            assert!(
                backends
                    .iter()
                    .any(|b| b.name.eq_ignore_ascii_case("vulkan")),
                "Vulkan requested but backend not registered: {backends:#?}"
            );
        }
    }
}
