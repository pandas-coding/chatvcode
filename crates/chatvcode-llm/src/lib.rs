//! # chatvcode-llm
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
//! use chatvcode_llm::{LlmConfig, LlamaService, LlmService as _};
//!
//! // Load a model
//! let config = LlmConfig::new("~/.chatvcode/models/codellama-7b.gguf");
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
//! (chatvcode-core, chatvcode-cli) use. Backend implementations (e.g.,
//! [`LlamaService`]) live in the [`service`] module.
//!
//! This decoupling ensures that upper layers never depend on
//! `llama.cpp` FFI details directly.

pub mod backend;
pub mod chat;
pub mod context;
pub mod error;
pub mod ffi;
pub mod gguf;
pub mod log;
pub mod model;
pub mod sampler;
pub mod service;
pub mod types;

// Re-export key types for convenience
pub use backend::{BackendInfo, available_backends, init, shutdown, supports_gpu_offload};
pub use context::{LlamaContext, LlamaEmbeddingContext, LlamaModel};
pub use error::{LlmError, LlmResult};
pub use gguf::{
    GGUF_MAGIC, GgufHeader, GgufMetadata, SUPPORTED_VERSIONS, discover_gguf_models,
    format_file_size, format_gguf_summary, format_param_count, infer_chat_template, is_gguf_file,
    load_model_safe, pre_validate_model, read_gguf_metadata, scan_model, validate_gguf,
};
pub use model::{
    ChatvcodeConfig, ChatConfig, DiscoveredModel, GenerationConfig, GpuRecommendation,
    MemoryEstimate, ModelConfig, ModelSource, default_config_path, estimate_memory,
    estimate_memory_from_metadata, list_models, list_models_in_dir, local_config_path,
    local_model_dir, model_search_dirs, recommend_gpu_layers, recommend_gpu_layers_from_metadata,
    format_bytes,
};
pub use service::{
    KvCacheState, LlamaEmbeddingService, LlamaService, LlmService, MockLlmService,
    auto_discover_model, dedent, default_model_dir,
};
pub use types::{
    ChatMessage, ChatPromptBuilder, ChatSession, ChatTemplate, GenerationParams, InferenceResponse,
    LlmConfig, ModelInfo, StopReason, StreamEvent, TokenUsage, token_estimate,
    token_estimate_messages,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        let _config = LlmConfig::default();
        let _params = GenerationParams::default();
        let _msg = ChatMessage::user("test");
        let _template = ChatTemplate::Auto;
        let _builder = ChatPromptBuilder::new(ChatTemplate::ChatML);
        let _session = ChatSession::new(ChatTemplate::Auto);
    }
}
