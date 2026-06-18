//! Unit tests for `LlmConfig`, `ModelInfo`, `GenerationParams` and stopping conditions.

use chatvcode_llm::{
    GenerationParams, LlmConfig, LlmService, MockLlmService, ModelInfo, StopReason,
    StreamEvent, TokenUsage,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

// ============================================================================
// LlmConfig unit tests
// ============================================================================

#[test]
fn llm_config_default_values() {
    let config = LlmConfig::default();
    assert_eq!(config.n_ctx, 8192);
    assert_eq!(config.n_batch, 8192);
    assert_eq!(config.n_ubatch, 512);
    assert!(config.n_threads > 0);
    assert!(config.n_threads_batch > 0);
    assert_eq!(config.n_gpu_layers, 0);
    assert!(config.use_mmap);
    assert!(!config.use_mlock);
    assert!(config.chat_template.is_none());
    assert!(!config.verbose_log);
    assert_eq!(config.model_path, PathBuf::new());
}

#[test]
fn llm_config_new_sets_model_path() {
    let config = LlmConfig::new("/models/test.gguf");
    assert_eq!(config.model_path, PathBuf::from("/models/test.gguf"));
    assert_eq!(config.n_ctx, 8192);
}

#[test]
fn llm_config_with_n_ctx_auto_upgrades_n_batch() {
    let config = LlmConfig::new("test.gguf").with_n_batch(512).with_n_ctx(16384);
    assert_eq!(config.n_ctx, 16384);
    assert_eq!(config.n_batch, 16384);
}

#[test]
fn llm_config_with_n_batch_smaller_than_n_ctx_is_upgraded() {
    let config = LlmConfig::new("test.gguf").with_n_ctx(8192).with_n_batch(1024);
    assert_eq!(config.n_batch, 8192);
}

#[test]
fn llm_config_with_n_batch_larger_than_n_ctx_preserved() {
    let config = LlmConfig::new("test.gguf").with_n_ctx(2048).with_n_batch(16384);
    assert_eq!(config.n_ctx, 2048);
    assert_eq!(config.n_batch, 16384);
}

#[test]
fn llm_config_with_n_threads_sets_both() {
    let config = LlmConfig::new("test.gguf").with_n_threads(16);
    assert_eq!(config.n_threads, 16);
    assert_eq!(config.n_threads_batch, 16);
}

#[test]
fn llm_config_with_n_gpu_layers() {
    let config = LlmConfig::new("test.gguf").with_n_gpu_layers(-1);
    assert_eq!(config.n_gpu_layers, -1);
    let config = LlmConfig::new("test.gguf").with_n_gpu_layers(35);
    assert_eq!(config.n_gpu_layers, 35);
}

#[test]
fn llm_config_with_mmap() {
    let config = LlmConfig::new("test.gguf").with_mmap(false);
    assert!(!config.use_mmap);
    let config = LlmConfig::new("test.gguf").with_mmap(true);
    assert!(config.use_mmap);
}

#[test]
fn llm_config_with_chat_template() {
    let config = LlmConfig::new("test.gguf").with_chat_template("chatml");
    assert_eq!(config.chat_template.as_deref(), Some("chatml"));
    let config = LlmConfig::new("test.gguf").with_chat_template("llama3");
    assert_eq!(config.chat_template.as_deref(), Some("llama3"));
}

#[test]
fn llm_config_with_verbose_log() {
    let config = LlmConfig::new("test.gguf").with_verbose_log(true);
    assert!(config.verbose_log);
}

#[test]
fn llm_config_builder_chain() {
    let config = LlmConfig::new("/path/to/model.gguf")
        .with_n_ctx(4096)
        .with_n_threads(8)
        .with_n_gpu_layers(32)
        .with_mmap(false)
        .with_chat_template("llama3")
        .with_verbose_log(true);

    assert_eq!(config.model_path, PathBuf::from("/path/to/model.gguf"));
    assert_eq!(config.n_ctx, 4096);
    assert_eq!(config.n_threads, 8);
    assert_eq!(config.n_gpu_layers, 32);
    assert!(!config.use_mmap);
    assert_eq!(config.chat_template.as_deref(), Some("llama3"));
    assert!(config.verbose_log);
}

// ============================================================================
// ModelInfo unit tests
// ============================================================================

#[test]
fn model_info_fields_accessible() {
    let info = ModelInfo {
        description: "Qwen2.5-Coder-7B-Instruct".into(),
        architecture: "qwen2".into(),
        n_params: 7_615_574_016,
        size_bytes: 4_431_000_000,
        n_ctx_train: 32768,
        n_embd: 3584,
        n_layer: 28,
        n_head: 28,
        n_head_kv: 4,
        n_vocab: 151936,
        vocab_type: "bpe".into(),
        ftype: "Q4_K_M".into(),
        chat_template_available: true,
        rope_type: "norm".into(),
        has_encoder: false,
        has_decoder: true,
    };

    assert_eq!(info.description, "Qwen2.5-Coder-7B-Instruct");
    assert_eq!(info.architecture, "qwen2");
    assert_eq!(info.n_params, 7_615_574_016);
    assert_eq!(info.n_ctx_train, 32768);
    assert_eq!(info.n_embd, 3584);
    assert_eq!(info.n_layer, 28);
    assert_eq!(info.n_head, 28);
    assert_eq!(info.n_head_kv, 4);
    assert_eq!(info.n_vocab, 151936);
    assert!(info.chat_template_available);
    assert!(!info.has_encoder);
    assert!(info.has_decoder);
}

#[test]
fn model_info_clone_and_debug() {
    let info = ModelInfo {
        description: "test".into(),
        architecture: "llama".into(),
        n_params: 1_000_000,
        size_bytes: 500_000,
        n_ctx_train: 4096,
        n_embd: 2048,
        n_layer: 16,
        n_head: 16,
        n_head_kv: 16,
        n_vocab: 32000,
        vocab_type: "bpe".into(),
        ftype: "f16".into(),
        chat_template_available: false,
        rope_type: "0".into(),
        has_encoder: true,
        has_decoder: false,
    };

    let cloned = info.clone();
    assert_eq!(cloned.description, info.description);
    assert_eq!(cloned.n_params, info.n_params);

    let debug_str = format!("{info:?}");
    assert!(debug_str.contains("llama"));
}

#[test]
fn mock_service_model_info() {
    let service = MockLlmService::new("test");
    let info = service.model_info().unwrap();

    assert_eq!(info.architecture, "mock");
    assert!(info.n_params > 0);
    assert!(info.n_vocab > 0);
    assert!(info.chat_template_available);
}

// ============================================================================
// GenerationParams unit tests
// ============================================================================

#[test]
fn generation_params_default_values() {
    let params = GenerationParams::default();
    assert!((params.temperature - 0.7).abs() < f32::EPSILON);
    assert!((params.top_p - 0.9).abs() < f32::EPSILON);
    assert_eq!(params.top_k, 40);
    assert!((params.min_p - 0.0).abs() < f32::EPSILON);
    assert!((params.repeat_penalty - 1.1).abs() < f32::EPSILON);
    assert_eq!(params.repeat_last_n, 64);
    assert_eq!(params.max_tokens, 2048);
    assert!(params.stop_strings.is_empty());
    assert_eq!(params.seed, u32::MAX);
}

#[test]
fn generation_params_greedy() {
    let params = GenerationParams::greedy();
    assert_eq!(params.temperature, 0.0);
    assert_eq!(params.top_k, 1);
    assert_eq!(params.top_p, 1.0);
    assert_eq!(params.min_p, 0.0);
}

#[test]
fn generation_params_with_temperature() {
    let params = GenerationParams::default().with_temperature(0.0);
    assert_eq!(params.temperature, 0.0);
    let params = GenerationParams::default().with_temperature(2.0);
    assert!((params.temperature - 2.0).abs() < f32::EPSILON);
}

#[test]
fn generation_params_with_top_p() {
    let params = GenerationParams::default().with_top_p(0.5);
    assert!((params.top_p - 0.5).abs() < f32::EPSILON);
    let params = GenerationParams::default().with_top_p(1.0);
    assert!((params.top_p - 1.0).abs() < f32::EPSILON);
}

#[test]
fn generation_params_with_top_k() {
    let params = GenerationParams::default().with_top_k(1);
    assert_eq!(params.top_k, 1);
    let params = GenerationParams::default().with_top_k(100);
    assert_eq!(params.top_k, 100);
    let params = GenerationParams::default().with_top_k(0);
    assert_eq!(params.top_k, 0);
}

#[test]
fn generation_params_with_max_tokens() {
    let params = GenerationParams::default().with_max_tokens(1);
    assert_eq!(params.max_tokens, 1);
    let params = GenerationParams::default().with_max_tokens(4096);
    assert_eq!(params.max_tokens, 4096);
}

#[test]
fn generation_params_with_seed() {
    let params = GenerationParams::default().with_seed(42);
    assert_eq!(params.seed, 42);
    let params = GenerationParams::default().with_seed(0);
    assert_eq!(params.seed, 0);
}

#[test]
fn generation_params_builder_chain() {
    let params = GenerationParams::default()
        .with_temperature(0.3)
        .with_top_p(0.95)
        .with_top_k(50)
        .with_max_tokens(1024)
        .with_seed(12345);

    assert!((params.temperature - 0.3).abs() < f32::EPSILON);
    assert!((params.top_p - 0.95).abs() < f32::EPSILON);
    assert_eq!(params.top_k, 50);
    assert_eq!(params.max_tokens, 1024);
    assert_eq!(params.seed, 12345);
}

#[test]
fn generation_params_clone_independent() {
    let params1 = GenerationParams::default().with_temperature(0.5);
    let mut params2 = params1.clone();
    params2.temperature = 1.5;
    assert!((params1.temperature - 0.5).abs() < f32::EPSILON);
    assert!((params2.temperature - 1.5).abs() < f32::EPSILON);
}

#[test]
fn generation_params_stop_strings() {
    let mut params = GenerationParams::default();
    assert!(params.stop_strings.is_empty());
    params.stop_strings = vec!["<|end|>".to_string(), "STOP".to_string()];
    assert_eq!(params.stop_strings.len(), 2);
    assert_eq!(params.stop_strings[0], "<|end|>");
}

// ============================================================================
// TokenUsage unit tests
// ============================================================================

#[test]
fn token_usage_new() {
    let usage = TokenUsage::new(100, 50);
    assert_eq!(usage.prompt_tokens, 100);
    assert_eq!(usage.completion_tokens, 50);
    assert_eq!(usage.total_tokens, 150);
}

#[test]
fn token_usage_default() {
    let usage = TokenUsage::default();
    assert_eq!(usage.prompt_tokens, 0);
    assert_eq!(usage.completion_tokens, 0);
    assert_eq!(usage.total_tokens, 0);
}

#[test]
fn token_usage_equality() {
    let a = TokenUsage::new(10, 20);
    let b = TokenUsage::new(10, 20);
    assert_eq!(a, b);
    let c = TokenUsage::new(10, 21);
    assert_ne!(a, c);
}

// ============================================================================
// StopReason unit tests
// ============================================================================

#[test]
fn stop_reason_variants() {
    let eos = StopReason::Eos;
    let max_tok = StopReason::MaxTokens;
    let stop_str = StopReason::StopString("<|end|>".to_string());
    let cancelled = StopReason::Cancelled;
    let error = StopReason::Error("OOM".to_string());

    assert_eq!(eos, StopReason::Eos);
    assert_eq!(max_tok, StopReason::MaxTokens);
    assert_eq!(stop_str, StopReason::StopString("<|end|>".to_string()));
    assert_ne!(stop_str, StopReason::StopString("other".to_string()));
    assert_eq!(cancelled, StopReason::Cancelled);
    assert_eq!(error, StopReason::Error("OOM".to_string()));
}

#[test]
fn stop_reason_debug_display() {
    let reason = StopReason::StopString("END".to_string());
    let s = format!("{reason:?}");
    assert!(s.contains("END"));
}

// ============================================================================
// Stopping conditions tests (sync inference)
// ============================================================================

#[test]
fn stopping_eos_when_response_fits() {
    let service = MockLlmService::new("Short answer");
    let params = GenerationParams { max_tokens: 1000, ..GenerationParams::default() };
    let response = service.infer("prompt", &params, None).unwrap();
    assert_eq!(response.stop_reason, StopReason::Eos);
}

#[test]
fn stopping_max_tokens_when_limit_reached() {
    let service = MockLlmService::new("This is a longer response text that exceeds the limit");
    let params = GenerationParams { max_tokens: 2, ..GenerationParams::default() };
    let response = service.infer("prompt", &params, None).unwrap();
    assert_eq!(response.stop_reason, StopReason::MaxTokens);
}

#[test]
fn stopping_cancelled_via_atomic_bool() {
    let service = MockLlmService::new("Should not appear");
    let params = GenerationParams::default();
    let cancel = AtomicBool::new(true);
    let response = service.infer("prompt", &params, Some(&cancel)).unwrap();
    assert_eq!(response.stop_reason, StopReason::Cancelled);
    assert!(response.text.is_empty());
    assert_eq!(response.token_usage.completion_tokens, 0);
}

#[test]
fn stopping_cancelled_mid_generation() {
    let service = MockLlmService::new("word1 word2 word3 word4 word5").with_tokens_per_second(2.0);
    let params = GenerationParams::default();
    let cancel = Arc::new(AtomicBool::new(false));
    let rx = service.infer_stream("prompt", &params, Some(cancel.clone())).unwrap();

    let mut count = 0;
    while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
        match event {
            StreamEvent::Token(_) => {
                count += 1;
                if count >= 1 {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            StreamEvent::Cancelled => break,
            StreamEvent::Completed => break,
            _ => {}
        }
    }
    assert!(count < 5, "Should have stopped before all tokens");
}

#[test]
fn stopping_stream_max_tokens() {
    let service = MockLlmService::new("one two three four five six seven eight");
    let params = GenerationParams { max_tokens: 3, ..GenerationParams::default() };
    let rx = service.infer_stream("prompt", &params, None).unwrap();

    let mut token_count = 0;
    while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
        if event.is_token() {
            token_count += 1;
        }
    }
    assert!(token_count <= 3, "Should not exceed max_tokens, got {token_count}");
}

#[test]
fn stopping_stream_eos_completes_normally() {
    let service = MockLlmService::new("Hello world");
    let params = GenerationParams { max_tokens: 1000, ..GenerationParams::default() };
    let rx = service.infer_stream("prompt", &params, None).unwrap();

    let mut events = Vec::new();
    while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
        events.push(event);
    }

    assert_eq!(events.first(), Some(&StreamEvent::Started));
    assert_eq!(events.last(), Some(&StreamEvent::Completed));
}

use std::sync::Arc;
