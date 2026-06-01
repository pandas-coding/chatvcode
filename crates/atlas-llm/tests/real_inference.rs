//! Integration tests for real model inference.
//!
//! These tests require a GGUF model file in ~/.codeatlas/models/
//! Run with: cargo test -p atlas-llm --test real_inference -- --ignored

use atlas_llm::{GenerationParams, LlamaService, LlmConfig, LlmService as _};
use std::path::PathBuf;

fn get_test_model_path() -> Option<PathBuf> {
    let dir = dirs::home_dir()?.join(".codeatlas").join("models");
    if !dir.exists() {
        return None;
    }
    // Find first .gguf file
    std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .find(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "gguf")
                .unwrap_or(false)
        })
        .map(|e| e.path())
}

/// Run with: cargo test -p atlas-llm --test real_inference -- --ignored
#[test]
#[ignore]
fn test_real_model_inference() {
    let model_path = match get_test_model_path() {
        Some(p) => p,
        None => {
            eprintln!("No GGUF model found in ~/.codeatlas/models/, skipping test");
            return;
        }
    };

    println!("Using model: {}", model_path.display());

    // Initialize backend
    atlas_llm::init();

    // Load model
    let config = LlmConfig::new(&model_path)
        .with_n_ctx(2048)
        .with_n_threads(4);
    let service = LlamaService::new(&config).expect("Failed to load model");

    // Print model info
    let info = service.model_info().unwrap();
    println!("Model: {}", info.description);
    println!("Architecture: {}", info.architecture);
    println!("Parameters: {}", info.n_params);

    // Run inference
    let params =
        GenerationParams { temperature: 0.7, max_tokens: 100, ..GenerationParams::default() };

    let response = service
        .infer("What is Rust?", &params, None)
        .expect("Inference failed");

    // Verify response
    println!("\n--- Response ---");
    println!("{}", response.text);
    println!("--- Stats ---");
    println!("Stop reason: {:?}", response.stop_reason);
    println!("Prompt tokens: {}", response.token_usage.prompt_tokens);
    println!("Completion tokens: {}", response.token_usage.completion_tokens);
    println!("Total tokens: {}", response.token_usage.total_tokens);
    println!("Duration: {:?}", response.duration);
    println!("Tokens/sec: {:.2}", response.tokens_per_second);

    // Assertions
    assert!(!response.text.is_empty(), "Response should not be empty");
    assert!(response.token_usage.prompt_tokens > 0);
    assert!(response.token_usage.completion_tokens > 0);
    assert!(response.duration.as_millis() > 0);
    assert!(response.tokens_per_second > 0.0);

    // Cleanup
    atlas_llm::shutdown();
}

#[test]
#[ignore] // Run with: cargo test -p atlas-llm --test real_inference -- --ignored
fn test_real_model_with_different_params() {
    let model_path = match get_test_model_path() {
        Some(p) => p,
        None => {
            eprintln!("No GGUF model found, skipping test");
            return;
        }
    };

    atlas_llm::init();

    let config = LlmConfig::new(&model_path)
        .with_n_ctx(2048)
        .with_n_threads(4);
    let service = LlamaService::new(&config).expect("Failed to load model");

    // Test greedy decoding
    let greedy_params = GenerationParams::greedy();
    let response1 = service.infer("Say hello", &greedy_params, None).unwrap();

    // Test with temperature
    let temp_params =
        GenerationParams { temperature: 1.0, max_tokens: 50, ..GenerationParams::default() };
    let response2 = service.infer("Say hello", &temp_params, None).unwrap();

    println!("Greedy response: {}", response1.text);
    println!("Temperature response: {}", response2.text);

    // Both should produce valid responses
    assert!(!response1.text.is_empty());
    assert!(!response2.text.is_empty());

    // Greedy should be deterministic (same input -> same output)
    let response1_again = service.infer("Say hello", &greedy_params, None).unwrap();
    assert_eq!(response1.text, response1_again.text, "Greedy decoding should be deterministic");

    atlas_llm::shutdown();
}
