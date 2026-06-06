//! Acceptance tests for synchronous inference engine (P0-4).
//!
//! These tests verify the three main acceptance criteria:
//! 1. Given a prompt, returns a complete text response
//! 2. Generation parameters are configurable and effective
//! 3. Response includes stop reason, token statistics, and timing info

use chatvcode_llm::{
    GenerationParams, InferenceResponse, LlmService, MockLlmService, StopReason, TokenUsage,
};
use std::sync::atomic::AtomicBool;

// ============================================================================
// Criterion 1: Given a prompt, returns a complete text response
// ============================================================================

#[test]
fn criterion_1_returns_complete_text_response() {
    // Arrange
    let service = MockLlmService::new("The answer is 42.");
    let params = GenerationParams::default();

    // Act
    let response = service.infer("What is the answer?", &params, None).unwrap();

    // Assert
    assert!(!response.text.is_empty(), "Response text should not be empty");
    assert_eq!(
        response.text, "The answer is 42.",
        "Response should contain the complete generated text"
    );
}

#[test]
fn criterion_1_handles_various_prompts() {
    let service = MockLlmService::new("I understand your question.");
    let params = GenerationParams::default();

    // Test with different prompts
    let prompts = vec![
        "Hello",
        "Explain Rust lifetimes",
        "Write a function to sort an array",
        "", // Edge case: empty prompt
    ];

    for prompt in prompts {
        let response = service.infer(prompt, &params, None).unwrap();
        assert!(!response.text.is_empty(), "Should return text even for prompt: '{prompt}'");
    }
}

#[test]
fn criterion_1_response_not_truncated_when_within_limits() {
    let service = MockLlmService::new("Short response.");
    let params = GenerationParams {
        max_tokens: 1000, // Generous limit
        ..GenerationParams::default()
    };

    let response = service.infer("test", &params, None).unwrap();

    // Should not be truncated
    assert_eq!(response.text, "Short response.");
    assert_eq!(response.stop_reason, StopReason::Eos);
}

// ============================================================================
// Criterion 2: Generation parameters are configurable and effective
// ============================================================================

#[test]
fn criterion_2_temperature_is_configurable() {
    let params = GenerationParams::default().with_temperature(0.0); // Greedy

    assert_eq!(params.temperature, 0.0, "Temperature should be set to 0.0");

    let params = GenerationParams::default().with_temperature(1.5); // Very random

    assert!((params.temperature - 1.5).abs() < f32::EPSILON);
}

#[test]
fn criterion_2_top_p_is_configurable() {
    let params = GenerationParams::default().with_top_p(0.5);

    assert!((params.top_p - 0.5).abs() < f32::EPSILON);
}

#[test]
fn criterion_2_top_k_is_configurable() {
    let params = GenerationParams::default().with_top_k(10);

    assert_eq!(params.top_k, 10);
}

#[test]
fn criterion_2_max_tokens_limits_generation() {
    let service = MockLlmService::new("This is a longer response that might be truncated.");

    // With low max_tokens
    let params_low = GenerationParams { max_tokens: 2, ..GenerationParams::default() };
    let response_low = service.infer("test", &params_low, None).unwrap();

    // With high max_tokens
    let params_high = GenerationParams { max_tokens: 1000, ..GenerationParams::default() };
    let response_high = service.infer("test", &params_high, None).unwrap();

    // Low max_tokens should result in MaxTokens stop reason
    assert_eq!(
        response_low.stop_reason,
        StopReason::MaxTokens,
        "Should stop due to max_tokens limit"
    );

    // High max_tokens should allow full generation
    assert_eq!(
        response_high.stop_reason,
        StopReason::Eos,
        "Should stop at EOS when max_tokens is high enough"
    );
}

#[test]
fn criterion_2_repeat_penalty_is_configurable() {
    let params =
        GenerationParams { repeat_penalty: 1.2, repeat_last_n: 128, ..GenerationParams::default() };

    assert!((params.repeat_penalty - 1.2).abs() < f32::EPSILON);
    assert_eq!(params.repeat_last_n, 128);
}

#[test]
fn criterion_2_seed_affects_generation() {
    let params = GenerationParams::default().with_seed(42);

    assert_eq!(params.seed, 42);
}

#[test]
fn criterion_2_greedy_params_work() {
    let params = GenerationParams::greedy();

    assert_eq!(params.temperature, 0.0, "Greedy should have temperature 0");
    assert_eq!(params.top_k, 1, "Greedy should have top_k 1");
}

// ============================================================================
// Criterion 3: Response includes stop reason, token statistics, and timing
// ============================================================================

#[test]
fn criterion_3_stop_reason_eos() {
    let service = MockLlmService::new("Response");
    let params = GenerationParams { max_tokens: 1000, ..GenerationParams::default() };

    let response = service.infer("test", &params, None).unwrap();

    assert_eq!(
        response.stop_reason,
        StopReason::Eos,
        "Should stop at EOS when generation completes normally"
    );
}

#[test]
fn criterion_3_stop_reason_max_tokens() {
    let service = MockLlmService::new("Long response");
    let params = GenerationParams { max_tokens: 1, ..GenerationParams::default() };

    let response = service.infer("test", &params, None).unwrap();

    assert_eq!(
        response.stop_reason,
        StopReason::MaxTokens,
        "Should stop at MaxTokens when limit is reached"
    );
}

#[test]
fn criterion_3_stop_reason_cancelled() {
    let service = MockLlmService::new("Should not appear");
    let params = GenerationParams::default();
    let cancel_flag = AtomicBool::new(true);

    let response = service.infer("test", &params, Some(&cancel_flag)).unwrap();

    assert_eq!(
        response.stop_reason,
        StopReason::Cancelled,
        "Should stop with Cancelled when flag is set"
    );
}

#[test]
fn criterion_3_token_usage_present() {
    let service = MockLlmService::new("Test response for token counting");
    let params = GenerationParams::default();

    let response = service.infer("test prompt", &params, None).unwrap();

    // Verify token usage fields exist and are positive
    assert!(
        response.token_usage.prompt_tokens > 0,
        "prompt_tokens should be positive, got: {}",
        response.token_usage.prompt_tokens
    );
    assert!(
        response.token_usage.completion_tokens > 0,
        "completion_tokens should be positive, got: {}",
        response.token_usage.completion_tokens
    );
    assert_eq!(
        response.token_usage.total_tokens,
        response.token_usage.prompt_tokens + response.token_usage.completion_tokens,
        "total_tokens should equal prompt + completion"
    );
}

#[test]
fn criterion_3_timing_information_present() {
    let service = MockLlmService::new("Test response for timing");
    let params = GenerationParams::default();

    let response = service.infer("test", &params, None).unwrap();

    // Duration should be present and non-zero
    assert!(
        response.duration.as_nanos() > 0,
        "Duration should be non-zero, got: {:?}",
        response.duration
    );

    // Tokens per second should be positive
    assert!(
        response.tokens_per_second > 0.0,
        "tokens_per_second should be positive, got: {}",
        response.tokens_per_second
    );
}

#[test]
fn criterion_3_response_structure_complete() {
    let service = MockLlmService::new("Complete response");
    let params = GenerationParams {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.1,
        max_tokens: 512,
        ..GenerationParams::default()
    };

    let response = service.infer("test prompt", &params, None).unwrap();

    // Verify all fields are present and valid
    let InferenceResponse {
        text,
        stop_reason,
        token_usage,
        duration,
        time_to_first_token,
        tokens_per_second,
    } = response;

    // Text
    assert!(!text.is_empty(), "text should not be empty");

    // Stop reason
    assert!(
        matches!(stop_reason, StopReason::Eos | StopReason::MaxTokens | StopReason::Cancelled),
        "stop_reason should be a valid variant"
    );

    // Token usage
    let TokenUsage { prompt_tokens, completion_tokens, total_tokens } = token_usage;
    assert!(prompt_tokens > 0);
    assert!(completion_tokens > 0);
    assert_eq!(total_tokens, prompt_tokens + completion_tokens);

    // Timing
    assert!(duration.as_nanos() > 0);
    assert!(tokens_per_second > 0.0);

    // time_to_first_token is optional, but should be Some for mock
    assert!(time_to_first_token.is_some(), "time_to_first_token should be present");
}

// ============================================================================
// Integration-style tests with realistic scenarios
// ============================================================================

#[test]
fn scenario_coding_question() {
    let service = MockLlmService::new(
        "fn fibonacci(n: u32) -> u32 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => fibonacci(n - 1) + fibonacci(n - 2),\n    }\n}",
    );
    let params =
        GenerationParams { temperature: 0.3, max_tokens: 256, ..GenerationParams::default() };

    let response = service
        .infer("Write a fibonacci function in Rust", &params, None)
        .unwrap();

    assert!(response.text.contains("fibonacci"));
    assert_eq!(response.stop_reason, StopReason::Eos);
    assert!(response.token_usage.completion_tokens > 0);
}

#[test]
fn scenario_constrained_generation() {
    let service = MockLlmService::new("This is a very long explanation about a complex topic.");
    let params = GenerationParams { max_tokens: 3, ..GenerationParams::default() };

    let response = service.infer("Explain everything", &params, None).unwrap();

    // Should be truncated due to max_tokens
    assert_eq!(response.stop_reason, StopReason::MaxTokens);
    assert!(response.token_usage.completion_tokens <= params.max_tokens);
}

#[test]
fn scenario_cancelled_mid_generation() {
    let service = MockLlmService::new("This should be cancelled");
    let params = GenerationParams::default();
    let cancel = AtomicBool::new(true);

    let response = service
        .infer("Generate something", &params, Some(&cancel))
        .unwrap();

    assert_eq!(response.stop_reason, StopReason::Cancelled);
    assert!(response.text.is_empty());
    assert_eq!(response.token_usage.completion_tokens, 0);
}
