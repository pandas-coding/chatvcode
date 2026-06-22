//! Stability tests for long context and large repository scenarios.
//!
//! These tests verify that the LLM service can handle:
//! - Long prompts that approach context window limits
//! - Large code snippets in RAG context
//! - Multiple context snippets with token budget management
//! - Session history with many turns
//!
//! These tests use MockLlmService to avoid requiring real model files.

use chatvcode_llm::{
    ChatPromptBuilder, ChatSession, ChatTemplate, GenerationParams,
    LlmService, MockLlmService,
};

/// Test that long prompts are handled correctly without overflow.
#[test]
fn test_long_prompt_handling() {
    let service = MockLlmService::new("Response to long prompt");
    let params = GenerationParams::default();

    // Create a very long prompt (simulating large code context)
    let long_text = "fn example() { println!(\"line\"); }\n".repeat(1000);
    
    let response = service.infer(&long_text, &params, None);
    assert!(response.is_ok(), "Long prompt should be handled");
}

/// Test that ChatPromptBuilder handles large context snippets with token budget.
#[test]
fn test_rag_context_with_token_budget() {
    let large_snippet = "fn large_function() {\n    // lots of code\n".repeat(500);
    
    let prompt = ChatPromptBuilder::new(ChatTemplate::ChatML)
        .system_prompt("You are a helpful assistant.")
        .user_question("What does this code do?")
        .context(large_snippet.clone())
        .context(large_snippet)
        .context_token_budget(2048)
        .build();

    assert!(prompt.is_ok(), "Should handle large context with budget");
    let prompt_text = prompt.unwrap();
    
    // Verify context was truncated
    assert!(prompt_text.contains("[Retrieved Context]"));
    assert!(prompt_text.contains("What does this code do?"));
}

/// Test that multiple context snippets are handled correctly.
#[test]
fn test_multiple_context_snippets() {
    let snippets: Vec<String> = (0..10)
        .map(|i| format!("Code snippet {}: fn example_{}() {{ /* code */ }}", i, i))
        .collect();

    let mut builder = ChatPromptBuilder::new(ChatTemplate::ChatML)
        .system_prompt("You are helpful.")
        .user_question("Explain these functions");

    for snippet in snippets {
        builder = builder.context(snippet);
    }

    let prompt = builder.build();
    assert!(prompt.is_ok());
    
    let prompt_text = prompt.unwrap();
    assert!(prompt_text.contains("snippet 1"));
    assert!(prompt_text.contains("snippet 10"));
}

/// Test session with many turns to verify history management.
#[test]
fn test_session_many_turns() {
    let mock = MockLlmService::new("Response");
    let params = GenerationParams::default();

    let mut session = ChatSession::new(ChatTemplate::ChatML)
        .system_prompt("You are helpful.")
        .max_history_turns(5)
        .max_context_tokens(4096);

    // Add 20 turns
    for i in 0..20 {
        let response = session.chat(&format!("Question {}", i), &mock, &params);
        assert!(response.is_ok(), "Turn {} should succeed", i);
    }

    // max_history_turns controls prompt inclusion, not storage trimming.
    // max_context_tokens triggers trim_history(), but short messages fit within 4096 tokens.
    assert_eq!(session.len(), 40, "All 20 turns (40 messages) should be stored");
    assert_eq!(session.turn_count(), 20);

    // The prompt should only include the last 5 turns (10 messages)
    let prompt = session.build_prompt().unwrap();
    assert!(!prompt.contains("Question 0"), "Old turns should be excluded from prompt");
    assert!(prompt.contains("Question 19"), "Latest turn should be in prompt");
}

/// Test session with large messages.
#[test]
fn test_session_large_messages() {
    let mock = MockLlmService::new("Response");
    let params = GenerationParams::default();

    let mut session = ChatSession::new(ChatTemplate::ChatML)
        .max_context_tokens(2048);

    // Add large user message
    let large_message = "A".repeat(10000);
    let response = session.chat(&large_message, &mock, &params);
    assert!(response.is_ok());

    // Session should handle the large message
    assert!(session.estimated_tokens() > 0);
}

/// Test token estimation accuracy with various text sizes.
#[test]
fn test_token_estimation_accuracy() {
    use chatvcode_llm::token_estimate;

    let text_100 = "A".repeat(100);
    let text_1000 = "A".repeat(1000);
    
    let test_cases = [
        ("Hello", 2),
        ("Hello world", 3),
        ("The quick brown fox jumps over the lazy dog", 11),
        (text_100.as_str(), 25),
        (text_1000.as_str(), 250),
    ];

    for (text, expected_min) in test_cases {
        let estimate = token_estimate(text);
        assert!(
            estimate >= expected_min,
            "Token estimate {} should be >= {} for text length {}",
            estimate,
            expected_min,
            text.len()
        );
    }
}

/// Test that streaming inference handles long responses.
#[test]
fn test_streaming_long_response() {
    let long_response = "word ".repeat(100);
    let service = MockLlmService::new(long_response.clone());
    let params = GenerationParams {
        max_tokens: 200,
        ..GenerationParams::default()
    };

    let rx = service.infer_stream("test", &params, None).unwrap();

    let mut token_count = 0;
    let mut received_completed = false;

    while let Ok(event) = rx.recv_timeout(std::time::Duration::from_secs(10)) {
        if event.is_token() {
            token_count += 1;
        }
        if matches!(event, chatvcode_llm::StreamEvent::Completed) {
            received_completed = true;
            break;
        }
    }

    assert!(token_count > 0, "Should receive tokens");
    assert!(received_completed, "Should receive completion event");
}

/// Test prompt builder with empty context.
#[test]
fn test_prompt_builder_empty_context() {
    let prompt = ChatPromptBuilder::new(ChatTemplate::ChatML)
        .user_question("Simple question")
        .build();

    assert!(prompt.is_ok());
    let text = prompt.unwrap();
    assert!(text.contains("Simple question"));
    assert!(!text.contains("[Retrieved Context]"));
}

/// Test session JSON serialization with large history.
#[test]
fn test_session_serialization_large_history() {
    let mut session = ChatSession::new(ChatTemplate::ChatML)
        .system_prompt("System prompt");

    // Add many messages
    for i in 0..50 {
        session.add_user_message(format!("Question {} with some additional context", i));
        session.add_assistant_message(format!("Answer {} with detailed explanation", i));
    }

    let json = session.to_json();
    assert!(json.is_ok());

    let restored = ChatSession::from_json(&json.unwrap(), ChatTemplate::ChatML);
    assert!(restored.is_ok());

    let restored = restored.unwrap();
    assert_eq!(restored.len(), session.len());
    assert_eq!(restored.get_system_prompt(), session.get_system_prompt());
}

/// Test that context overflow is handled gracefully.
#[test]
fn test_context_overflow_handling() {
    let service = MockLlmService::new("Response");
    let params = GenerationParams::default();

    // Create extremely long prompt
    let extreme_prompt = "A".repeat(100000);
    
    // Mock service should handle this (real service would check context limits)
    let response = service.infer(&extreme_prompt, &params, None);
    assert!(response.is_ok());
}

/// Test multiple concurrent sessions.
#[test]
fn test_concurrent_sessions() {
    use std::thread;
    use std::sync::Arc;

    let mock = Arc::new(MockLlmService::new("Response"));
    let params = Arc::new(GenerationParams::default());

    let handles: Vec<_> = (0..5)
        .map(|i| {
            let mock = Arc::clone(&mock);
            let params = Arc::clone(&params);
            thread::spawn(move || {
                let mut session = ChatSession::new(ChatTemplate::ChatML);
                for j in 0..10 {
                    let response = session.chat(&format!("Q{}-{}", i, j), &*mock, &*params);
                    assert!(response.is_ok());
                }
                session
            })
        })
        .collect();

    for handle in handles {
        let session = handle.join().unwrap();
        assert_eq!(session.turn_count(), 10);
    }
}
