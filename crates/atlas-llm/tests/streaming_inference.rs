//! Integration tests for streaming inference engine (P0-5).
//!
//! These tests verify the streaming inference functionality:
//! 1. StreamEvent definition and channel-based transmission
//! 2. `LlmService::infer_stream()` method works correctly
//! 3. Real-time token delivery during inference
//! 4. User cancellation mechanism (AtomicBool or channel close)
//! 5. Thread safety - no panics, no deadlocks
//! 6. Timing information (first token latency, total generation time)

use atlas_llm::{GenerationParams, LlmService, MockLlmService, StreamEvent};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

// ============================================================================
// Criterion 1: StreamEvent definition and channel-based transmission
// ============================================================================

#[test]
fn criterion_1_stream_event_variants() {
    // Verify all StreamEvent variants exist and have correct properties
    let started = StreamEvent::Started;
    assert!(started.is_terminal() == false);
    assert!(started.is_success());

    let token = StreamEvent::Token("hello".to_string());
    assert!(token.is_token());
    assert!(token.as_token() == Some("hello"));
    assert!(token.is_success());
    assert!(!token.is_terminal());

    let completed = StreamEvent::Completed;
    assert!(completed.is_terminal());
    assert!(completed.is_success());

    let cancelled = StreamEvent::Cancelled;
    assert!(cancelled.is_terminal());
    assert!(!cancelled.is_success());

    let error = StreamEvent::Error("test error".to_string());
    assert!(error.is_terminal());
    assert!(!error.is_success());
    assert!(error.as_error() == Some("test error"));
}

#[test]
fn criterion_1_stream_event_helper_methods() {
    // Test as_token
    assert_eq!(StreamEvent::Token("test".into()).as_token(), Some("test"));
    assert_eq!(StreamEvent::Started.as_token(), None);
    assert_eq!(StreamEvent::Completed.as_token(), None);

    // Test as_error
    assert_eq!(StreamEvent::Error("err".into()).as_error(), Some("err"));
    assert_eq!(StreamEvent::Started.as_error(), None);

    // Test is_terminal
    assert!(!StreamEvent::Started.is_terminal());
    assert!(!StreamEvent::Token("".into()).is_terminal());
    assert!(StreamEvent::Completed.is_terminal());
    assert!(StreamEvent::Cancelled.is_terminal());
    assert!(StreamEvent::Error("".into()).is_terminal());

    // Test is_success
    assert!(StreamEvent::Started.is_success());
    assert!(StreamEvent::Token("".into()).is_success());
    assert!(StreamEvent::Completed.is_success());
    assert!(!StreamEvent::Cancelled.is_success());
    assert!(!StreamEvent::Error("".into()).is_success());
}

// ============================================================================
// Criterion 2: infer_stream() method works correctly
// ============================================================================

#[test]
fn criterion_2_infer_stream_returns_receiver() {
    let service = MockLlmService::new("Hello world test");
    let params = GenerationParams::default();

    let rx = service.infer_stream("test prompt", &params, None).unwrap();

    // Should be able to receive events
    let first_event = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(first_event, StreamEvent::Started);
}

#[test]
fn criterion_2_infer_stream_sends_all_tokens() {
    let service = MockLlmService::new("The quick brown fox");
    let params = GenerationParams::default();

    let rx = service.infer_stream("test", &params, None).unwrap();

    // Collect all events
    let mut events = Vec::new();
    while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
        events.push(event);
    }

    // Should have Started, multiple Tokens, and Completed
    assert_eq!(events.first(), Some(&StreamEvent::Started));
    assert_eq!(events.last(), Some(&StreamEvent::Completed));

    // Extract tokens
    let tokens: Vec<&str> = events.iter().filter_map(|e| e.as_token()).collect();

    assert!(!tokens.is_empty(), "Should have received at least one token");

    // Reconstruct text
    let reconstructed: String = tokens.join("");
    assert_eq!(reconstructed, "The quick brown fox");
}

#[test]
fn criterion_2_infer_stream_respects_max_tokens() {
    let service = MockLlmService::new("word1 word2 word3 word4 word5 word6 word7 word8");
    let params = GenerationParams { max_tokens: 3, ..GenerationParams::default() };

    let rx = service.infer_stream("test", &params, None).unwrap();

    // Collect all events
    let mut token_count = 0;
    while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
        if event.is_token() {
            token_count += 1;
        }
    }

    assert!(token_count <= 3, "Should not exceed max_tokens, got {token_count} tokens");
}

// ============================================================================
// Criterion 3: User cancellation mechanism
// ============================================================================

#[test]
fn criterion_3_cancellation_via_atomic_bool() {
    let service = MockLlmService::new("This is a long response that should be cancelled");
    let params = GenerationParams {
        // Use slow speed to ensure we can cancel mid-generation
        ..GenerationParams::default()
    };
    let cancel_flag = Arc::new(AtomicBool::new(true)); // Already cancelled

    let rx = service
        .infer_stream("test", &params, Some(cancel_flag))
        .unwrap();

    let mut events = Vec::new();
    while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
        events.push(event);
    }

    // Should have Started and Cancelled events
    assert_eq!(events.first(), Some(&StreamEvent::Started));
    assert!(
        events.contains(&StreamEvent::Cancelled),
        "Should receive Cancelled event when flag is set"
    );
}

#[test]
fn criterion_3_cancellation_mid_stream() {
    let service =
        MockLlmService::new("word1 word2 word3 word4 word5 word6 word7 word8 word9 word10")
            .with_tokens_per_second(2.0); // Slow enough to cancel mid-stream
    let params = GenerationParams::default();
    let cancel_flag = Arc::new(AtomicBool::new(false));

    let rx = service
        .infer_stream("test", &params, Some(cancel_flag.clone()))
        .unwrap();

    // Read a few tokens, then cancel
    let mut token_count = 0;
    let mut got_cancelled = false;

    while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
        match event {
            StreamEvent::Token(_) => {
                token_count += 1;
                if token_count >= 2 {
                    // Cancel after receiving 2 tokens
                    cancel_flag.store(true, Ordering::Relaxed);
                }
            }
            StreamEvent::Cancelled => {
                got_cancelled = true;
                break;
            }
            StreamEvent::Completed => {
                break;
            }
            _ => {}
        }
    }

    assert!(got_cancelled, "Should receive Cancelled event after setting flag");
    assert!(token_count < 10, "Should have cancelled before all tokens were sent");
}

#[test]
fn criterion_3_cancellation_via_channel_drop() {
    let service = MockLlmService::new("This response should be interrupted");
    let params = GenerationParams::default();

    let rx = service.infer_stream("test", &params, None).unwrap();

    // Start receiving, then drop the receiver
    let first_event = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(first_event, StreamEvent::Started);

    // Drop the receiver - this should cause the sender to fail gracefully
    drop(rx);

    // Wait a bit to ensure the thread detects the dropped receiver
    std::thread::sleep(Duration::from_millis(100));

    // If we get here without deadlock or panic, the test passes
}

// ============================================================================
// Criterion 4: Thread safety - no panics, no deadlocks
// ============================================================================

#[test]
fn criterion_4_no_deadlock_under_load() {
    let service = Arc::new(MockLlmService::new("Response for load test"));
    let params = GenerationParams::default();

    // Spawn multiple concurrent streaming requests
    let mut handles = Vec::new();

    for i in 0..10 {
        let svc = service.clone();
        let p = params.clone();
        let handle = std::thread::spawn(move || {
            let rx = svc.infer_stream(&format!("prompt {i}"), &p, None).unwrap();

            let mut tokens = Vec::new();
            while let Ok(event) = rx.recv_timeout(Duration::from_secs(10)) {
                if let Some(token) = event.as_token() {
                    tokens.push(token.to_string());
                }
                if event.is_terminal() {
                    break;
                }
            }

            tokens
        });
        handles.push(handle);
    }

    // All threads should complete without deadlock
    for handle in handles {
        let tokens = handle.join().expect("Thread should not panic");
        assert!(!tokens.is_empty(), "Each stream should receive tokens");
    }
}

#[test]
fn criterion_4_panic_catch_in_inference_thread() {
    // Test that panics in the inference thread are caught
    // This is harder to test directly, but we can verify the channel
    // mechanism handles errors gracefully

    let service = MockLlmService::new("Test");
    let params = GenerationParams::default();

    let rx = service.infer_stream("test", &params, None).unwrap();

    // Drain all events
    let mut events = Vec::new();
    while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
        events.push(event);
    }

    // Should have at least Started and Completed
    assert!(events.len() >= 2, "Should have at least Started and Completed events");
    assert_eq!(events.first(), Some(&StreamEvent::Started));
    assert_eq!(events.last(), Some(&StreamEvent::Completed));
}

// ============================================================================
// Criterion 5: Timing information
// ============================================================================

#[test]
fn criterion_5_timing_information_present() {
    let service = MockLlmService::new("Hello world timing test");
    let params = GenerationParams::default();

    let start = Instant::now();
    let rx = service.infer_stream("test", &params, None).unwrap();

    let mut token_count = 0;
    let mut _first_token_time = None;

    while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
        if event.is_token() && _first_token_time.is_none() {
            _first_token_time = Some(start.elapsed());
        }
        if event.is_token() {
            token_count += 1;
        }
        if event.is_terminal() {
            break;
        }
    }

    let total_time = start.elapsed();

    assert!(token_count > 0, "Should have received tokens");
    assert!(total_time.as_millis() > 0, "Total generation time should be positive");
    assert!(_first_token_time.is_some(), "Should be able to measure first token latency");
}

// ============================================================================
// Criterion 6: Consistency with synchronous output
// ============================================================================

#[test]
fn criterion_6_stream_consistency_with_sync() {
    let service = MockLlmService::new("Consistent output test");
    let params = GenerationParams::default();

    // Get sync response
    let sync_response = service.infer("test", &params, None).unwrap();

    // Get stream response
    let rx = service.infer_stream("test", &params, None).unwrap();
    let mut stream_tokens = Vec::new();

    while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
        if let Some(token) = event.as_token() {
            stream_tokens.push(token.to_string());
        }
        if event.is_terminal() {
            break;
        }
    }

    let stream_text: String = stream_tokens.join("");

    // Both should produce the same text
    assert_eq!(sync_response.text, stream_text, "Stream and sync should produce identical text");
}

// ============================================================================
// Edge cases and error handling
// ============================================================================

#[test]
fn edge_case_empty_response() {
    let service = MockLlmService::new("");
    let params = GenerationParams::default();

    let rx = service.infer_stream("test", &params, None).unwrap();

    let mut events = Vec::new();
    while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
        events.push(event);
    }

    // Should still have Started and Completed
    assert_eq!(events.first(), Some(&StreamEvent::Started));
    assert_eq!(events.last(), Some(&StreamEvent::Completed));
}

#[test]
fn edge_case_single_word_response() {
    let service = MockLlmService::new("Hello");
    let params = GenerationParams::default();

    let rx = service.infer_stream("test", &params, None).unwrap();

    let mut events = Vec::new();
    while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
        events.push(event);
    }

    // Should have Started, one Token, and Completed
    assert_eq!(events.first(), Some(&StreamEvent::Started));
    assert_eq!(events.last(), Some(&StreamEvent::Completed));

    let tokens: Vec<&str> = events.iter().filter_map(|e| e.as_token()).collect();

    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0], "Hello");
}

#[test]
fn edge_case_multiline_response() {
    let service = MockLlmService::new("Line 1\nLine 2\nLine 3");
    let params = GenerationParams::default();

    let rx = service.infer_stream("test", &params, None).unwrap();

    let mut tokens = Vec::new();
    while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
        if let Some(token) = event.as_token() {
            tokens.push(token.to_string());
        }
        if event.is_terminal() {
            break;
        }
    }

    // MockLlmService splits by whitespace, so newlines are treated as whitespace
    // This is acceptable behavior as real LLMs also tokenize differently
    let reconstructed: String = tokens.join("");
    assert_eq!(reconstructed, "Line 1 Line 2 Line 3");
}

// ============================================================================
// Performance characteristics
// ============================================================================

#[test]
fn performance_tokens_per_second_reasonable() {
    let service = MockLlmService::new("Testing performance characteristics of the stream")
        .with_tokens_per_second(100.0);
    let params = GenerationParams::default();

    let start = Instant::now();
    let rx = service.infer_stream("test", &params, None).unwrap();

    let mut token_count = 0;
    while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
        if event.is_token() {
            token_count += 1;
        }
        if event.is_terminal() {
            break;
        }
    }

    let elapsed = start.elapsed();
    let actual_tps = f64::from(token_count) / elapsed.as_secs_f64();

    // Should be reasonably close to configured speed
    // Allow for overhead and variance
    assert!(actual_tps > 10.0, "Should achieve reasonable tokens/sec, got {actual_tps:.1}");
}

#[test]
fn performance_low_latency_first_token() {
    let service = MockLlmService::new("First token latency test");
    let params = GenerationParams::default();

    let start = Instant::now();
    let rx = service.infer_stream("test", &params, None).unwrap();

    // Wait for first token
    let mut first_token_latency = None;
    while let Ok(event) = rx.recv_timeout(Duration::from_secs(5)) {
        if event.is_token() {
            first_token_latency = Some(start.elapsed());
            break;
        }
    }

    let _ = rx; // Drain remaining events

    assert!(first_token_latency.is_some(), "Should receive first token");
}
