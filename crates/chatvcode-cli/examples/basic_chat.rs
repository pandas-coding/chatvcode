//! Example: Basic chat with mock LLM service
//!
//! This example demonstrates the `chatvcode chat` command flow
//! using a mock LLM service, without requiring a real GGUF model file.
//!
//! Run with:
//!   cargo run --example basic_chat

use chatvcode_cli::chatvcode_core::{
    ChatOptions, ChatResponse, ChunkKind, SourceReference, build_rag_prompt,
};
use chatvcode_cli::chatvcode_llm::{
    ChatPromptBuilder, ChatTemplate, GenerationParams, LlmService, MockLlmService, StopReason,
    StreamEvent, TokenUsage,
};
use std::path::PathBuf;
use std::time::Duration;

fn main() {
    println!("=== Basic Chat Example ===\n");

    // 1. Demonstrate prompt building with RAG context
    demo_prompt_building();

    // 2. Demonstrate mock LLM inference (sync)
    demo_mock_inference();

    // 3. Demonstrate mock LLM inference (streaming)
    demo_mock_streaming();

    // 4. Demonstrate chat response formatting
    demo_response_formatting();

    // 5. Demonstrate template variations
    demo_template_variations();

    println!("\n=== All demos completed ===");
}

fn demo_prompt_building() {
    println!("--- Prompt Building ---");

    let options = ChatOptions::new("/tmp/project").with_chat_template(ChatTemplate::ChatML);

    // With context
    let snippets = vec![
        "--- src/main.rs:10-20 (function: main) [score: 0.950] ---\nfn main() {\n    println!(\"hello\");\n}\n---"
            .to_string(),
    ];

    let prompt = build_rag_prompt("What does the main function do?", &snippets, &options).unwrap();
    println!("Prompt with context:\n{}\n", prompt);
    assert!(prompt.contains("What does the main function do?"));
    assert!(prompt.contains("main"));
    assert!(prompt.contains("<|im_start|>"));

    // Without context
    let prompt_no_ctx = build_rag_prompt("What is Rust?", &[], &options).unwrap();
    println!("Prompt without context:\n{}\n", prompt_no_ctx);
    assert!(prompt.contains("What does the main function do?"));
}

fn demo_mock_inference() {
    println!("--- Mock Inference (Sync) ---");

    let service = MockLlmService::new("The main function prints 'hello' to stdout.");
    let params = GenerationParams::default();
    let cancel = std::sync::atomic::AtomicBool::new(false);

    let response = service
        .infer("What does the main function do?", &params, Some(&cancel))
        .unwrap();

    println!("Response: {}", response.text);
    println!("Stop reason: {:?}", response.stop_reason);
    println!(
        "Tokens: {} prompt + {} completion = {} total",
        response.token_usage.prompt_tokens,
        response.token_usage.completion_tokens,
        response.token_usage.total_tokens,
    );
    println!("Duration: {:.1}ms\n", response.duration.as_millis());

    assert_eq!(response.text, "The main function prints 'hello' to stdout.");
    assert_eq!(response.stop_reason, StopReason::Eos);
}

fn demo_mock_streaming() {
    println!("--- Mock Inference (Streaming) ---");

    let service = MockLlmService::new("Rust is a systems programming language focused on safety.");
    let params = GenerationParams::default();
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    let rx = service
        .infer_stream("What is Rust?", &params, Some(cancel))
        .unwrap();

    let mut full_text = String::new();
    let mut event_count = 0;

    loop {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(event) => {
                event_count += 1;
                match &event {
                    StreamEvent::Started => print!("Stream: "),
                    StreamEvent::Token(token) => {
                        print!("{token}");
                        full_text.push_str(token);
                    }
                    StreamEvent::Completed => println!("\n[completed]"),
                    StreamEvent::Cancelled => println!("\n[cancelled]"),
                    StreamEvent::Error(msg) => println!("\n[error: {msg}]"),
                }
                if event.is_terminal() {
                    break;
                }
            }
            Err(_) => break,
        }
    }

    println!("Received {event_count} events");
    println!("Full text: {full_text}\n");

    assert!(!full_text.is_empty());
    assert!(full_text.contains("Rust"));
}

fn demo_response_formatting() {
    println!("--- Response Formatting ---");

    let sources = vec![
        SourceReference {
            chunk_id: "src/main.rs:Function:main:10".to_string(),
            file_path: PathBuf::from("src/main.rs"),
            kind: ChunkKind::Function,
            symbol_name: Some("main".to_string()),
            start_line: 10,
            end_line: 20,
            score: 0.95,
            snippet: "fn main() {\n    println!(\"hello\");\n}".to_string(),
        },
        SourceReference {
            chunk_id: "src/lib.rs:Struct:Config:5".to_string(),
            file_path: PathBuf::from("src/lib.rs"),
            kind: ChunkKind::Struct,
            symbol_name: Some("Config".to_string()),
            start_line: 5,
            end_line: 15,
            score: 0.82,
            snippet: "struct Config { debug: bool }".to_string(),
        },
    ];

    let response = ChatResponse {
        answer: "The main function prints 'hello' to stdout, and Config holds configuration."
            .to_string(),
        sources,
        token_usage: TokenUsage::new(50, 20),
        stop_reason: StopReason::Eos,
        duration: Duration::from_millis(200),
        search_duration: Duration::from_millis(20),
        inference_duration: Duration::from_millis(180),
        retrieved_count: 2,
        used_count: 2,
    };

    println!("Answer: {}", response.answer);
    println!("{}", response.format_sources());
    println!("Time: {:.1}s", response.duration.as_secs_f64(),);

    assert!(!response.is_no_context());
    assert_eq!(response.sources.len(), 2);

    // Empty sources case
    let empty_response = ChatResponse {
        answer: "I don't know.".to_string(),
        sources: vec![],
        token_usage: TokenUsage::new(10, 5),
        stop_reason: StopReason::Eos,
        duration: Duration::from_millis(100),
        search_duration: Duration::from_millis(10),
        inference_duration: Duration::from_millis(90),
        retrieved_count: 0,
        used_count: 0,
    };
    assert!(empty_response.is_no_context());
    let empty_sources = empty_response.format_sources();
    println!("\nNo context sources: {empty_sources}");
    assert!(empty_sources.contains("No sources available"));
}

fn demo_template_variations() {
    println!("--- Template Variations ---");

    let templates = vec![
        ("Auto", ChatTemplate::Auto),
        ("Raw", ChatTemplate::Raw),
        ("ChatML", ChatTemplate::ChatML),
        ("Llama3", ChatTemplate::Llama3),
    ];

    for (name, template) in templates {
        let builder = ChatPromptBuilder::new(template)
            .system_prompt("You are a coding assistant.")
            .user_question("What does the main function do?");

        match builder.build() {
            Ok(prompt) => {
                println!("{name} template:\n{prompt}\n");
            }
            Err(e) => {
                println!("{name} template error: {e}\n");
            }
        }
    }
}
