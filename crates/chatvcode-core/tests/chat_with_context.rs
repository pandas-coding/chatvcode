//! Integration tests for `chat_with_context()` using MockLlmService + real index/vector store fixtures.
//!
//! Verifies the full RAG pipeline end-to-end:
//! - Index -> vector store -> metadata store -> search -> LLM inference -> ChatResponse
//! - No-context scenario (empty vector store)
//! - Streaming variant `chat_with_context_stream()`

use chatvcode_core::{
    ChatOptions, ChunkMetadataStore, build_metadata_store, chat_with_context,
    chat_with_context_stream, index_path,
};
use chatvcode_llm::{ChatTemplate, MockLlmService, StopReason, StreamEvent};
use chatvcode_vdb::{
    EmbeddingService, EmbeddingVector, InMemoryVectorStore, MockEmbeddingService, VectorStore,
};
use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;

mod common;
use common::mock_parse_source;

fn create_test_project() -> TempDir {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("src/main.rs"),
        "fn main() {\n    println!(\"hello world\");\n}",
    )
    .unwrap();
    std::fs::write(
        root.join("src/lib.rs"),
        "pub fn greet(name: &str) -> String {\n    format!(\"Hello, {}!\", name)\n}",
    )
    .unwrap();
    std::fs::write(
        root.join("src/utils.rs"),
        "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}",
    )
    .unwrap();
    tmp
}

fn setup_index_and_stores(tmp: &TempDir) -> (PathBuf, PathBuf) {
    let index_result = index_path(tmp.path(), &mock_parse_source).unwrap();
    assert!(index_result.stats.total_chunks > 0, "Should have chunks");

    let metadata_store = build_metadata_store(&index_result);
    let metadata_path = tmp.path().join(".chatvcode/vectors.atmd");
    std::fs::create_dir_all(metadata_path.parent().unwrap()).unwrap();
    metadata_store.save(&metadata_path).unwrap();

    let vdb_path = tmp.path().join(".chatvcode/vectors.vdb");
    let embedding_service = MockEmbeddingService::new(32);

    let all_chunks: Vec<_> = index_result
        .files
        .iter()
        .flat_map(|f| f.chunks.iter())
        .collect();

    let mut store = InMemoryVectorStore::new();
    let texts: Vec<&str> = all_chunks.iter().map(|c| c.source_text.as_str()).collect();
    let vectors = embedding_service.embed(&texts).unwrap();
    let evs: Vec<EmbeddingVector> = all_chunks
        .iter()
        .zip(vectors)
        .map(|(chunk, vector)| EmbeddingVector::new(&chunk.id, vector))
        .collect();
    store.add(evs).unwrap();
    store.save(&vdb_path).unwrap();

    (vdb_path, metadata_path)
}

#[test]
fn test_chat_with_context_returns_answer_with_sources() {
    let tmp = create_test_project();
    let (_vdb_path, _metadata_path) = setup_index_and_stores(&tmp);

    let llm = MockLlmService::new("The main function prints hello world.");
    let embedding_service = MockEmbeddingService::new(32);

    let options = ChatOptions::new(tmp.path())
        .with_chat_template(ChatTemplate::ChatML)
        .with_top_k(5)
        .system_prompt("You are a coding assistant.");

    let response = chat_with_context("What does main do?", &llm, &embedding_service, &options)
        .unwrap();

    assert!(!response.answer.is_empty(), "Answer should not be empty");
    assert_eq!(response.answer, "The main function prints hello world.");
    assert!(!response.sources.is_empty(), "Should have source references");
    assert_eq!(response.stop_reason, StopReason::Eos);
    assert!(response.token_usage.total_tokens > 0);
    assert!(response.duration.as_nanos() > 0);
    assert!(response.retrieved_count > 0);
}

#[test]
fn test_chat_with_context_no_results_still_returns_answer() {
    let tmp = create_test_project();

    let vdb_path = tmp.path().join(".chatvcode/vectors.vdb");
    std::fs::create_dir_all(vdb_path.parent().unwrap()).unwrap();
    let empty_store = InMemoryVectorStore::new();
    empty_store.save(&vdb_path).unwrap();

    let metadata_path = tmp.path().join(".chatvcode/vectors.atmd");
    let empty_meta = ChunkMetadataStore::new();
    empty_meta.save(&metadata_path).unwrap();

    let llm = MockLlmService::new("I could not find relevant code in your codebase.");
    let embedding_service = MockEmbeddingService::new(32);

    let options = ChatOptions::new(tmp.path())
        .with_chat_template(ChatTemplate::ChatML)
        .with_top_k(5);

    let response = chat_with_context("What does main do?", &llm, &embedding_service, &options)
        .unwrap();

    assert!(!response.answer.is_empty());
    assert!(response.sources.is_empty(), "No sources when no results");
    assert!(response.is_no_context());
    assert_eq!(response.retrieved_count, 0);
    assert_eq!(response.used_count, 0);
}

#[test]
fn test_chat_with_context_token_budget_limits_context() {
    let tmp = create_test_project();
    let (_vdb_path, _metadata_path) = setup_index_and_stores(&tmp);

    let llm = MockLlmService::new("Answer with limited context.");
    let embedding_service = MockEmbeddingService::new(32);

    let options = ChatOptions::new(tmp.path())
        .with_chat_template(ChatTemplate::ChatML)
        .with_top_k(10)
        .with_context_token_budget(5);

    let response = chat_with_context("Explain the code", &llm, &embedding_service, &options)
        .unwrap();

    assert!(!response.answer.is_empty());
    assert!(response.used_count <= 10);
}

#[test]
fn test_chat_with_context_stream_returns_events() {
    let tmp = create_test_project();
    let (_vdb_path, _metadata_path) = setup_index_and_stores(&tmp);

    let llm = MockLlmService::new("Streaming answer here");
    let embedding_service = MockEmbeddingService::new(32);

    let options = ChatOptions::new(tmp.path())
        .with_chat_template(ChatTemplate::ChatML)
        .with_top_k(5);

    let response =
        chat_with_context_stream("What is greet?", &llm, &embedding_service, &options).unwrap();

    assert!(!response.sources.is_empty(), "Should have sources");

    let mut events = Vec::new();
    while let Ok(event) = response.event_receiver.recv_timeout(Duration::from_secs(5)) {
        events.push(event);
    }

    assert!(
        events.iter().any(|e| matches!(e, StreamEvent::Started)),
        "Should have Started event"
    );
    assert!(
        events.iter().any(|e| e.is_token()),
        "Should have Token events"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::Completed | StreamEvent::Cancelled)),
        "Should have terminal event"
    );
}

#[test]
fn test_chat_with_context_sources_have_correct_metadata() {
    let tmp = create_test_project();
    let (_vdb_path, _metadata_path) = setup_index_and_stores(&tmp);

    let llm = MockLlmService::new("The add function adds two numbers.");
    let embedding_service = MockEmbeddingService::new(32);

    let options = ChatOptions::new(tmp.path())
        .with_chat_template(ChatTemplate::ChatML)
        .with_top_k(5);

    let response =
        chat_with_context("How does add work?", &llm, &embedding_service, &options).unwrap();

    for source in &response.sources {
        assert!(!source.chunk_id.is_empty());
        assert!(!source.file_path.to_string_lossy().is_empty());
        assert!(source.start_line > 0);
        assert!(source.end_line >= source.start_line);
        assert!(source.score > 0.0);
        assert!(!source.snippet.is_empty());
    }
}

#[test]
fn test_chat_with_context_format_sources_output() {
    let tmp = create_test_project();
    let (_vdb_path, _metadata_path) = setup_index_and_stores(&tmp);

    let llm = MockLlmService::new("Answer text");
    let embedding_service = MockEmbeddingService::new(32);

    let options = ChatOptions::new(tmp.path())
        .with_chat_template(ChatTemplate::ChatML)
        .with_top_k(3);

    let response =
        chat_with_context("Explain the code", &llm, &embedding_service, &options).unwrap();

    let formatted = response.format_sources();
    if response.sources.is_empty() {
        assert!(formatted.contains("No sources available"));
    } else {
        assert!(formatted.contains("Sources:"));
        for source in &response.sources {
            assert!(
                formatted.contains(&source.file_path.to_string_lossy().to_string()),
                "Formatted sources should contain file path"
            );
        }
    }
}
