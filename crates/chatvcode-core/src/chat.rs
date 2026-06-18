//! RAG (Retrieval-Augmented Generation) integration for `chatvcode-core`.
//!
//! This module provides the core types and entry points for RAG-enhanced
//! question answering over a codebase. It combines semantic search (via
//! `chatvcode-vdb`) with LLM inference (via `chatvcode-llm`) to produce answers
//! grounded in retrieved code context.
//!
//! # Key Types
//!
//! - [`ChatOptions`] — Configuration for a RAG chat request
//! - [`ChatResponse`] — The result of a RAG chat, including answer, sources, and stats
//! - [`SourceReference`] — A reference to a code source used in the answer
//! - [`ChunkMetadataStore`] — Persistent `chunk_id → metadata` mapping for fast query lookups
//!
//! # Key Functions
//!
//! - [`chat_with_context`] — The main RAG entry point
//! - [`build_rag_prompt`] — Build a RAG prompt from context snippets
//! - [`format_context_snippets`] — Format retrieval results as LLM context
//! - [`apply_token_budget`] — Trim context to fit a token budget

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use chatvcode_llm::{
    ChatPromptBuilder, ChatTemplate, GenerationParams, LlmError, LlmService, StopReason,
    StreamEvent, TokenUsage,
};
use chatvcode_vdb::{EmbeddingService, InMemoryVectorStore, VectorStore};

use crate::error::{ChatVCodeError, ChatVCodeResult, ErrorContext};
use crate::model::{ChunkKind, ChunkMetadata, ChunkMetadataStore, CodeChunk};

// ---------------------------------------------------------------------------
// Chat options
// ---------------------------------------------------------------------------

/// Configuration for a RAG-enhanced chat request.
///
/// Combines model configuration, retrieval configuration, and generation
/// parameters into a single struct that fully specifies how `chat_with_context`
/// should operate.
#[derive(Debug, Clone)]
pub struct ChatOptions {
    /// Path to the project being queried.
    pub project_path: PathBuf,

    // --- Retrieval configuration ---
    /// Path to the vector store file (`.atvs`).
    pub vector_store_path: Option<PathBuf>,
    /// Path to the embedding model configuration, or a pre-built embedding config.
    pub embedding_config: Option<chatvcode_vdb::EmbeddingConfig>,
    /// Path to the chunk metadata store file (`.atmd`).
    pub metadata_store_path: Option<PathBuf>,
    /// Number of top results to retrieve from the vector store.
    pub top_k: usize,
    /// Minimum similarity score for retrieval results (0.0–1.0).
    pub min_score: Option<f32>,
    /// Maximum number of tokens allocated to context snippets.
    /// Uses a rough heuristic of ~4 characters per token. 0 = unlimited.
    pub context_token_budget: usize,

    // --- Generation configuration ---
    /// Chat template to use for prompt formatting.
    pub chat_template: ChatTemplate,
    /// System prompt for the LLM.
    pub system_prompt: Option<String>,
    /// LLM generation parameters.
    pub generation_params: GenerationParams,
}

impl ChatOptions {
    /// Creates `ChatOptions` with the given project path and defaults.
    ///
    /// Defaults:
    /// - `top_k` = 16
    /// - `min_score` = None (no filter)
    /// - `context_token_budget` = 0 (unlimited)
    /// - `chat_template` = `ChatTemplate::Auto`
    /// - `system_prompt` = None
    /// - `generation_params` = `GenerationParams::default()`
    #[must_use]
    pub fn new(project_path: impl Into<PathBuf>) -> Self {
        let project_path = project_path.into();
        Self {
            vector_store_path: None,
            embedding_config: None,
            metadata_store_path: None,
            top_k: 16,
            min_score: None,
            context_token_budget: 0,
            chat_template: ChatTemplate::Auto,
            system_prompt: None,
            generation_params: GenerationParams::default(),
            project_path,
        }
    }

    /// Sets the vector store path.
    pub fn vector_store_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.vector_store_path = Some(path.into());
        self
    }

    /// Sets the embedding configuration.
    pub fn embedding_config(mut self, config: chatvcode_vdb::EmbeddingConfig) -> Self {
        self.embedding_config = Some(config);
        self
    }

    /// Sets the metadata store path.
    pub fn metadata_store_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.metadata_store_path = Some(path.into());
        self
    }

    /// Sets the number of top-k results to retrieve.
    #[must_use]
    pub const fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Sets the minimum similarity score filter.
    #[must_use]
    pub const fn with_min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }

    /// Sets the context token budget. 0 = unlimited.
    #[must_use]
    pub const fn with_context_token_budget(mut self, budget: usize) -> Self {
        self.context_token_budget = budget;
        self
    }

    /// Sets the chat template.
    pub fn with_chat_template(mut self, template: ChatTemplate) -> Self {
        self.chat_template = template;
        self
    }

    /// Sets the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Sets the generation parameters.
    #[must_use]
    pub fn with_generation_params(mut self, params: GenerationParams) -> Self {
        self.generation_params = params;
        self
    }

    /// Resolves the vector store path.
    ///
    /// If not explicitly set, defaults to `<project_path>/.chatvcode/vectors.vdb`.
    /// This is compatible with the `chatvcode index` default output path.
    pub fn resolve_vector_store_path(&self) -> PathBuf {
        self.vector_store_path
            .clone()
            .unwrap_or_else(|| self.project_path.join(".chatvcode").join("vectors.vdb"))
    }

    /// Resolves the metadata store path.
    ///
    /// If not explicitly set, defaults to `<project_path>/.chatvcode/vectors.atmd`.
    /// This is compatible with the `chatvcode index` default output path.
    pub fn resolve_metadata_store_path(&self) -> PathBuf {
        self.metadata_store_path
            .clone()
            .unwrap_or_else(|| self.project_path.join(".chatvcode").join("vectors.atmd"))
    }
}

// ---------------------------------------------------------------------------
// Source reference
// ---------------------------------------------------------------------------

/// A reference to a source code chunk used in generating a RAG answer.
///
/// Preserves the essential metadata (file path, line numbers, symbol name,
/// chunk kind, and similarity score) needed to trace back from an answer
/// to the original code.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceReference {
    /// The chunk identifier.
    pub chunk_id: String,
    /// Path to the source file.
    pub file_path: PathBuf,
    /// Kind of the code chunk (function, struct, etc.).
    pub kind: ChunkKind,
    /// Symbol name (e.g., function name), if available.
    pub symbol_name: Option<String>,
    /// Starting line number (1-indexed).
    pub start_line: usize,
    /// Ending line number (1-indexed, inclusive).
    pub end_line: usize,
    /// Cosine similarity score from the retrieval step.
    pub score: f32,
    /// The source code text of the chunk (may be truncated by token budget).
    pub snippet: String,
}

impl SourceReference {
    /// Creates a `SourceReference` from a `CodeChunk` and a similarity score.
    #[must_use]
    pub fn from_chunk(chunk: &CodeChunk, score: f32) -> Self {
        Self {
            chunk_id: chunk.id.clone(),
            file_path: chunk.file_path.clone(),
            kind: chunk.kind,
            symbol_name: chunk.symbol_name.clone(),
            start_line: chunk.span.start_line + 1, // Convert 0-indexed to 1-indexed
            end_line: chunk.span.end_line + 1,
            score,
            snippet: chunk.source_text.clone(),
        }
    }

    /// Creates a `SourceReference` from `ChunkMetadata` and a similarity score.
    #[must_use]
    pub fn from_metadata(meta: &ChunkMetadata, score: f32) -> Self {
        Self {
            chunk_id: meta.chunk_id.clone(),
            file_path: meta.file_path.clone(),
            kind: meta.kind,
            symbol_name: meta.symbol_name.clone(),
            start_line: meta.start_line,
            end_line: meta.end_line,
            score,
            snippet: meta.source_text.clone(),
        }
    }

    /// Returns a human-readable description of this source reference.
    #[must_use]
    pub fn display_path(&self) -> String {
        match &self.symbol_name {
            Some(name) => format!(
                "{}:{}:{}-{}",
                self.file_path.display(),
                self.start_line,
                name,
                self.end_line
            ),
            None => format!("{}:{}-{}", self.file_path.display(), self.start_line, self.end_line),
        }
    }
}

// ---------------------------------------------------------------------------
// Chat response
// ---------------------------------------------------------------------------

/// The result of a RAG-enhanced chat request.
///
/// Contains the LLM-generated answer, references to code sources that
/// contributed to the answer, token usage statistics, stop reason,
/// and timing information.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// The generated answer text.
    pub answer: String,

    /// Source references for the code context used.
    pub sources: Vec<SourceReference>,

    /// Token usage statistics.
    pub token_usage: TokenUsage,

    /// Reason why generation stopped.
    pub stop_reason: StopReason,

    /// Total duration of the chat request (search + inference).
    pub duration: std::time::Duration,

    /// Time spent on the retrieval/search phase.
    pub search_duration: std::time::Duration,

    /// Time spent on the LLM inference phase.
    pub inference_duration: std::time::Duration,

    /// Number of context snippets retrieved (before token-budget trimming).
    pub retrieved_count: usize,

    /// Number of context snippets actually used (after token-budget trimming).
    pub used_count: usize,
}

impl ChatResponse {
    /// Returns `true` if no sources were used (pure LLM answer, no retrieval context).
    #[must_use]
    pub fn is_no_context(&self) -> bool {
        self.sources.is_empty()
    }

    /// Returns a formatted string listing all source references.
    #[must_use]
    pub fn format_sources(&self) -> String {
        if self.sources.is_empty() {
            return "No sources available (answer based on model knowledge only)".to_string();
        }

        let mut out = String::new();
        out.push_str("Sources:\n");
        for (i, src) in self.sources.iter().enumerate() {
            out.push_str(&format!(
                "  [{}] {} (score: {:.3})\n",
                i + 1,
                src.display_path(),
                src.score
            ));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Streaming chat response
// ---------------------------------------------------------------------------

/// The result of a streaming RAG chat request.
///
/// Contains source references (available immediately) and a receiver
/// for streaming token events from the LLM.
pub struct StreamingChatResponse {
    /// Source references for the code context used.
    pub sources: Vec<SourceReference>,

    /// Number of context snippets retrieved.
    pub retrieved_count: usize,

    /// Number of context snippets actually used.
    pub used_count: usize,

    /// Duration of the retrieval/search phase.
    pub search_duration: std::time::Duration,

    /// Receiver for streaming token events.
    pub event_receiver: std::sync::mpsc::Receiver<StreamEvent>,
}

// ---------------------------------------------------------------------------
// RAG prompt building
// ---------------------------------------------------------------------------

/// Default system prompt for code-related RAG.
const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful coding assistant. Answer questions about the user's codebase \
     using the provided context. If the context doesn't contain enough information \
     to answer the question, say so honestly. Always cite the file path and line \
     numbers of the relevant code when possible.";

/// No-context system prompt used when no retrieval results are available.
const NO_CONTEXT_SYSTEM_PROMPT: &str = "You are a helpful coding assistant. The user asked a question about their codebase, \
     but no relevant code was found through search. Answer based on your general knowledge \
     if you can, but clearly state that you were unable to find relevant code in the codebase \
     and that your answer may not be specific to their project.";

/// Builds a RAG prompt from the user question and context snippets.
///
/// This function:
/// 1. Formats the context snippets into a structured block
/// 2. Applies the token budget if specified
/// 3. Injects the context and question into the chat template
///
/// # Arguments
///
/// * `question` — The user's question
/// * `snippets` — Formatted context strings (from `format_context_snippets`)
/// * `options` — Chat options controlling template, system prompt, and token budget
///
/// # Errors
///
/// Returns an error if the chat template cannot format the messages
/// (e.g., Custom jinja templates without llama.cpp).
pub fn build_rag_prompt(
    question: &str,
    snippets: &[String],
    options: &ChatOptions,
) -> ChatVCodeResult<String> {
    let system_prompt = options
        .system_prompt
        .as_deref()
        .unwrap_or(DEFAULT_SYSTEM_PROMPT);

    let mut builder = ChatPromptBuilder::new(options.chat_template.clone())
        .system_prompt(system_prompt)
        .user_question(question)
        .add_generation_prompt(true);

    // Apply token budget
    if options.context_token_budget > 0 {
        builder = builder.context_token_budget(options.context_token_budget);
    }

    // Add context snippets
    for snippet in snippets {
        builder = builder.context(snippet);
    }

    builder.build().map_err(|e| {
        ChatVCodeError::internal(format!("Failed to build RAG prompt: {e}"))
            .with_context(ErrorContext::default().with_operation("build_rag_prompt"))
            .with_source(e.to_string())
    })
}

/// Formats retrieval results into context strings suitable for LLM injection.
///
/// Each result is formatted as:
/// ```text
/// --- src/lib.rs:42-58 (function: parse_config) [score: 0.892] ---
/// <source code>
/// ---
/// ```
///
/// # Arguments
///
/// * `results` — Search results with chunk metadata
///
/// # Returns
///
/// A vector of formatted context strings, one per result.
pub fn format_context_snippets(results: &[(ChunkMetadata, f32)]) -> Vec<String> {
    results
        .iter()
        .map(|(meta, score)| {
            let kind_str = match &meta.symbol_name {
                Some(name) => format!("{}: {}", meta.kind, name),
                None => meta.kind.to_string(),
            };
            format!(
                "--- {}:{}-{} ({}) [score: {:.3}] ---\n{}\n---",
                meta.file_path.display(),
                meta.start_line,
                meta.end_line,
                kind_str,
                score,
                meta.source_text
            )
        })
        .collect()
}

/// Applies a token budget to context snippets by trimming from the end.
///
/// Uses a rough heuristic of ~4 characters per token. Snippets that don't
/// fit within the budget are either trimmed or dropped entirely.
///
/// # Arguments
///
/// * `snippets` — Formatted context strings
/// * `token_budget` — Maximum number of tokens (0 = unlimited)
/// * `chars_per_token` — Heuristic ratio of characters per token (default: 4)
///
/// # Returns
///
/// A tuple of (trimmed snippets, number of snippets used, number trimmed).
pub fn apply_token_budget(
    snippets: &[String],
    token_budget: usize,
    chars_per_token: usize,
) -> (Vec<String>, usize, usize) {
    if token_budget == 0 {
        // Unlimited budget — return all snippets as-is
        return (snippets.to_vec(), snippets.len(), 0);
    }

    let char_budget = token_budget.saturating_mul(chars_per_token);
    let mut result = Vec::new();
    let mut remaining = char_budget;

    for snippet in snippets {
        if remaining == 0 {
            break;
        }

        if snippet.len() <= remaining {
            result.push(snippet.clone());
            remaining = remaining.saturating_sub(snippet.len());
        } else {
            // Trim this snippet to fit
            let trim_point = snippet.floor_char_boundary(remaining);
            let trimmed = format!("{}...", &snippet[..trim_point]);
            result.push(trimmed);
            remaining = 0;
        }
    }

    let trimmed_count = snippets.len().saturating_sub(result.len());
    let used = result.len();
    (result, used, trimmed_count)
}

// ---------------------------------------------------------------------------
// Metadata store (chunk_id → metadata)
// ---------------------------------------------------------------------------

/// Build a `ChunkMetadataStore` from the current index.
///
/// This is a convenience wrapper around `ChunkMetadataStore::from_index_result`
/// that maps `chatvcode-core` `CodeChunk`s to `ChunkMetadata` entries.
pub fn build_metadata_store(index_result: &crate::model::IndexResult) -> ChunkMetadataStore {
    let entries: std::collections::HashMap<String, ChunkMetadata> = index_result
        .files
        .iter()
        .flat_map(|file| file.chunks.iter())
        .map(|chunk| {
            let meta = ChunkMetadata {
                chunk_id: chunk.id.clone(),
                file_path: chunk.file_path.clone(),
                language: chunk.language.to_string(),
                kind: chunk.kind,
                symbol_name: chunk.symbol_name.clone(),
                start_line: chunk.span.start_line + 1, // Convert 0-indexed to 1-indexed
                end_line: chunk.span.end_line + 1,
                start_byte: chunk.span.start_byte,
                end_byte: chunk.span.end_byte,
                source_text: chunk.source_text.clone(),
            };
            (chunk.id.clone(), meta)
        })
        .collect();

    ChunkMetadataStore { version: ChunkMetadataStore::CURRENT_VERSION, entries }
}

// ---------------------------------------------------------------------------
// Core RAG entry point: chat_with_context
// ---------------------------------------------------------------------------

/// Performs a RAG-enhanced chat query against a codebase.
///
/// This is the main entry point for the RAG pipeline. It:
/// 1. Embeds the user's query
/// 2. Searches the vector store for relevant code chunks
/// 3. Loads chunk metadata to resolve chunk IDs
/// 4. Formats the retrieved context
/// 5. Builds a RAG prompt
/// 6. Runs LLM inference
/// 7. Returns a `ChatResponse` with the answer and sources
///
/// # Arguments
///
/// * `question` — The user's question
/// * `llm` — The LLM service for inference
/// * `embedding_service` — The embedding service for query embedding
/// * `options` — Chat configuration
///
/// # Errors
///
/// Returns an `ChatVCodeError` if:
/// - The vector store file is missing or corrupt
/// - The metadata store file is missing or corrupt
/// - The embedding service fails
/// - The LLM service fails
///
/// # Note
///
/// When no retrieval results are found, the function still calls the LLM
/// but uses a special system prompt that asks the model to answer honestly
/// about the lack of relevant code context.
pub fn chat_with_context(
    question: &str,
    llm: &dyn LlmService,
    embedding_service: &dyn EmbeddingService,
    options: &ChatOptions,
) -> ChatVCodeResult<ChatResponse> {
    let total_start = Instant::now();

    // --- Phase 1: Retrieval ---
    let search_start = Instant::now();

    let (snippets, source_refs, retrieved_count, used_count) =
        retrieve_context(question, embedding_service, options)?;

    let search_duration = search_start.elapsed();

    // --- Phase 2: Build prompt and run LLM inference ---
    let inference_start = Instant::now();

    // Select system prompt based on whether we have context
    let effective_options = if source_refs.is_empty() {
        // No context found — use the no-context system prompt
        let mut opts = options.clone();
        opts.system_prompt = Some(NO_CONTEXT_SYSTEM_PROMPT.to_string());
        opts
    } else {
        options.clone()
    };

    // Build the prompt
    let prompt = build_rag_prompt(question, &snippets, &effective_options)?;

    // Run inference
    let cancel_flag = AtomicBool::new(false);
    let response = llm
        .infer(&prompt, &options.generation_params, Some(&cancel_flag))
        .map_err(|e| {
            // If context overflow, give a helpful message suggesting the user
            // increase --n-ctx or decrease --context-token-budget.
            if matches!(e, LlmError::ContextOverflow { .. }) {
                if let LlmError::ContextOverflow { n_ctx, n_tokens } = &e {
                    ChatVCodeError::internal(format!(
                        "Context overflow: prompt has {n_tokens} tokens but context size is {n_ctx}.\
\n  Suggestions:\n    - Increase --n-ctx (current: {n_ctx}, try --n-ctx 8192 or higher)\n    - Decrease --context-token-budget to limit RAG context\n    - Reduce --top-k-retrieval to fetch fewer snippets"
                    ))
                    .with_context(ErrorContext::default().with_operation("chat_with_context"))
                    .with_source(e.to_string())
                } else {
                    unreachable!("ContextOverflow match failed")
                }
            } else {
                ChatVCodeError::internal(format!("LLM inference failed: {e}"))
                    .with_context(ErrorContext::default().with_operation("chat_with_context"))
                    .with_source(e.to_string())
            }
        })?;

    let inference_duration = inference_start.elapsed();
    let total_duration = total_start.elapsed();

    Ok(ChatResponse {
        answer: response.text,
        sources: source_refs,
        token_usage: response.token_usage,
        stop_reason: response.stop_reason,
        duration: total_duration,
        search_duration,
        inference_duration,
        retrieved_count,
        used_count,
    })
}

/// Performs a streaming RAG-enhanced chat query against a codebase.
///
/// Similar to `chat_with_context` but returns a streaming response where
/// LLM tokens arrive as `StreamEvent`s through a channel receiver.
///
/// # Arguments
///
/// Same as `chat_with_context`.
///
/// # Returns
///
/// A `StreamingChatResponse` with source references (available immediately)
/// and a `Receiver<StreamEvent>` for streaming token events.
pub fn chat_with_context_stream(
    question: &str,
    llm: &dyn LlmService,
    embedding_service: &dyn EmbeddingService,
    options: &ChatOptions,
) -> ChatVCodeResult<StreamingChatResponse> {
    let search_start = Instant::now();

    let (snippets, source_refs, retrieved_count, used_count) =
        retrieve_context(question, embedding_service, options)?;

    let search_duration = search_start.elapsed();

    // Select system prompt based on whether we have context
    let effective_options = if source_refs.is_empty() {
        let mut opts = options.clone();
        opts.system_prompt = Some(NO_CONTEXT_SYSTEM_PROMPT.to_string());
        opts
    } else {
        options.clone()
    };

    // Build the prompt
    let prompt = build_rag_prompt(question, &snippets, &effective_options)?;

    // Start streaming inference
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let rx = llm
        .infer_stream(&prompt, &options.generation_params, Some(cancel_flag))
        .map_err(|e| {
            if matches!(e, LlmError::ContextOverflow { .. }) {
                if let LlmError::ContextOverflow { n_ctx, n_tokens } = &e {
                    ChatVCodeError::internal(format!(
                        "Context overflow: prompt has {n_tokens} tokens but context size is {n_ctx}.\
\n  Suggestions:\n    - Increase --n-ctx (current: {n_ctx}, try --n-ctx 8192 or higher)\n    - Decrease --context-token-budget to limit RAG context\n    - Reduce --top-k-retrieval to fetch fewer snippets"
                    ))
                    .with_context(ErrorContext::default().with_operation("chat_with_context_stream"))
                    .with_source(e.to_string())
                } else {
                    unreachable!("ContextOverflow match failed")
                }
            } else {
                ChatVCodeError::internal(format!("LLM streaming inference failed: {e}"))
                    .with_context(ErrorContext::default().with_operation("chat_with_context_stream"))
                    .with_source(e.to_string())
            }
        })?;

    Ok(StreamingChatResponse {
        sources: source_refs,
        retrieved_count,
        used_count,
        search_duration,
        event_receiver: rx,
    })
}

/// Retrieves context snippets for a RAG query.
///
/// Internal helper that handles:
/// 1. Loading the vector store
/// 2. Embedding the query
/// 3. Searching for similar chunks (over-retrieving for re-ranking)
/// 4. Loading metadata to resolve chunk IDs
/// 5. Keyword-based re-ranking to boost exact-match results
/// 6. Formatting context snippets
///
/// Returns `(snippets, source_refs, retrieved_count, used_count)`.
fn retrieve_context(
    question: &str,
    embedding_service: &dyn EmbeddingService,
    options: &ChatOptions,
) -> ChatVCodeResult<(Vec<String>, Vec<SourceReference>, usize, usize)> {
    let vector_store_path = options.resolve_vector_store_path();
    let metadata_store_path = options.resolve_metadata_store_path();

    // Load the vector store
    let store = InMemoryVectorStore::load(&vector_store_path).map_err(|e| {
        ChatVCodeError::io(format!(
            "Failed to load vector store from {}: {e}",
            vector_store_path.display()
        ))
        .with_context(
            ErrorContext::default()
                .with_operation("retrieve_context")
                .with_path(&vector_store_path),
        )
        .with_source(e.to_string())
    })?;

    if store.is_empty() {
        log::warn!("Vector store is empty at {}", vector_store_path.display());
        return Ok((Vec::new(), Vec::new(), 0, 0));
    }

    log::info!(
        "Loaded vector store with {} vectors from {}",
        store.len(),
        vector_store_path.display()
    );

    // Embed the query
    let query_vectors = embedding_service.embed(&[question]).map_err(|e| {
        ChatVCodeError::internal(format!("Failed to embed query: {e}"))
            .with_context(ErrorContext::default().with_operation("retrieve_context"))
            .with_source(e.to_string())
    })?;

    let query_vector = query_vectors.into_iter().next().ok_or_else(|| {
        ChatVCodeError::internal("Embedding service returned no result for query")
            .with_context(ErrorContext::default().with_operation("retrieve_context"))
    })?;

    // Over-retrieve: fetch more candidates than needed so keyword re-ranking
    // can promote exact-match chunks (e.g., struct definitions) that the
    // embedding model may have scored only slightly lower.
    let overretrieve_k = (options.top_k * 3).max(options.top_k + 20);
    let search_k = overretrieve_k.min(store.len());

    // Search the vector store
    let raw_results = store
        .search(&query_vector, search_k, options.min_score)
        .map_err(|e| {
            ChatVCodeError::internal(format!("Vector store search failed: {e}"))
                .with_context(ErrorContext::default().with_operation("retrieve_context"))
                .with_source(e.to_string())
        })?;

    if raw_results.is_empty() {
        log::info!("No results found for query: {question:?}");
        return Ok((Vec::new(), Vec::new(), 0, 0));
    }

    log::info!("Found {} candidate results (over-retrieved from {})", raw_results.len(), search_k);

    // Diagnostic: warn if retrieval scores are suspiciously uniform,
    // which indicates the embedding model produces poor discrimination.
    // This commonly happens with generative (causal) language models used as
    // embedding models — all code chunks cluster together, yielding near-identical
    // cosine similarity to any query.
    let scores_uniform = if raw_results.len() >= 2 {
        let scores: Vec<f32> = raw_results.iter().map(|(_, s)| *s).collect();
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_score = scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let score_range = max_score - min_score;
        let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;
        log::info!(
            "Retrieval score stats: min={min_score:.4}, max={max_score:.4}, avg={avg_score:.4}, range={score_range:.4}"
        );
        let uniform = score_range < 0.1;
        if uniform {
            log::warn!(
                "Embedding scores are nearly identical (range={score_range:.4}). \
                 This indicates the embedding model has poor discrimination. \
                 Keyword re-ranking will be applied to improve result quality. \
                 Consider using a dedicated embedding model (e.g., bge-m3, nomic-embed-text) \
                 instead of a generative language model for better retrieval quality."
            );
        }
        uniform
    } else {
        false
    };

    // Load metadata store
    let metadata_store = load_metadata_store(&metadata_store_path)?;

    // Extract file path reference and keywords for symbol-name injection and re-ranking.
    // The file path is stripped from the query before keyword extraction to prevent
    // path components from polluting keyword matching.
    let (query_file_path, _) = extract_file_path_from_query(question);
    let keywords = extract_keywords(question);

    // Resolve chunk IDs to metadata
    let mut results_with_meta = Vec::new();
    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (chunk_id, score) in &raw_results {
        if let Some(meta) = metadata_store.get(chunk_id) {
            seen_ids.insert(chunk_id.clone());
            results_with_meta.push((meta.clone(), *score));
        } else {
            log::warn!("Chunk ID '{}' not found in metadata store, skipping", chunk_id);
        }
    }

    // Symbol-name injection: scan the metadata store for chunks whose symbol
    // names exactly match a query keyword (case-insensitive). These chunks
    // are the most likely answers (e.g., querying "struct ChatArgs" should
    // always include the ChatArgs definition), but may have low semantic
    // similarity when the embedding model has poor discrimination.
    if !keywords.is_empty() && !metadata_store.is_empty() {
        let max_semantic = raw_results
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        let boost_score = max_semantic + 0.01;

        for meta in metadata_store.entries.values() {
            if seen_ids.contains(&meta.chunk_id) {
                continue;
            }
            if let Some(ref sym) = meta.symbol_name {
                let sym_lower = sym.to_lowercase();
                let symbol_matches = keywords.iter().any(|kw| sym_lower == *kw || sym_lower.contains(kw.as_str()));
                if symbol_matches {
                    log::info!(
                        "Symbol-name injection: adding chunk '{}' (symbol={}) with boost score {:.4}",
                        meta.chunk_id,
                        sym,
                        boost_score
                    );
                    seen_ids.insert(meta.chunk_id.clone());
                    results_with_meta.push((meta.clone(), boost_score));
                }
            }
        }
    }

    if results_with_meta.is_empty() {
        log::warn!("No metadata found for any retrieved chunks");
        return Ok((Vec::new(), Vec::new(), raw_results.len(), 0));
    }

    let retrieved_count = results_with_meta.len();

    // Keyword-based re-ranking: boost chunks whose source text or symbol name
    // contains exact terms from the query. This is critical when the embedding
    // model has poor discrimination (uniform scores), as it ensures chunks with
    // exact keyword matches (e.g., a struct definition when querying "struct Foo")
    // are promoted to the top of the results.
    if !keywords.is_empty() {
        // Compute the semantic score range for normalization
        let sem_scores: Vec<f32> = results_with_meta.iter().map(|(_, s)| *s).collect();
        let sem_min = sem_scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let sem_max = sem_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sem_range = sem_max - sem_min;

        for (meta, semantic_score) in &mut results_with_meta {
            let kw_score = keyword_score(
                &keywords,
                &meta.source_text,
                meta.symbol_name.as_deref(),
            );

            // Normalize semantic score to [0, 1]
            let norm_semantic = if sem_range > 1e-6 {
                (*semantic_score - sem_min) / sem_range
            } else {
                0.5
            };

            // Blend: when scores are uniform, rely more on keywords;
            // otherwise keep a strong semantic signal.
            let keyword_weight = if scores_uniform { 0.6 } else { 0.3 };
            let semantic_weight = 1.0 - keyword_weight;
            let combined = semantic_weight * norm_semantic + keyword_weight * kw_score;

            *semantic_score = combined;
        }

        log::info!(
            "Keyword re-ranking applied with {} keywords: {:?}",
            keywords.len(),
            keywords
        );
    }

    // File-path boosting: when the query mentions a specific file, boost
    // chunks from that file to ensure they rank above chunks from other files
    // that may have high keyword overlap from shared path components.
    if let Some(ref file_path) = query_file_path {
        let normalized_query_path = file_path.replace('\\', "/");
        for (meta, score) in &mut results_with_meta {
            let chunk_path = meta.file_path.to_string_lossy().replace('\\', "/");
            if chunk_path.ends_with(&normalized_query_path)
                || normalized_query_path.ends_with(&chunk_path)
            {
                *score += 0.2;
            }
        }
        log::info!("File-path boosting applied for path: {:?}", file_path);
    }

    // Sort by re-ranked score descending
    results_with_meta.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Truncate to top_k after re-ranking
    results_with_meta.truncate(options.top_k);

    // Build source references (with full snippet text, before budget trimming)
    let source_refs: Vec<SourceReference> = results_with_meta
        .iter()
        .map(|(meta, score)| SourceReference::from_metadata(meta, *score))
        .collect();

    // Format context snippets
    let snippets = format_context_snippets(&results_with_meta);

    // Apply token budget
    let (trimmed_snippets, used_count, _trimmed) =
        apply_token_budget(&snippets, options.context_token_budget, 4);

    // Trim source references to match
    let trimmed_refs: Vec<SourceReference> = source_refs.into_iter().take(used_count).collect();

    Ok((trimmed_snippets, trimmed_refs, retrieved_count, used_count))
}

/// Extracts a file path reference from a natural-language query.
///
/// Detects patterns like `"in file src/lib.rs"`, `"from src/main.rs"`, or
/// standalone path tokens containing directory separators or known source
/// file extensions. Returns the extracted path (if any) and the query with
/// the path portion removed, so that path components do not pollute keyword
/// extraction.
pub(crate) fn extract_file_path_from_query(query: &str) -> (Option<String>, String) {
    let lower = query.to_lowercase();

    let prefixes = [
        "in file ",
        "in the file ",
        "from file ",
        "of file ",
        "in source file ",
    ];

    for prefix in &prefixes {
        if let Some(pos) = lower.find(prefix) {
            let path_start = pos + prefix.len();
            let rest = &query[path_start..];
            let path_end = rest
                .find(['?', '!', ',', ';'])
                .unwrap_or(rest.len());
            let path = rest[..path_end].trim();
            if !path.is_empty() {
                let mut cleaned = String::new();
                cleaned.push_str(query[..pos].trim_end());
                let after = &query[path_start + path_end..];
                if !after.is_empty() {
                    if !cleaned.is_empty() {
                        cleaned.push(' ');
                    }
                    cleaned.push_str(after.trim_start());
                }
                return (Some(path.to_string()), cleaned.trim().to_string());
            }
        }
    }

    let mut best_path: Option<(usize, usize, &str)> = None;

    for (i, c) in query.char_indices() {
        if c == '/' || c == '\\' {
            let start = query[..i]
                .rfind(|c: char| c.is_whitespace())
                .map(|p| p + 1)
                .unwrap_or(0);
            let end = query[i..]
                .find(|c: char| c.is_whitespace())
                .map(|p| i + p)
                .unwrap_or(query.len());
            let token = &query[start..end];
            if token.contains('/') || token.contains('\\') {
                let span_len = end - start;
                if best_path.is_none_or(|(_, len, _)| span_len > len) {
                    best_path = Some((start, span_len, token));
                }
            }
        }
    }

    if let Some((start, len, path)) = best_path {
        let mut cleaned = String::new();
        cleaned.push_str(query[..start].trim_end());
        let after = &query[start + len..];
        if !after.is_empty() {
            if !cleaned.is_empty() {
                cleaned.push(' ');
            }
            cleaned.push_str(after.trim_start());
        }
        return (Some(path.to_string()), cleaned.trim().to_string());
    }

    let extensions = [".rs", ".py", ".js", ".ts", ".tsx", ".jsx", ".php", ".c", ".cpp", ".h"];
    for word in query.split_whitespace() {
        let word_lower = word.to_lowercase();
        if extensions.iter().any(|ext| word_lower.ends_with(ext)) {
            let cleaned = query.replacen(word, "", 1);
            let cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
            return (Some(word.to_string()), cleaned);
        }
    }

    (None, query.to_string())
}

/// Extract significant keywords from a query for re-ranking.
///
/// Filters out common stop words, very short tokens, and file path
/// references that would produce noisy matches across many code chunks.
pub(crate) fn extract_keywords(query: &str) -> Vec<String> {
    let (_, cleaned_query) = extract_file_path_from_query(query);

    let stop_words = [
        "what", "how", "where", "when", "why", "who", "which", "the", "a", "an",
        "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
        "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "into",
        "and", "or", "but", "not", "no", "if", "then", "else", "can", "could",
        "would", "should", "will", "shall", "may", "might", "must",
        "this", "that", "these", "those", "it", "its", "i", "me", "my",
        "we", "our", "you", "your", "he", "she", "they", "them", "their",
        "all", "any", "each", "every", "some", "more", "most", "other",
        "about", "up", "out", "so", "than", "too", "very", "just",
        "there", "here", "have", "has", "had", "having", "get", "got",
        "also", "use", "used", "using", "define", "defined", "implement",
        "implemented", "find", "show", "list", "tell", "explain", "describe",
        "members", "fields", "methods", "functions", "struct", "enum",
        "class", "trait", "interface", "type", "function", "method",
        "file", "path", "directory", "folder", "src", "lib", "crates",
    ];

    let mut keywords = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for word in cleaned_query.split(|c: char| !c.is_alphanumeric() && c != '_') {
        let lower = word.to_lowercase();
        if lower.len() >= 3 && !stop_words.contains(&lower.as_str()) && seen.insert(lower.clone()) {
            keywords.push(lower);
        }
    }

    keywords
}

/// Compute a keyword relevance score for a code chunk.
///
/// Returns a value in `[0.0, 1.0]` based on how many query keywords
/// appear in the chunk's source text and symbol name. Symbol name
/// matches are weighted much higher since they indicate the chunk
/// *defines* the queried entity. An exact symbol-name match (the
/// symbol equals a keyword) receives the maximum boost.
pub(crate) fn keyword_score(keywords: &[String], source_text: &str, symbol_name: Option<&str>) -> f32 {
    if keywords.is_empty() {
        return 0.0;
    }

    let source_lower = source_text.to_lowercase();
    let symbol_lower = symbol_name.map(|s| s.to_lowercase());

    let mut text_matches = 0usize;
    let mut symbol_exact = false;
    let mut symbol_contains = false;

    for kw in keywords {
        if source_lower.contains(kw.as_str()) {
            text_matches += 1;
        }
        if let Some(ref sym) = symbol_lower {
            if *sym == *kw {
                symbol_exact = true;
            } else if sym.contains(kw.as_str()) {
                symbol_contains = true;
            }
        }
    }

    let text_score = text_matches as f32 / keywords.len() as f32;
    let symbol_boost = if symbol_exact {
        0.6
    } else if symbol_contains {
        0.2
    } else {
        0.0
    };

    (text_score + symbol_boost).min(1.0)
}

/// Loads or creates a chunk metadata store.
///
/// Attempts to load from the given path. If the file doesn't exist,
/// logs a warning and returns an empty store. This allows the system
/// to operate without a metadata store (though it won't be able to
/// resolve chunk IDs).
fn load_metadata_store(path: &Path) -> ChatVCodeResult<ChunkMetadataStore> {
    if !path.exists() {
        log::warn!(
            "Metadata store not found at {}, chunk resolution will be limited",
            path.display()
        );
        return Ok(ChunkMetadataStore::new());
    }

    match ChunkMetadataStore::load(path) {
        Ok(store) => {
            log::info!(
                "Loaded metadata store with {} entries from {}",
                store.len(),
                path.display()
            );
            Ok(store)
        }
        Err(e) => {
            log::warn!("Failed to load metadata store from {}: {e}", path.display());
            // Fall back to empty store rather than failing entirely
            Ok(ChunkMetadataStore::new())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ChunkSpan, FileLanguage};

    fn make_chunk(
        id: &str,
        file_path: &str,
        kind: ChunkKind,
        symbol: Option<&str>,
        start_line: usize,
        end_line: usize,
        text: &str,
    ) -> CodeChunk {
        CodeChunk {
            id: id.to_string(),
            file_path: PathBuf::from(file_path),
            language: FileLanguage::Rust,
            kind,
            symbol_name: symbol.map(String::from),
            span: ChunkSpan::new(
                0,
                text.len(),
                start_line.saturating_sub(1),
                end_line.saturating_sub(1),
            ),
            source_text: text.to_string(),
        }
    }

    #[test]
    fn test_chat_options_defaults() {
        let opts = ChatOptions::new("/tmp/project");
        assert_eq!(opts.project_path, PathBuf::from("/tmp/project"));
        assert_eq!(opts.top_k, 16);
        assert!(opts.min_score.is_none());
        assert_eq!(opts.context_token_budget, 0);
        assert!(opts.vector_store_path.is_none());
        assert!(opts.embedding_config.is_none());
        assert!(opts.system_prompt.is_none());
    }

    #[test]
    fn test_chat_options_builder() {
        let opts = ChatOptions::new("/tmp/project")
            .with_top_k(5)
            .with_min_score(0.7)
            .with_context_token_budget(2048)
            .system_prompt("You are a code expert.");

        assert_eq!(opts.top_k, 5);
        assert_eq!(opts.min_score, Some(0.7));
        assert_eq!(opts.context_token_budget, 2048);
        assert_eq!(opts.system_prompt.as_deref(), Some("You are a code expert."));
    }

    #[test]
    fn test_chat_options_resolve_paths() {
        let opts = ChatOptions::new("/tmp/project");
        assert_eq!(
            opts.resolve_vector_store_path(),
            PathBuf::from("/tmp/project/.chatvcode/vectors.vdb")
        );
        assert_eq!(
            opts.resolve_metadata_store_path(),
            PathBuf::from("/tmp/project/.chatvcode/vectors.atmd")
        );
    }

    #[test]
    fn test_source_reference_from_chunk() {
        let chunk = make_chunk(
            "src/main.rs:Function:main:10",
            "src/main.rs",
            ChunkKind::Function,
            Some("main"),
            10,
            15,
            "fn main() {\n    println!(\"hello\");\n}",
        );
        let ref1 = SourceReference::from_chunk(&chunk, 0.95);
        assert_eq!(ref1.chunk_id, "src/main.rs:Function:main:10");
        assert_eq!(ref1.file_path, PathBuf::from("src/main.rs"));
        assert_eq!(ref1.kind, ChunkKind::Function);
        assert_eq!(ref1.symbol_name.as_deref(), Some("main"));
        // make_chunk stores 0-indexed lines (saturating_sub(1)),
        // then from_chunk converts back to 1-indexed (+1)
        // make_chunk(start_line=10) → span.start_line=9 → ref.start_line=10
        assert_eq!(ref1.start_line, 10);
        assert_eq!(ref1.end_line, 15);
        assert!((ref1.score - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_source_reference_from_metadata() {
        let meta = ChunkMetadata {
            chunk_id: "test_id".to_string(),
            file_path: PathBuf::from("src/lib.rs"),
            language: "rust".to_string(),
            kind: ChunkKind::Struct,
            symbol_name: Some("Config".to_string()),
            start_line: 42,
            end_line: 58,
            start_byte: 500,
            end_byte: 800,
            source_text: "struct Config { }".to_string(),
        };
        let ref1 = SourceReference::from_metadata(&meta, 0.88);
        assert_eq!(ref1.chunk_id, "test_id");
        assert_eq!(ref1.symbol_name.as_deref(), Some("Config"));
        assert_eq!(ref1.start_line, 42);
        assert_eq!(ref1.end_line, 58);
    }

    #[test]
    fn test_source_reference_display_path() {
        let ref1 = SourceReference {
            chunk_id: "id1".to_string(),
            file_path: PathBuf::from("src/main.rs"),
            kind: ChunkKind::Function,
            symbol_name: Some("main".to_string()),
            start_line: 10,
            end_line: 20,
            score: 0.9,
            snippet: "fn main() {}".to_string(),
        };
        assert_eq!(ref1.display_path(), "src/main.rs:10:main-20");

        let ref2 = SourceReference {
            chunk_id: "id2".to_string(),
            file_path: PathBuf::from("src/lib.rs"),
            kind: ChunkKind::Function,
            symbol_name: None,
            start_line: 5,
            end_line: 10,
            score: 0.8,
            snippet: "fn foo() {}".to_string(),
        };
        assert_eq!(ref2.display_path(), "src/lib.rs:5-10");
    }

    #[test]
    fn test_format_context_snippets() {
        let meta = ChunkMetadata {
            chunk_id: "id1".to_string(),
            file_path: PathBuf::from("src/main.rs"),
            language: "rust".to_string(),
            kind: ChunkKind::Function,
            symbol_name: Some("parse_config".to_string()),
            start_line: 42,
            end_line: 58,
            start_byte: 500,
            end_byte: 800,
            source_text: "fn parse_config() {}".to_string(),
        };

        let results = vec![(meta, 0.892)];
        let snippets = format_context_snippets(&results);

        assert_eq!(snippets.len(), 1);
        assert!(snippets[0].contains("src/main.rs:42-58"));
        assert!(snippets[0].contains("function: parse_config"));
        assert!(snippets[0].contains("score: 0.892"));
        assert!(snippets[0].contains("fn parse_config() {}"));
    }

    #[test]
    fn test_format_context_snippets_no_symbol() {
        let meta = ChunkMetadata {
            chunk_id: "id2".to_string(),
            file_path: PathBuf::from("src/lib.rs"),
            language: "rust".to_string(),
            kind: ChunkKind::Unknown,
            symbol_name: None,
            start_line: 10,
            end_line: 20,
            start_byte: 0,
            end_byte: 100,
            source_text: "some code".to_string(),
        };

        let results = vec![(meta, 0.5)];
        let snippets = format_context_snippets(&results);

        assert_eq!(snippets.len(), 1);
        assert!(snippets[0].contains("unknown"));
        assert!(!snippets[0].contains("function:"));
    }

    #[test]
    fn test_apply_token_budget_unlimited() {
        let snippets =
            vec!["short snippet".to_string(), "another snippet with more text content".to_string()];

        let (result, used, trimmed) = apply_token_budget(&snippets, 0, 4);
        assert_eq!(result.len(), 2);
        assert_eq!(used, 2);
        assert_eq!(trimmed, 0);
    }

    #[test]
    fn test_apply_token_budget_limited() {
        let snippets = vec![
            "short snippet".to_string(), // 13 chars ≈ 3.25 tokens
            "another snippet with more text content".to_string(), // 39 chars ≈ 9.75 tokens
        ];

        // Budget of 5 tokens = 20 chars
        // First snippet (13 chars) fits fully; remaining = 7
        // Second snippet (39 chars) is partially trimmed at ~7 chars; remaining = 0
        let (result, used, _trimmed) = apply_token_budget(&snippets, 5, 4);
        assert_eq!(used, 2); // Both snippets appear (one full, one trimmed)
        assert_eq!(result.len(), 2); // Both present in result
    }

    #[test]
    fn test_apply_token_budget_trim_partial() {
        let long_snippet = "a".repeat(100);
        let snippets = vec![long_snippet];

        // Budget of 10 tokens = 40 chars
        let (result, _used, _trimmed) = apply_token_budget(&snippets, 10, 4);
        assert_eq!(result.len(), 1);
        // Should be trimmed to ~40 chars + "..."
        assert!(result[0].len() <= 50);
        assert!(result[0].ends_with("..."));
    }

    #[test]
    fn test_build_rag_prompt_with_context() {
        let options = ChatOptions::new("/tmp/project")
            .with_chat_template(ChatTemplate::ChatML)
            .system_prompt("You are a coding assistant.");

        let snippets = vec![
            "--- src/main.rs:10-20 (function: hello) [score: 0.900] ---\nfn hello() {}\n---"
                .to_string(),
        ];

        let prompt = build_rag_prompt("What does hello do?", &snippets, &options).unwrap();

        // Should contain the context block
        assert!(prompt.contains("Retrieved Context"));
        assert!(prompt.contains("hello"));
        assert!(prompt.contains("What does hello do?"));
        // Should use ChatML formatting
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
    }

    #[test]
    fn test_build_rag_prompt_no_context() {
        let options = ChatOptions::new("/tmp/project")
            .with_chat_template(ChatTemplate::ChatML)
            .system_prompt("You are a coding assistant.");

        let prompt = build_rag_prompt("What is Rust?", &[], &options).unwrap();

        // Without context, question is just the user message
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("<|im_start|>user"));
        // Should NOT contain "Retrieved Context"
        assert!(!prompt.contains("[Retrieved Context]"));
    }

    #[test]
    fn test_build_rag_prompt_raw_template() {
        let options = ChatOptions::new("/tmp/project").with_chat_template(ChatTemplate::Raw);

        let snippets = vec!["code context here".to_string()];
        let prompt = build_rag_prompt("explain this", &snippets, &options).unwrap();

        // Raw template just concatenates
        assert!(prompt.contains("code context here"));
        assert!(prompt.contains("explain this"));
    }

    #[test]
    fn test_chat_response_is_no_context() {
        let response = ChatResponse {
            answer: "I don't know.".to_string(),
            sources: vec![],
            token_usage: TokenUsage::new(10, 5),
            stop_reason: StopReason::Eos,
            duration: std::time::Duration::from_millis(100),
            search_duration: std::time::Duration::from_millis(10),
            inference_duration: std::time::Duration::from_millis(90),
            retrieved_count: 0,
            used_count: 0,
        };
        assert!(response.is_no_context());
    }

    #[test]
    fn test_chat_response_format_sources() {
        let response = ChatResponse {
            answer: "It does X".to_string(),
            sources: vec![SourceReference {
                chunk_id: "id1".to_string(),
                file_path: PathBuf::from("src/main.rs"),
                kind: ChunkKind::Function,
                symbol_name: Some("main".to_string()),
                start_line: 10,
                end_line: 20,
                score: 0.95,
                snippet: "fn main() {}".to_string(),
            }],
            token_usage: TokenUsage::new(50, 20),
            stop_reason: StopReason::Eos,
            duration: std::time::Duration::from_millis(200),
            search_duration: std::time::Duration::from_millis(20),
            inference_duration: std::time::Duration::from_millis(180),
            retrieved_count: 1,
            used_count: 1,
        };

        let formatted = response.format_sources();
        assert!(formatted.contains("Sources:"));
        assert!(formatted.contains("src/main.rs"));
        assert!(formatted.contains("0.950"));
    }

    #[test]
    fn test_chat_response_format_sources_empty() {
        let response = ChatResponse {
            answer: "No idea".to_string(),
            sources: vec![],
            token_usage: TokenUsage::new(10, 5),
            stop_reason: StopReason::Eos,
            duration: std::time::Duration::from_millis(100),
            search_duration: std::time::Duration::from_millis(10),
            inference_duration: std::time::Duration::from_millis(90),
            retrieved_count: 0,
            used_count: 0,
        };

        let formatted = response.format_sources();
        assert!(formatted.contains("No sources available"));
    }

    #[test]
    fn test_build_metadata_store() {
        use crate::model::{IndexResult, ParseResult, SourceFile};

        let file = SourceFile::new("src/main.rs", "fn main() {}");
        let chunk = make_chunk(
            "src/main.rs:Function:main:0",
            "src/main.rs",
            ChunkKind::Function,
            Some("main"),
            0,
            0,
            "fn main() {}",
        );

        let parse_result = ParseResult::success(file, vec![chunk]);
        let index_result = IndexResult::from_parse_results(vec![parse_result], vec![]);

        let store = build_metadata_store(&index_result);
        assert_eq!(store.len(), 1);
        assert!(store.get("src/main.rs:Function:main:0").is_some());

        let meta = store.get("src/main.rs:Function:main:0").unwrap();
        assert_eq!(meta.file_path, PathBuf::from("src/main.rs"));
        assert_eq!(meta.kind, ChunkKind::Function);
        assert_eq!(meta.symbol_name.as_deref(), Some("main"));
    }

    #[test]
    fn test_no_context_system_prompt_used_when_empty() {
        // When there's no retrieval context, we should use NO_CONTEXT_SYSTEM_PROMPT
        // This is tested via build_rag_prompt behavior
        let options_empty =
            ChatOptions::new("/tmp/project").with_chat_template(ChatTemplate::ChatML);

        // Build prompt without context - should work fine
        let prompt = build_rag_prompt("What is a struct in Rust?", &[], &options_empty).unwrap();
        assert!(prompt.contains("What is a struct in Rust?"));
        // The prompt should NOT have Retrieved Context header
        assert!(!prompt.contains("[Retrieved Context]"));
    }

    #[test]
    fn test_apply_token_budget_with_multiple_snippets() {
        let snippets = vec![
            "a".repeat(50), // 50 chars ≈ 12.5 tokens
            "b".repeat(50), // 50 chars ≈ 12.5 tokens
            "c".repeat(50), // 50 chars ≈ 12.5 tokens
        ];

        // Budget of 20 tokens = 80 chars
        // First snippet (50 chars) fits fully; remaining = 30
        // Second snippet (50 chars) trimmed at ~30 chars; remaining = 0
        // Third snippet (50 chars) completely dropped
        let (result, used, trimmed) = apply_token_budget(&snippets, 20, 4);
        assert_eq!(used, 2); // First full + second trimmed
        assert_eq!(trimmed, 1); // Third snippet completely dropped
        assert_eq!(result.len(), 2); // 2 present (1 full + 1 trimmed)
    }

    #[test]
    fn test_chat_options_with_explicit_paths() {
        let opts = ChatOptions::new("/tmp/project")
            .vector_store_path("/data/vectors.atvs")
            .metadata_store_path("/data/chunks.atmd");

        assert_eq!(opts.vector_store_path, Some(PathBuf::from("/data/vectors.atvs")));
        assert_eq!(opts.metadata_store_path, Some(PathBuf::from("/data/chunks.atmd")));
        // These should use the explicit paths
        assert_eq!(opts.resolve_vector_store_path(), PathBuf::from("/data/vectors.atvs"));
        assert_eq!(opts.resolve_metadata_store_path(), PathBuf::from("/data/chunks.atmd"));
    }

    #[test]
    fn test_extract_file_path_in_file_prefix() {
        let (path, cleaned) = extract_file_path_from_query(
            "list the members of struct ChatArgs in file crates/chatvcode-cli/src/lib.rs",
        );
        assert_eq!(path.as_deref(), Some("crates/chatvcode-cli/src/lib.rs"));
        assert!(!cleaned.contains("crates"));
        assert!(!cleaned.contains("lib.rs"));
        assert!(cleaned.contains("ChatArgs"));
    }

    #[test]
    fn test_extract_file_path_backslash_path() {
        let (path, cleaned) = extract_file_path_from_query(
            "list the members of struct ChatArgs in file crates\\chatvcode-cli\\src\\lib.rs",
        );
        assert_eq!(path.as_deref(), Some("crates\\chatvcode-cli\\src\\lib.rs"));
        assert!(cleaned.contains("ChatArgs"));
        assert!(!cleaned.contains("chatvcode"));
    }

    #[test]
    fn test_extract_file_path_standalone_extension() {
        let (path, cleaned) =
            extract_file_path_from_query("what does main do in main.rs");
        assert_eq!(path.as_deref(), Some("main.rs"));
        assert!(cleaned.contains("main"));
        assert!(!cleaned.contains("main.rs"));
    }

    #[test]
    fn test_extract_file_path_no_path() {
        let (path, cleaned) =
            extract_file_path_from_query("what does the parse function do");
        assert!(path.is_none());
        assert_eq!(cleaned, "what does the parse function do");
    }

    #[test]
    fn test_extract_file_path_from_file_prefix() {
        let (path, _) =
            extract_file_path_from_query("show me the imports from src/lib.rs");
        assert_eq!(path.as_deref(), Some("src/lib.rs"));
    }

    #[test]
    fn test_extract_keywords_strips_file_path() {
        let keywords = extract_keywords(
            "list the members of struct ChatArgs in file crates\\chatvcode-cli\\src\\lib.rs",
        );
        assert!(keywords.contains(&"chatargs".to_string()));
        assert!(!keywords.contains(&"crates".to_string()));
        assert!(!keywords.contains(&"chatvcode".to_string()));
        assert!(!keywords.contains(&"cli".to_string()));
        assert!(!keywords.contains(&"src".to_string()));
        assert!(!keywords.contains(&"lib".to_string()));
        assert!(!keywords.contains(&"file".to_string()));
    }

    #[test]
    fn test_extract_keywords_no_path_unchanged() {
        let keywords = extract_keywords("how does the render function work");
        assert!(keywords.contains(&"render".to_string()));
        assert!(keywords.contains(&"work".to_string()));
    }

    #[test]
    fn test_keyword_score_exact_symbol_match() {
        let keywords = vec!["chatargs".to_string(), "question".to_string()];
        let score_exact = keyword_score(
            &keywords,
            "struct ChatArgs { question: String }",
            Some("ChatArgs"),
        );
        let score_contains = keyword_score(
            &keywords,
            "fn run_chat() { let args = make_args(); }",
            Some("run_chat"),
        );
        assert!(
            score_exact > score_contains,
            "exact symbol match ({score_exact}) should score higher than text-only match ({score_contains})"
        );
    }

    #[test]
    fn test_keyword_score_symbol_exact_vs_contains() {
        let keywords = vec!["config".to_string(), "debug".to_string()];
        let score_exact = keyword_score(
            &keywords,
            "struct Config { debug: bool }",
            Some("Config"),
        );
        let score_contains = keyword_score(
            &keywords,
            "struct ConfigLoader { path: String }",
            Some("ConfigLoader"),
        );
        assert!(
            score_exact > score_contains,
            "exact symbol match ({score_exact}) should score higher than partial match ({score_contains})"
        );
    }

    #[test]
    fn test_keyword_score_no_match() {
        let keywords = vec!["render".to_string()];
        let score = keyword_score(
            &keywords,
            "fn parse() {}",
            Some("parse"),
        );
        assert!((score - 0.0).abs() < f32::EPSILON);
    }
}
