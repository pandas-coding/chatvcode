use serde::{Deserialize, Serialize};

/// A vector embedding associated with a code chunk.
///
/// Stores the chunk identifier, the floating-point vector, and its dimension.
/// Used by [`VectorStore`](crate::VectorStore) for storage and retrieval.
///
/// # Examples
///
/// ```
/// use atlas_vdb::EmbeddingVector;
///
/// let ev = EmbeddingVector::new("chunk_1", vec![1.0, 0.0, 0.0]);
/// assert_eq!(ev.chunk_id, "chunk_1");
/// assert_eq!(ev.dimension, 3);
/// assert_eq!(ev.len(), 3);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingVector {
    /// The unique identifier of the associated code chunk.
    pub chunk_id: String,
    /// The embedding vector data.
    pub vector: Vec<f32>,
    /// The dimension (length) of the vector.
    pub dimension: usize,
}

impl EmbeddingVector {
    /// Creates a new embedding vector. The dimension is inferred from the vector length.
    pub fn new(chunk_id: impl Into<String>, vector: Vec<f32>) -> Self {
        let dimension = vector.len();
        Self { chunk_id: chunk_id.into(), vector, dimension }
    }

    pub fn len(&self) -> usize {
        self.dimension
    }

    pub fn is_empty(&self) -> bool {
        self.dimension == 0
    }
}

/// A semantic search query.
///
/// Specifies the search text, the number of top results to return (`top_k`),
/// and an optional minimum similarity score threshold.
///
/// # Examples
///
/// ```
/// use atlas_vdb::SearchQuery;
///
/// let query = SearchQuery::new("find error handling", 5);
/// assert_eq!(query.top_k, 5);
/// assert!(query.min_score.is_none());
///
/// let query = query.with_min_score(0.5);
/// assert_eq!(query.min_score, Some(0.5));
/// ```
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// The search query text.
    pub text: String,
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Optional minimum cosine similarity threshold. Results below this score are filtered out.
    pub min_score: Option<f32>,
}

impl SearchQuery {
    /// Creates a new search query with the given text and top_k.
    pub fn new(text: impl Into<String>, top_k: usize) -> Self {
        Self { text: text.into(), top_k, min_score: None }
    }

    /// Sets the minimum similarity score threshold. Results with scores below this are excluded.
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }
}

/// A single result from a semantic search operation.
///
/// Contains the chunk identifier, similarity score, and optionally the
/// associated [`CodeChunk`] metadata.
///
/// # Examples
///
/// ```
/// use atlas_vdb::SearchResult;
///
/// let result = SearchResult::new("chunk_42", 0.95);
/// assert_eq!(result.chunk_id, "chunk_42");
/// assert!((result.score - 0.95).abs() < 1e-6);
/// assert!(result.chunk.is_none());
/// ```
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The unique identifier of the matching code chunk.
    pub chunk_id: String,
    /// The cosine similarity score (0 to 1, higher is better).
    pub score: f32,
    /// The resolved [`CodeChunk`] metadata, if available.
    pub chunk: Option<CodeChunk>,
}

impl SearchResult {
    /// Creates a new search result with the given chunk ID and score.
    pub fn new(chunk_id: impl Into<String>, score: f32) -> Self {
        Self { chunk_id: chunk_id.into(), score, chunk: None }
    }

    /// Attaches [`CodeChunk`] metadata to this search result.
    pub fn with_chunk(mut self, chunk: CodeChunk) -> Self {
        self.chunk = Some(chunk);
        self
    }
}

/// A code chunk extracted from a source file.
///
/// Represents a semantic unit of code (function, struct, etc.) with its
/// identifier, type, location, and source text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    /// A unique identifier for this chunk, typically derived from file path and span.
    pub id: String,
    /// The path to the source file containing this chunk.
    pub file_path: std::path::PathBuf,
    /// The type of code construct this chunk represents.
    pub kind: ChunkKind,
    /// The symbol name (e.g., function name, struct name) if available.
    pub symbol_name: Option<String>,
    /// The byte and line span of this chunk within the source file.
    pub span: ChunkSpan,
    /// The raw source code text of this chunk.
    pub source_text: String,
}

/// Classification of code chunk types.
///
/// Each variant corresponds to a syntactic construct recognized by the
/// tree-sitter parser and `atlas-parser`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkKind {
    /// A function definition.
    Function,
    /// A struct definition.
    Struct,
    /// An enum definition.
    Enum,
    /// A trait definition.
    Trait,
    /// An impl block.
    Impl,
    /// A class definition.
    Class,
    /// An interface definition.
    Interface,
    /// A type alias definition.
    TypeAlias,
    /// A module declaration.
    Module,
    /// A constant definition.
    Constant,
    /// A method definition (within an impl or class).
    Method,
    /// An unrecognized chunk type.
    Unknown,
}

/// Byte and line range of a code chunk within its source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkSpan {
    /// Starting byte offset (0-indexed) in the source file.
    pub start_byte: usize,
    /// Ending byte offset (exclusive) in the source file.
    pub end_byte: usize,
    /// Starting line number (1-indexed).
    pub start_line: usize,
    /// Ending line number (1-indexed, inclusive).
    pub end_line: usize,
}
