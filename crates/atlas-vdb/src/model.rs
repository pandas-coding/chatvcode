use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingVector {
    pub chunk_id: String,
    pub vector: Vec<f32>,
    pub dimension: usize,
}

impl EmbeddingVector {
    pub fn new(chunk_id: impl Into<String>, vector: Vec<f32>) -> Self {
        let dimension = vector.len();
        Self {
            chunk_id: chunk_id.into(),
            vector,
            dimension,
        }
    }

    pub fn len(&self) -> usize {
        self.dimension
    }

    pub fn is_empty(&self) -> bool {
        self.dimension == 0
    }
}

#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub text: String,
    pub top_k: usize,
    pub min_score: Option<f32>,
}

impl SearchQuery {
    pub fn new(text: impl Into<String>, top_k: usize) -> Self {
        Self {
            text: text.into(),
            top_k,
            min_score: None,
        }
    }

    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk_id: String,
    pub score: f32,
    pub chunk: Option<CodeChunk>,
}

impl SearchResult {
    pub fn new(chunk_id: impl Into<String>, score: f32) -> Self {
        Self {
            chunk_id: chunk_id.into(),
            score,
            chunk: None,
        }
    }

    pub fn with_chunk(mut self, chunk: CodeChunk) -> Self {
        self.chunk = Some(chunk);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    pub id: String,
    pub file_path: std::path::PathBuf,
    pub kind: ChunkKind,
    pub symbol_name: Option<String>,
    pub span: ChunkSpan,
    pub source_text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Class,
    Interface,
    TypeAlias,
    Module,
    Constant,
    Method,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkSpan {
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
}
