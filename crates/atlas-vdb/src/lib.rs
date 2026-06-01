#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::match_same_arms,
    clippy::option_if_let_else,
    clippy::format_push_string,
    clippy::doc_markdown,
    clippy::missing_const_for_fn,
    clippy::map_unwrap_or,
    clippy::items_after_statements,
    clippy::similar_names,
    clippy::significant_drop_tightening,
    clippy::unused_self,
    clippy::assigning_clones
)]
mod cache;
mod config;
mod download;
mod embedding;
mod error;
mod hnsw;
mod model;
mod onnx;
mod quantization;
mod registry;
mod similarity;
mod store;

pub use cache::{CachedSearchResult, SearchCache, cached_search};
pub use config::{EmbeddingConfig, ExecutionProvider};
pub use download::ModelDownloader;
pub use embedding::{EmbeddingService, MockEmbeddingService};
pub use error::{VdbContext, VdbError, VdbErrorKind, VdbErrorSeverity, VdbResult};
pub use hnsw::HnswVectorStore;
pub use model::{EmbeddingVector, SearchQuery, SearchResult};
pub use onnx::OnnxEmbeddingService;
pub use quantization::QuantizedVectorStore;
pub use registry::{ModelInfo, ModelRegistry, builtin_model_names, builtin_models};
pub use similarity::{cosine_similarity, dot_product};
pub use store::{CompactVectorStore, InMemoryVectorStore, VectorStore};
