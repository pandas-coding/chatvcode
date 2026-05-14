use std::path::Path;

use crate::error::VdbResult;
use crate::model::EmbeddingVector;

pub trait VectorStore: Send + Sync {
    fn add(&mut self, vectors: Vec<EmbeddingVector>) -> VdbResult<()>;
    fn search(&self, query: &[f32], top_k: usize) -> VdbResult<Vec<(String, f32)>>;
    fn save(&self, path: &Path) -> VdbResult<()>;
    fn load(path: &Path) -> VdbResult<Self>
    where
        Self: Sized;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn clear(&mut self);
}
