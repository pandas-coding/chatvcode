use crate::error::VdbResult;

pub trait EmbeddingService: Send + Sync {
    fn embed(&self, texts: &[&str]) -> VdbResult<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}
