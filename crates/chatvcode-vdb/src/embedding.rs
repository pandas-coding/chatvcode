use crate::error::VdbResult;

/// Trait for text embedding services.
///
/// Implementations convert text strings into fixed-dimension floating-point
/// vectors suitable for similarity search. The trait requires `Send + Sync`
/// so services can be shared across threads.
///
/// # Required methods
///
/// - [`embed`](EmbeddingService::embed): Converts a batch of text strings to vectors.
/// - [`dimension`](EmbeddingService::dimension): Returns the output vector dimension.
///
/// # Examples
///
/// Implementing a custom embedding service:
///
/// ```
/// use chatvcode_vdb::EmbeddingService;
/// use chatvcode_vdb::VdbResult;
///
/// struct MyEmbedder {
///     dim: usize,
/// }
///
/// impl EmbeddingService for MyEmbedder {
///     fn embed(&self, texts: &[&str]) -> VdbResult<Vec<Vec<f32>>> {
///         Ok(texts.iter().map(|_| vec![0.0; self.dim]).collect())
///     }
///
///     fn dimension(&self) -> usize {
///         self.dim
///     }
/// }
/// ```
pub trait EmbeddingService: Send + Sync {
    /// Converts the given text strings into embedding vectors.
    ///
    /// The returned vectors are in the same order as the input texts.
    fn embed(&self, texts: &[&str]) -> VdbResult<Vec<Vec<f32>>>;
    /// Returns the output dimension of the embedding vectors.
    fn dimension(&self) -> usize;
}

/// A mock embedding service for testing.
///
/// Generates deterministic pseudo-embeddings from input text bytes.
/// Each byte of the input is mapped to a float value in the vector,
/// and the resulting vector is L2-normalized. Identical inputs always
/// produce identical outputs.
///
/// # Examples
///
/// ```
/// use chatvcode_vdb::MockEmbeddingService;
/// use chatvcode_vdb::EmbeddingService;
///
/// let service = MockEmbeddingService::new(64);
/// assert_eq!(service.dimension(), 64);
///
/// let vectors = service.embed(&["hello", "world"]).unwrap();
/// assert_eq!(vectors.len(), 2);
/// assert_eq!(vectors[0].len(), 64);
/// ```
pub struct MockEmbeddingService {
    dimension: usize,
}

impl MockEmbeddingService {
    /// Creates a new mock embedding service with the given output dimension.
    #[must_use]
    pub const fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl EmbeddingService for MockEmbeddingService {
    fn embed(&self, texts: &[&str]) -> VdbResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            let mut vector = vec![0.0f32; self.dimension];
            if !text.is_empty() {
                let bytes = text.as_bytes();
                for (i, &byte) in bytes.iter().enumerate() {
                    if i >= self.dimension {
                        break;
                    }
                    vector[i] = f32::from(byte) / 255.0;
                }
                let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for val in &mut vector {
                        *val /= norm;
                    }
                }
            }
            results.push(vector);
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_embedding_dimension() {
        for dim in [8, 32, 64, 128, 256] {
            let service = MockEmbeddingService::new(dim);
            assert_eq!(service.dimension(), dim);
            let results = service.embed(&["hello"]).unwrap();
            assert_eq!(results[0].len(), dim);
        }
    }

    #[test]
    fn test_mock_embedding_consistency_same_call() {
        let service = MockEmbeddingService::new(64);
        let r1 = service.embed(&["hello world"]).unwrap();
        let r2 = service.embed(&["hello world"]).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_mock_embedding_consistency_across_batches() {
        let service = MockEmbeddingService::new(32);
        let single = service.embed(&["alpha"]).unwrap()[0].clone();
        let batch = service.embed(&["alpha", "beta"]).unwrap();
        assert_eq!(single, batch[0]);
    }

    #[test]
    fn test_mock_embedding_different_inputs() {
        let service = MockEmbeddingService::new(64);
        let r1 = service.embed(&["hello"]).unwrap();
        let r2 = service.embed(&["world"]).unwrap();
        assert_ne!(r1[0], r2[0]);
    }

    #[test]
    fn test_mock_embedding_empty_text() {
        let service = MockEmbeddingService::new(32);
        let results = service.embed(&[""]).unwrap();
        assert_eq!(results[0].len(), 32);
        assert!(results[0].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_mock_embedding_batch_ordering() {
        let service = MockEmbeddingService::new(16);
        let results = service.embed(&["alpha", "beta", "gamma"]).unwrap();
        assert_eq!(results.len(), 3);
        let single_alpha = service.embed(&["alpha"]).unwrap()[0].clone();
        let single_beta = service.embed(&["beta"]).unwrap()[0].clone();
        assert_eq!(results[0], single_alpha);
        assert_eq!(results[1], single_beta);
    }

    #[test]
    fn test_mock_embedding_empty_batch() {
        let service = MockEmbeddingService::new(16);
        let results = service.embed(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_mock_embedding_normalized() {
        let service = MockEmbeddingService::new(32);
        let results = service.embed(&["some text here"]).unwrap();
        let norm: f32 = results[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "non-empty text embedding should be normalized");
    }
}
