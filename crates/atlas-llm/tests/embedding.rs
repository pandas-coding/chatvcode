//! Integration tests for LlamaEmbeddingService (GGUF-based embeddings).
//!
//! These tests require a GGUF model file in ~/.codeatlas/models/
//! Run with: cargo test -p atlas-llm --test embedding -- --ignored

use atlas_llm::{LlamaEmbeddingService, LlamaModel};
use std::path::PathBuf;
use std::sync::Arc;

fn get_test_model_path() -> Option<PathBuf> {
    let dir = dirs::home_dir()?.join(".codeatlas").join("models");
    if !dir.exists() {
        return None;
    }
    // Find first .gguf file
    std::fs::read_dir(&dir)
        .ok()?
        .filter_map(std::result::Result::ok)
        .find(|e| e.path().extension().is_some_and(|ext| ext == "gguf"))
        .map(|e| e.path())
}

/// Test that LlamaEmbeddingService can load a model and produce embeddings.
///
/// Run with: cargo test -p atlas-llm --test embedding -- --ignored test_embedding_service_basic
#[test]
#[ignore]
fn test_embedding_service_basic() {
    let model_path = match get_test_model_path() {
        Some(p) => p,
        None => {
            eprintln!("No GGUF model found in ~/.codeatlas/models/, skipping test");
            return;
        }
    };

    eprintln!("Loading model from: {}", model_path.display());

    let model = LlamaModel::load(&model_path, 0, true, false)
        .expect("Failed to load GGUF model");
    let model = Arc::new(model);

    let service =
        LlamaEmbeddingService::new(model, 512, 4).expect("Failed to create embedding service");

    // Verify dimension is non-zero
    let dim = service.dimension();
    assert!(dim > 0, "Embedding dimension should be positive, got {dim}");
    eprintln!("Embedding dimension: {dim}");

    // Embed a single text
    let vectors = service
        .embed(&["Hello, world!"])
        .expect("Failed to embed text");
    assert_eq!(vectors.len(), 1, "Should return one embedding vector");
    assert_eq!(vectors[0].len(), dim, "Vector dimension should match service dimension");

    // Verify L2 normalization (norm should be ~1.0)
    let norm: f32 = vectors[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "L2 norm should be ~1.0, got {norm}"
    );
    eprintln!("First embedding norm: {norm:.6}");

    // Embed multiple texts
    let texts = ["Rust programming language", "Python programming language"];
    let vectors = service
        .embed(&texts)
        .expect("Failed to embed multiple texts");
    assert_eq!(vectors.len(), 2, "Should return two embedding vectors");
    assert_eq!(vectors[0].len(), dim);
    assert_eq!(vectors[1].len(), dim);

    // Both vectors should be L2-normalized
    for (i, vec) in vectors.iter().enumerate() {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Vector {i} L2 norm should be ~1.0, got {norm}"
        );
    }

    // Similar texts should have higher cosine similarity than unrelated ones
    let dot_similar: f32 = vectors[0]
        .iter()
        .zip(vectors[1].iter())
        .map(|(a, b)| a * b)
        .sum();

    let vectors_unrelated = service
        .embed(&["quantum physics equation"])
        .expect("Failed to embed unrelated text");
    let dot_unrelated: f32 = vectors[0]
        .iter()
        .zip(vectors_unrelated[0].iter())
        .map(|(a, b)| a * b)
        .sum();

    eprintln!(
        "Cosine similarity (related): {dot_similar:.4}, (unrelated): {dot_unrelated:.4}"
    );
    assert!(
        dot_similar > dot_unrelated,
        "Similar texts should have higher cosine similarity than unrelated texts. \
         Related: {dot_similar:.4}, Unrelated: {dot_unrelated:.4}"
    );
}

/// Test that LlamaEmbeddingService::from_path works.
///
/// Run with: cargo test -p atlas-llm --test embedding -- --ignored test_embedding_from_path
#[test]
#[ignore]
fn test_embedding_from_path() {
    let model_path = match get_test_model_path() {
        Some(p) => p,
        None => {
            eprintln!("No GGUF model found in ~/.codeatlas/models/, skipping test");
            return;
        }
    };

    let service =
        LlamaEmbeddingService::from_path(&model_path, 512, 4, 0).expect("Failed to create service");

    let dim = service.dimension();
    assert!(dim > 0, "Dimension should be positive, got {dim}");
    eprintln!("Embedding dimension from path: {dim}");
}

/// Test that embedding dimension matches the model's n_embd.
///
/// Run with: cargo test -p atlas-llm --test embedding -- --ignored test_embedding_dimension_matches_model
#[test]
#[ignore]
fn test_embedding_dimension_matches_model() {
    let model_path = match get_test_model_path() {
        Some(p) => p,
        None => {
            eprintln!("No GGUF model found in ~/.codeatlas/models/, skipping test");
            return;
        }
    };

    let model = LlamaModel::load(&model_path, 0, true, false).expect("Failed to load model");
    let model_dim = model.n_embd() as usize;
    eprintln!("Model n_embd: {model_dim}");

    let service = LlamaEmbeddingService::new(Arc::new(model), 512, 4)
        .expect("Failed to create service");

    assert_eq!(
        service.dimension(),
        model_dim,
        "Service dimension should match model n_embd"
    );
}