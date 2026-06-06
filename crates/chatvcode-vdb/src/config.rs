use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{VdbContext, VdbError, VdbResult};

/// Supported execution providers for ONNX Runtime inference.
///
/// Execution providers enable hardware acceleration (GPU, NPU, etc.)
/// for model inference, significantly improving embedding throughput.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionProvider {
    /// CPU-only inference (default, always available).
    Cpu,
    /// NVIDIA CUDA GPU acceleration.
    Cuda,
    /// `DirectML` acceleration (Windows GPU via DX12).
    DirectML,
    /// `CoreML` acceleration (Apple Silicon / macOS).
    CoreML,
    /// `ROCm` acceleration (AMD GPU).
    Rocm,
    /// `OpenVINO` acceleration (Intel).
    OpenVino,
}

/// Configuration for an embedding model.
///
/// Specifies the ONNX model file, optional tokenizer, expected output
/// dimension, and maximum token length for input text.
///
/// # Examples
///
/// ```no_run
/// use chatvcode_vdb::EmbeddingConfig;
///
/// let config = EmbeddingConfig::new("model.onnx", 384, 512)
///     .with_tokenizer_path("tokenizer.json");
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Path to the ONNX model file.
    pub model_path: PathBuf,
    /// Optional path to the tokenizer file (e.g., `tokenizer.json`).
    /// Required when constructing an [`OnnxEmbeddingService`].
    pub tokenizer_path: Option<PathBuf>,
    /// Expected output dimension of the embedding vectors.
    pub dimension: usize,
    /// Maximum number of tokens for input text. Longer inputs are truncated.
    pub max_tokens: usize,
    /// Execution provider for hardware acceleration.
    /// Default is [`ExecutionProvider::Cpu`].
    pub execution_provider: ExecutionProvider,
}

impl EmbeddingConfig {
    /// Creates a new configuration with the given model path, dimension, and max tokens.
    pub fn new(model_path: impl Into<PathBuf>, dimension: usize, max_tokens: usize) -> Self {
        Self {
            model_path: model_path.into(),
            tokenizer_path: None,
            dimension,
            max_tokens,
            execution_provider: ExecutionProvider::Cpu,
        }
    }

    /// Sets the tokenizer file path.
    ///
    /// The tokenizer is required for ONNX-based embedding via [`OnnxEmbeddingService`].
    pub fn with_tokenizer_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.tokenizer_path = Some(path.into());
        self
    }

    /// Sets the execution provider for hardware acceleration.
    ///
    /// Use [`ExecutionProvider::Cuda`] for NVIDIA GPU inference,
    /// [`ExecutionProvider::DirectML`] for Windows GPU, or
    /// [`ExecutionProvider::CoreML`] for Apple Silicon.
    ///
    /// Default is [`ExecutionProvider::Cpu`].
    #[must_use]
    pub const fn with_execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.execution_provider = provider;
        self
    }

    /// Validates the configuration.
    ///
    /// Checks that the model file exists, dimension and `max_tokens` are non-zero,
    /// and the tokenizer file exists if specified.
    ///
    /// # Errors
    ///
    /// Returns [`VdbErrorKind::ModelLoad`] if the model file is missing,
    /// [`VdbErrorKind::InvalidInput`] if dimension or `max_tokens` is zero,
    /// or [`VdbErrorKind::TokenizerLoad`] if the tokenizer file is missing.
    pub fn validate(&self) -> VdbResult<()> {
        if !self.model_path.exists() {
            return Err(VdbError::model_load("Model file not found").with_context(
                VdbContext::default()
                    .with_path(&self.model_path)
                    .with_operation("validate"),
            ));
        }

        if self.dimension == 0 {
            return Err(VdbError::invalid_input("Dimension must be greater than 0").with_context(
                VdbContext::default()
                    .with_path(&self.model_path)
                    .with_operation("validate"),
            ));
        }

        if self.max_tokens == 0 {
            return Err(VdbError::invalid_input("Max tokens must be greater than 0").with_context(
                VdbContext::default()
                    .with_path(&self.model_path)
                    .with_operation("validate"),
            ));
        }

        if let Some(ref tokenizer_path) = self.tokenizer_path
            && !tokenizer_path.exists()
        {
            return Err(VdbError::tokenizer_load("Tokenizer file not found").with_context(
                VdbContext::default()
                    .with_path(tokenizer_path)
                    .with_operation("validate"),
            ));
        }

        Ok(())
    }
}
