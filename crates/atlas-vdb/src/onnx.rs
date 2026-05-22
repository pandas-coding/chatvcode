use std::sync::Mutex;

use ort::session::Session;
use ort::value::Value;
use tokenizers::Tokenizer;

use crate::config::{EmbeddingConfig, ExecutionProvider};
use crate::embedding::EmbeddingService;
use crate::error::{VdbContext, VdbError, VdbResult};

/// An embedding service backed by ONNX Runtime.
///
/// Loads an ONNX model and a tokenizer to convert text strings into
/// embedding vectors via mean-pooling over the output token embeddings.
///
/// Uses a [`Mutex`] around the ONNX session to allow sharing across threads.
///
/// Supports GPU acceleration via execution providers configured through
/// [`EmbeddingConfig::with_execution_provider`].
///
/// # Examples
///
/// ```no_run
/// use atlas_vdb::{EmbeddingConfig, OnnxEmbeddingService, EmbeddingService, ExecutionProvider};
///
/// let config = EmbeddingConfig::new("model.onnx", 384, 512)
///     .with_tokenizer_path("tokenizer.json");
///
/// #[cfg(feature = "cuda")]
/// let config = config.with_execution_provider(ExecutionProvider::Cuda);
///
/// let service = OnnxEmbeddingService::new(config).unwrap();
///
/// let vectors = service.embed(&["fn main() {}"]).unwrap();
/// assert_eq!(vectors.len(), 1);
/// assert_eq!(vectors[0].len(), 384);
/// ```
pub struct OnnxEmbeddingService {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    config: EmbeddingConfig,
}

impl OnnxEmbeddingService {
    /// Creates a new ONNX embedding service from the given configuration.
    ///
    /// Validates the config, loads the ONNX model and tokenizer, and
    /// initializes the inference session with the configured execution
    /// provider (CPU by default, GPU if configured).
    ///
    /// # Errors
    ///
    /// Returns [`VdbErrorKind::ModelLoad`] if the ONNX model fails to load,
    /// [`VdbErrorKind::TokenizerLoad`] if the tokenizer file is missing or
    /// fails to load, or [`VdbErrorKind::InvalidInput`] if the config is
    /// invalid (zero dimension/max_tokens).
    pub fn new(config: EmbeddingConfig) -> VdbResult<Self> {
        config.validate()?;

        log::info!(
            "Loading ONNX model from {} (provider: {:?})",
            config.model_path.display(),
            config.execution_provider
        );

        let mut builder = Session::builder().map_err(|e| {
            VdbError::model_load("Failed to create ONNX session builder")
                .with_context(
                    VdbContext::default()
                        .with_path(&config.model_path)
                        .with_operation("model_load"),
                )
                .with_source(e.to_string())
        })?;

        match config.execution_provider {
            ExecutionProvider::Cuda => {
                builder = builder
                    .with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default().build()])
                    .map_err(|e| {
                        VdbError::model_load("Failed to configure CUDA execution provider (GPU may not be available)")
                            .with_context(
                                VdbContext::default()
                                    .with_path(&config.model_path)
                                    .with_operation("model_load"),
                            )
                            .with_source(e.to_string())
                    })?;
                log::info!("CUDA execution provider configured");
            }
            ExecutionProvider::DirectML => {
                builder = builder
                    .with_execution_providers([
                        ort::execution_providers::DirectMLExecutionProvider::default().build(),
                    ])
                    .map_err(|e| {
                        VdbError::model_load("Failed to configure DirectML execution provider")
                            .with_context(
                                VdbContext::default()
                                    .with_path(&config.model_path)
                                    .with_operation("model_load"),
                            )
                            .with_source(e.to_string())
                    })?;
                log::info!("DirectML execution provider configured");
            }
            _ => {}
        }

        let session = builder.commit_from_file(&config.model_path).map_err(|e| {
            VdbError::model_load("Failed to load ONNX model from file")
                .with_context(
                    VdbContext::default()
                        .with_path(&config.model_path)
                        .with_operation("model_load"),
                )
                .with_source(e.to_string())
        })?;

        log::info!("ONNX model loaded successfully, dimension={}", config.dimension);

        let tokenizer_path = config.tokenizer_path.as_ref().ok_or_else(|| {
            VdbError::tokenizer_load("Tokenizer path is required but not provided").with_context(
                VdbContext::default()
                    .with_path(&config.model_path)
                    .with_operation("tokenizer_load"),
            )
        })?;

        log::info!("Loading tokenizer from {}", tokenizer_path.display());

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            VdbError::tokenizer_load("Failed to load tokenizer from file")
                .with_context(
                    VdbContext::default()
                        .with_path(tokenizer_path)
                        .with_operation("tokenizer_load"),
                )
                .with_source(e.to_string())
        })?;

        log::info!("Tokenizer loaded successfully");

        Ok(Self { session: Mutex::new(session), tokenizer, config })
    }

    fn tokenize(&self, text: &str) -> VdbResult<Vec<i64>> {
        let encoding = self.tokenizer.encode(text, true).map_err(|e| {
            VdbError::inference("Failed to tokenize text")
                .with_context(
                    VdbContext::default()
                        .with_path(&self.config.model_path)
                        .with_operation("tokenize"),
                )
                .with_source(e.to_string())
        })?;

        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        if ids.len() > self.config.max_tokens {
            log::debug!(
                "Truncating tokens from {} to {} (max_tokens)",
                ids.len(),
                self.config.max_tokens
            );
            ids.truncate(self.config.max_tokens);
        }

        Ok(ids)
    }

    fn create_attention_mask(ids: &[i64]) -> Vec<i64> {
        ids.iter().map(|&id| if id == 0 { 0 } else { 1 }).collect()
    }

    fn infer_single(&self, ids: &[i64]) -> VdbResult<Vec<f32>> {
        let attention_mask = Self::create_attention_mask(ids);
        let seq_len = ids.len();

        let input_ids_value =
            Value::from_array(([1, seq_len], ids.to_vec())).map_err(|e: ort::Error| {
                VdbError::inference("Failed to create input_ids tensor")
                    .with_context(
                        VdbContext::default()
                            .with_path(&self.config.model_path)
                            .with_operation("inference"),
                    )
                    .with_source(e.to_string())
            })?;

        let attention_mask_value =
            Value::from_array(([1, seq_len], attention_mask)).map_err(|e: ort::Error| {
                VdbError::inference("Failed to create attention_mask tensor")
                    .with_context(
                        VdbContext::default()
                            .with_path(&self.config.model_path)
                            .with_operation("inference"),
                    )
                    .with_source(e.to_string())
            })?;

        let mut session = self.session.lock().map_err(|e| {
            VdbError::inference("Failed to lock ONNX session")
                .with_context(
                    VdbContext::default()
                        .with_path(&self.config.model_path)
                        .with_operation("inference"),
                )
                .with_source(e.to_string())
        })?;
        let outputs = session
            .run(ort::inputs![input_ids_value, attention_mask_value])
            .map_err(|e: ort::Error| {
                VdbError::inference("ONNX inference failed")
                    .with_context(
                        VdbContext::default()
                            .with_path(&self.config.model_path)
                            .with_operation("inference"),
                    )
                    .with_source(e.to_string())
            })?;

        let output_tensor = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e: ort::Error| {
                VdbError::inference("Failed to extract output tensor")
                    .with_context(
                        VdbContext::default()
                            .with_path(&self.config.model_path)
                            .with_operation("inference"),
                    )
                    .with_source(e.to_string())
            })?;

        let (output_shape, output_data) = output_tensor;
        let hidden_size = output_shape.iter().last().copied().unwrap_or(0) as usize;

        if hidden_size == 0 {
            return Err(VdbError::inference("Output tensor has zero hidden size").with_context(
                VdbContext::default()
                    .with_path(&self.config.model_path)
                    .with_operation("inference"),
            ));
        }

        let mut embedding = vec![0.0f32; hidden_size];
        let mut count = 0usize;

        for (i, &val) in output_data.iter().enumerate() {
            let idx = i % hidden_size;
            embedding[idx] += val;
            if idx == hidden_size - 1 {
                count += 1;
            }
        }

        if count > 0 {
            for val in embedding.iter_mut() {
                *val /= count as f32;
            }
        }

        if embedding.len() != self.config.dimension {
            return Err(VdbError::inference(format!(
                "Output dimension mismatch: expected {}, got {}",
                self.config.dimension,
                embedding.len()
            ))
            .with_context(
                VdbContext::default()
                    .with_path(&self.config.model_path)
                    .with_operation("inference"),
            ));
        }

        Ok(embedding)
    }
}

impl EmbeddingService for OnnxEmbeddingService {
    fn embed(&self, texts: &[&str]) -> VdbResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            if text.is_empty() {
                results.push(vec![0.0f32; self.config.dimension]);
                continue;
            }

            let ids = self.tokenize(text)?;

            if ids.is_empty() {
                results.push(vec![0.0f32; self.config.dimension]);
                continue;
            }

            let embedding = self.infer_single(&ids)?;
            results.push(embedding);
        }

        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }
}
