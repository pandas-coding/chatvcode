use std::sync::Mutex;

use ort::session::Session;
use ort::value::Value;
use tokenizers::Tokenizer;

use crate::config::EmbeddingConfig;
use crate::embedding::EmbeddingService;
use crate::error::{VdbError, VdbResult};

pub struct OnnxEmbeddingService {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    config: EmbeddingConfig,
}

impl OnnxEmbeddingService {
    pub fn new(config: EmbeddingConfig) -> VdbResult<Self> {
        config.validate()?;

        let session = Session::builder()
            .map_err(|e: ort::Error| {
                VdbError::model_load(format!("Failed to create ONNX session builder: {e}"))
                    .with_source(e.to_string())
            })?
            .commit_from_file(&config.model_path)
            .map_err(|e: ort::Error| {
                VdbError::model_load(format!(
                    "Failed to load ONNX model from {}: {e}",
                    config.model_path.display()
                ))
                .with_source(e.to_string())
            })?;

        let tokenizer_path = config.tokenizer_path.as_ref().ok_or_else(|| {
            VdbError::tokenizer_load("Tokenizer path is required but not provided")
        })?;

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            VdbError::tokenizer_load(format!(
                "Failed to load tokenizer from {}: {e}",
                tokenizer_path.display()
            ))
            .with_source(e.to_string())
        })?;

        Ok(Self { session: Mutex::new(session), tokenizer, config })
    }

    fn tokenize(&self, text: &str) -> VdbResult<Vec<i64>> {
        let encoding = self.tokenizer.encode(text, true).map_err(|e| {
            VdbError::inference(format!("Failed to tokenize text: {e}")).with_source(e.to_string())
        })?;

        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        if ids.len() > self.config.max_tokens {
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
                VdbError::inference(format!("Failed to create input_ids tensor: {e}"))
                    .with_source(e.to_string())
            })?;

        let attention_mask_value =
            Value::from_array(([1, seq_len], attention_mask)).map_err(|e: ort::Error| {
                VdbError::inference(format!("Failed to create attention_mask tensor: {e}"))
                    .with_source(e.to_string())
            })?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| VdbError::inference(format!("Failed to lock ONNX session: {e}")))?;
        let outputs = session
            .run(ort::inputs![input_ids_value, attention_mask_value])
            .map_err(|e: ort::Error| {
                VdbError::inference(format!("ONNX inference failed: {e}"))
                    .with_source(e.to_string())
            })?;

        let output_tensor = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e: ort::Error| {
                VdbError::inference(format!("Failed to extract output tensor: {e}"))
                    .with_source(e.to_string())
            })?;

        let (output_shape, output_data) = output_tensor;
        let hidden_size = output_shape.iter().last().copied().unwrap_or(0) as usize;

        if hidden_size == 0 {
            return Err(VdbError::inference("Output tensor has zero hidden size"));
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
            )));
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
