use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{VdbError, VdbResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model_path: PathBuf,
    pub tokenizer_path: Option<PathBuf>,
    pub dimension: usize,
    pub max_tokens: usize,
}

impl EmbeddingConfig {
    pub fn new(model_path: impl Into<PathBuf>, dimension: usize, max_tokens: usize) -> Self {
        Self { model_path: model_path.into(), tokenizer_path: None, dimension, max_tokens }
    }

    pub fn with_tokenizer_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.tokenizer_path = Some(path.into());
        self
    }

    pub fn validate(&self) -> VdbResult<()> {
        if !self.model_path.exists() {
            return Err(VdbError::model_load(format!(
                "Model file not found: {}",
                self.model_path.display()
            )));
        }

        if self.dimension == 0 {
            return Err(VdbError::invalid_input("Dimension must be greater than 0"));
        }

        if self.max_tokens == 0 {
            return Err(VdbError::invalid_input("Max tokens must be greater than 0"));
        }

        if let Some(ref tokenizer_path) = self.tokenizer_path
            && !tokenizer_path.exists()
        {
            return Err(VdbError::tokenizer_load(format!(
                "Tokenizer file not found: {}",
                tokenizer_path.display()
            )));
        }

        Ok(())
    }
}
