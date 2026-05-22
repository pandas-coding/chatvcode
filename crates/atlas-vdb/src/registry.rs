use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::config::EmbeddingConfig;
use crate::embedding::EmbeddingService;
use crate::error::VdbResult;
use crate::onnx::OnnxEmbeddingService;

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub description: String,
    pub dimension: usize,
    pub max_tokens: usize,
    pub model_file: String,
    pub tokenizer_file: Option<String>,
}

impl ModelInfo {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        dimension: usize,
        max_tokens: usize,
        model_file: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            dimension,
            max_tokens,
            model_file: model_file.into(),
            tokenizer_file: None,
        }
    }

    pub fn with_tokenizer(mut self, tokenizer_file: impl Into<String>) -> Self {
        self.tokenizer_file = Some(tokenizer_file.into());
        self
    }
}

pub fn builtin_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo::new(
            "all-MiniLM-L6-v2",
            "Sentence Transformers all-MiniLM-L6-v2 (384-d)",
            384,
            256,
            "model.onnx",
        )
        .with_tokenizer("tokenizer.json"),
        ModelInfo::new(
            "all-mpnet-base-v2",
            "Sentence Transformers all-mpnet-base-v2 (768-d)",
            768,
            384,
            "model.onnx",
        )
        .with_tokenizer("tokenizer.json"),
        ModelInfo::new("e5-small-v2", "Intfloat e5-small-v2 (384-d)", 384, 512, "model.onnx")
            .with_tokenizer("tokenizer.json"),
        ModelInfo::new("bge-small-en", "BAAI bge-small-en-v1.5 (384-d)", 384, 512, "model.onnx")
            .with_tokenizer("tokenizer.json"),
        ModelInfo::new("gte-small", "thenlper gte-small (384-d)", 384, 512, "model.onnx")
            .with_tokenizer("tokenizer.json"),
    ]
}

pub fn builtin_model_names() -> Vec<&'static str> {
    vec!["all-MiniLM-L6-v2", "all-mpnet-base-v2", "e5-small-v2", "bge-small-en", "gte-small"]
}

pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
    loaded_services: HashMap<String, Arc<dyn EmbeddingService>>,
    models_dir: PathBuf,
}

impl ModelRegistry {
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        let dir = models_dir.into();
        let mut models = HashMap::new();
        for model in builtin_models() {
            models.insert(model.name.clone(), model);
        }
        Self { models, loaded_services: HashMap::new(), models_dir: dir }
    }

    pub fn register(&mut self, info: ModelInfo) {
        self.models.insert(info.name.clone(), info);
    }

    pub fn list_models(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }

    pub fn get_model_info(&self, name: &str) -> Option<&ModelInfo> {
        self.models.get(name)
    }

    pub fn build_config(&self, name: &str) -> Option<EmbeddingConfig> {
        let info = self.models.get(name)?;
        let model_dir = self.models_dir.join(&info.name);
        let model_path = model_dir.join(&info.model_file);
        let tokenizer_path = info.tokenizer_file.as_ref().map(|f| model_dir.join(f));

        let mut config = EmbeddingConfig::new(model_path, info.dimension, info.max_tokens);
        if let Some(tp) = tokenizer_path {
            config = config.with_tokenizer_path(tp);
        }
        Some(config)
    }

    pub fn load_service(&mut self, name: &str) -> VdbResult<Arc<dyn EmbeddingService>> {
        if let Some(service) = self.loaded_services.get(name) {
            return Ok(Arc::clone(service));
        }

        let config = self.build_config(name).ok_or_else(|| {
            crate::VdbError::model_load(format!("Model '{}' not found in registry", name))
        })?;

        let service: Arc<dyn EmbeddingService> = Arc::new(OnnxEmbeddingService::new(config)?);
        self.loaded_services
            .insert(name.to_string(), Arc::clone(&service));
        Ok(service)
    }

    pub fn loaded_models(&self) -> Vec<&str> {
        self.loaded_services.keys().map(|s| s.as_str()).collect()
    }

    pub fn models_dir(&self) -> &PathBuf {
        &self.models_dir
    }
}
