use std::path::PathBuf;

use crate::error::{VdbContext, VdbError, VdbResult};

pub struct ModelDownloader {
    cache_dir: PathBuf,
}

impl ModelDownloader {
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self { cache_dir: cache_dir.into() }
    }

    #[must_use]
    pub const fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }

    #[must_use]
    pub fn is_model_available(&self, model_id: &str) -> bool {
        let model_dir = self.cache_dir.join(model_id);
        model_dir.exists() && model_dir.join("model.onnx").exists()
    }

    pub fn download_model(&self, repo_id: &str, files: &[&str]) -> VdbResult<PathBuf> {
        let model_dir = self.cache_dir.join(repo_id);
        std::fs::create_dir_all(&model_dir).map_err(|e| {
            VdbError::io("Failed to create model cache directory")
                .with_context(
                    VdbContext::default()
                        .with_path(&model_dir)
                        .with_operation("download_model"),
                )
                .with_source(e.to_string())
        })?;

        let api = hf_hub::api::sync::Api::new().map_err(|e: hf_hub::api::sync::ApiError| {
            VdbError::io("Failed to initialize HuggingFace API")
                .with_context(VdbContext::default().with_operation("download_model"))
                .with_source(e.to_string())
        })?;

        let repo = api.model(repo_id.to_string());

        for &file_name in files {
            let expected_path = model_dir.join(file_name);
            if expected_path.exists() {
                log::info!("Model file already cached: {}", expected_path.display());
                continue;
            }

            log::info!("Downloading {file_name} from {repo_id}/{file_name}");

            let path = repo
                .get(file_name)
                .map_err(|e: hf_hub::api::sync::ApiError| {
                    VdbError::io(format!("Failed to download model file '{file_name}'"))
                        .with_context(VdbContext::default().with_operation("download_model"))
                        .with_source(e.to_string())
                })?;

            std::fs::copy(&path, &expected_path).map_err(|e| {
                VdbError::io(format!(
                    "Failed to copy model file from {} to {}",
                    path.display(),
                    expected_path.display()
                ))
                .with_context(
                    VdbContext::default()
                        .with_path(&expected_path)
                        .with_operation("download_model"),
                )
                .with_source(e.to_string())
            })?;

            log::info!("Downloaded {} to {}", file_name, expected_path.display());
        }

        Ok(model_dir)
    }

    pub fn download_sentence_transformer(&self, model_name: &str) -> VdbResult<PathBuf> {
        let repo_id = format!("sentence-transformers/{model_name}");
        let file_name = format!("models--sentence-transformers--{}", model_name.replace('/', "--"));

        let model_dir = self.cache_dir.join(&file_name);
        std::fs::create_dir_all(&model_dir).map_err(|e| {
            VdbError::io("Failed to create model cache directory")
                .with_context(
                    VdbContext::default()
                        .with_path(&model_dir)
                        .with_operation("download_model"),
                )
                .with_source(e.to_string())
        })?;

        let api = hf_hub::api::sync::Api::new().map_err(|e: hf_hub::api::sync::ApiError| {
            VdbError::io("Failed to initialize HuggingFace API")
                .with_context(VdbContext::default().with_operation("download_model"))
                .with_source(e.to_string())
        })?;

        let repo = api.model(repo_id.clone());

        let files = vec!["model.onnx", "tokenizer.json"];

        for &file_name in &files {
            let expected_path = model_dir.join(file_name);
            if expected_path.exists() {
                log::info!("Model file already cached: {}", expected_path.display());
                continue;
            }

            log::info!("Downloading {file_name} from {repo_id}");

            match repo.get(file_name) {
                Ok(path) => {
                    std::fs::copy(&path, &expected_path).map_err(|e| {
                        VdbError::io(format!(
                            "Failed to copy model file from {} to {}",
                            path.display(),
                            expected_path.display()
                        ))
                        .with_context(
                            VdbContext::default()
                                .with_path(&expected_path)
                                .with_operation("download_model"),
                        )
                        .with_source(e.to_string())
                    })?;
                    log::info!("Downloaded {} to {}", file_name, expected_path.display());
                }
                Err(e) => {
                    log::warn!("File '{file_name}' not found in repo '{repo_id}': {e}");
                }
            }
        }

        if !model_dir.join("model.onnx").exists() {
            return Err(VdbError::model_load(format!(
                "Downloaded model '{model_name}' missing model.onnx file"
            ))
            .with_context(
                VdbContext::default()
                    .with_path(&model_dir)
                    .with_operation("download_model"),
            ));
        }

        Ok(model_dir)
    }
}

impl Default for ModelDownloader {
    fn default() -> Self {
        let cache_dir =
            dirs_home().unwrap_or_else(|| PathBuf::from(".cache").join("chatvcode").join("models"));
        Self::new(cache_dir)
    }
}

fn dirs_home() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".cache").join("chatvcode").join("models"))
}
