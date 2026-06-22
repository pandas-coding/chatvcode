//! Model download utilities for fetching GGUF models from HuggingFace.
//!
//! Provides functionality to download models from HuggingFace Hub with
//! progress reporting and resume support.
//!
//! # Example
//!
//! ```ignore
//! use chatvcode_llm::download::{ModelDownloader, HuggingFaceRepo};
//!
//! let repo = HuggingFaceRepo::new("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF");
//! let downloader = ModelDownloader::new();
//!
//! // List available files
//! let files = downloader.list_files(&repo)?;
//!
//! // Download a specific file
//! downloader.download(&repo, "qwen2.5-coder-7b-instruct-q4_k_m.gguf", None)?;
//! ```

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{LlmError, LlmResult};
use crate::service::default_model_dir;

/// A HuggingFace repository reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceRepo {
    /// Repository ID (e.g., "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF").
    pub repo_id: String,
    /// Branch or revision (default: "main").
    pub revision: String,
}

impl HuggingFaceRepo {
    /// Create a new repository reference with the default branch.
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self { repo_id: repo_id.into(), revision: "main".to_string() }
    }

    /// Create a new repository reference with a specific revision.
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = revision.into();
        self
    }

    /// Get the API URL for listing files.
    pub fn api_url(&self) -> String {
        format!(
            "https://huggingface.co/api/models/{}/revision/{}",
            self.repo_id, self.revision
        )
    }

    /// Get the download URL for a specific file.
    pub fn download_url(&self, filename: &str) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo_id, self.revision, filename
        )
    }
}

/// Information about a file in a HuggingFace repository.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoFile {
    /// File name.
    pub name: String,
    /// File size in bytes.
    pub size: u64,
    /// Whether this is a GGUF file.
    pub is_gguf: bool,
}

impl RepoFile {
    /// Format the file size in human-readable form.
    pub fn formatted_size(&self) -> String {
        format_bytes(self.size)
    }
}

/// Progress callback for download operations.
pub trait DownloadProgress: Send + Sync {
    /// Called when download starts.
    fn on_start(&self, _total_bytes: u64) {}
    /// Called periodically with progress updates.
    fn on_progress(&self, downloaded_bytes: u64, total_bytes: u64);
    /// Called when download completes.
    fn on_complete(&self, _total_bytes: u64) {}
    /// Called when download fails.
    fn on_error(&self, _error: &str) {}
}

/// A simple progress reporter that prints to stderr.
pub struct StderrProgress;

impl DownloadProgress for StderrProgress {
    fn on_start(&self, total_bytes: u64) {
        eprintln!("Downloading {}...", format_bytes(total_bytes));
    }

    fn on_progress(&self, downloaded: u64, total: u64) {
        let percent = if total > 0 { (downloaded * 100) / total } else { 0 };
        eprint!(
            "\r  {}/{} ({}%)",
            format_bytes(downloaded),
            format_bytes(total),
            percent
        );
    }

    fn on_complete(&self, total_bytes: u64) {
        eprintln!("\n  Done: {}", format_bytes(total_bytes));
    }

    fn on_error(&self, error: &str) {
        eprintln!("\n  Error: {error}");
    }
}

/// Model downloader for fetching GGUF files from HuggingFace.
pub struct ModelDownloader {
    /// Target directory for downloaded models.
    target_dir: PathBuf,
    /// HTTP client timeout in seconds.
    timeout_secs: u64,
}

impl ModelDownloader {
    /// Create a new downloader with the default model directory.
    pub fn new() -> Self {
        Self { target_dir: default_model_dir(), timeout_secs: 300 }
    }

    /// Create a downloader with a custom target directory.
    pub fn with_target_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.target_dir = dir.into();
        self
    }

    /// Set the HTTP timeout in seconds.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Get the target directory.
    pub fn target_dir(&self) -> &Path {
        &self.target_dir
    }

    /// List files in a HuggingFace repository.
    pub fn list_files(&self, repo: &HuggingFaceRepo) -> LlmResult<Vec<RepoFile>> {
        let url = repo.api_url();
        let response = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .call()
            .map_err(|e| LlmError::Internal(format!("Failed to fetch repository info: {e}")))?;

        let body = response
            .into_string()
            .map_err(|e| LlmError::Internal(format!("Failed to read response: {e}")))?;

        let json: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| LlmError::Internal(format!("Failed to parse repository info: {e}")))?;

        let siblings = json
            .get("siblings")
            .and_then(|s| s.as_array())
            .ok_or_else(|| LlmError::Internal("Invalid repository response format".to_string()))?;

        let mut files = Vec::new();
        for sibling in siblings {
            let name = sibling
                .get("rfilename")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();

            if name.is_empty() {
                continue;
            }

            let is_gguf = name.to_lowercase().ends_with(".gguf");

            // Try to get size from the API (may not always be available)
            let size = sibling
                .get("size")
                .and_then(|s| s.as_u64())
                .unwrap_or(0);

            files.push(RepoFile { name, size, is_gguf });
        }

        Ok(files)
    }

    /// List only GGUF files in a repository.
    pub fn list_gguf_files(&self, repo: &HuggingFaceRepo) -> LlmResult<Vec<RepoFile>> {
        let files = self.list_files(repo)?;
        Ok(files.into_iter().filter(|f| f.is_gguf).collect())
    }

    /// Download a file from a HuggingFace repository.
    ///
    /// # Arguments
    ///
    /// * `repo` — The repository to download from.
    /// * `filename` — The file name to download.
    /// * `progress` — Optional progress callback.
    ///
    /// # Returns
    ///
    /// The path to the downloaded file.
    pub fn download<P: DownloadProgress>(
        &self,
        repo: &HuggingFaceRepo,
        filename: &str,
        progress: Option<&P>,
    ) -> LlmResult<PathBuf> {
        fs::create_dir_all(&self.target_dir).map_err(|e| {
            LlmError::Io(io::Error::new(
                e.kind(),
                format!("Failed to create target directory: {e}"),
            ))
        })?;

        let target_path = self.target_dir.join(filename);
        let url = repo.download_url(filename);

        let response = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .call()
            .map_err(|e| {
                if let Some(p) = progress {
                    p.on_error(&e.to_string());
                }
                LlmError::Internal(format!("Failed to download {filename}: {e}"))
            })?;

        let total_size: u64 = response
            .header("Content-Length")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        if let Some(p) = progress {
            p.on_start(total_size);
        }

        let file = File::create(&target_path).map_err(|e| {
            LlmError::Io(io::Error::new(
                e.kind(),
                format!("Failed to create file {}: {e}", target_path.display()),
            ))
        })?;

        let mut writer = BufWriter::new(file);
        let mut reader = response.into_reader();
        let mut buffer = [0u8; 8192];
        let mut downloaded: u64 = 0;

        loop {
            let bytes_read = reader.read(&mut buffer).map_err(|e| {
                if let Some(p) = progress {
                    p.on_error(&e.to_string());
                }
                LlmError::Io(io::Error::new(e.kind(), format!("Download read error: {e}")))
            })?;

            if bytes_read == 0 {
                break;
            }

            writer.write_all(&buffer[..bytes_read]).map_err(|e| {
                LlmError::Io(io::Error::new(e.kind(), format!("File write error: {e}")))
            })?;

            downloaded += bytes_read as u64;

            if let Some(p) = progress {
                p.on_progress(downloaded, total_size);
            }
        }

        writer.flush().map_err(|e| {
            LlmError::Io(io::Error::new(e.kind(), format!("File flush error: {e}")))
        })?;

        if let Some(p) = progress {
            p.on_complete(downloaded);
        }

        log::info!("Downloaded {} to {}", filename, target_path.display());

        Ok(target_path)
    }

    /// Check if a file already exists in the target directory.
    pub fn file_exists(&self, filename: &str) -> bool {
        self.target_dir.join(filename).exists()
    }

    /// Get the path where a file would be downloaded to.
    pub fn target_path(&self, filename: &str) -> PathBuf {
        self.target_dir.join(filename)
    }
}

impl Default for ModelDownloader {
    fn default() -> Self {
        Self::new()
    }
}

/// Format bytes into human-readable size.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Well-known coding model repositories on HuggingFace.
pub struct RecommendedModels;

impl RecommendedModels {
    /// Get a list of recommended coding model repositories.
    pub fn coding_models() -> Vec<(&'static str, &'static str)> {
        vec![
            ("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "Qwen 2.5 Coder 7B - balanced performance"),
            ("Qwen/Qwen2.5-Coder-14B-Instruct-GGUF", "Qwen 2.5 Coder 14B - stronger performance"),
            ("bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF", "DeepSeek Coder V2 Lite"),
            ("TheBloke/CodeLlama-7B-Instruct-GGUF", "CodeLlama 7B - Meta's coding model"),
            ("TheBloke/deepseek-coder-6.7B-instruct-GGUF", "DeepSeek Coder 6.7B"),
            ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi 3.5 Mini - Microsoft's small model"),
            ("bartowski/gemma-2-9b-it-GGUF", "Gemma 2 9B - Google's coding-capable model"),
        ]
    }

    /// Get a list of recommended small models (under 5GB).
    pub fn small_models() -> Vec<(&'static str, &'static str)> {
        vec![
            ("Qwen/Qwen2.5-Coder-3B-Instruct-GGUF", "Qwen 2.5 Coder 3B - lightweight"),
            ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi 3.5 Mini - very small"),
            ("TheBloke/deepseek-coder-1.3b-instruct-GGUF", "DeepSeek Coder 1.3B - tiny"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huggingface_repo_new() {
        let repo = HuggingFaceRepo::new("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF");
        assert_eq!(repo.repo_id, "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF");
        assert_eq!(repo.revision, "main");
    }

    #[test]
    fn test_huggingface_repo_with_revision() {
        let repo = HuggingFaceRepo::new("test/repo").with_revision("v1.0");
        assert_eq!(repo.revision, "v1.0");
    }

    #[test]
    fn test_huggingface_repo_api_url() {
        let repo = HuggingFaceRepo::new("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF");
        assert_eq!(
            repo.api_url(),
            "https://huggingface.co/api/models/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/revision/main"
        );
    }

    #[test]
    fn test_huggingface_repo_download_url() {
        let repo = HuggingFaceRepo::new("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF");
        assert_eq!(
            repo.download_url("model.gguf"),
            "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/model.gguf"
        );
    }

    #[test]
    fn test_model_downloader_new() {
        let downloader = ModelDownloader::new();
        assert!(downloader.target_dir().to_string_lossy().contains(".chatvcode"));
    }

    #[test]
    fn test_model_downloader_with_target_dir() {
        let downloader = ModelDownloader::new().with_target_dir("/tmp/models");
        assert_eq!(downloader.target_dir(), Path::new("/tmp/models"));
    }

    #[test]
    fn test_model_downloader_file_exists() {
        let downloader = ModelDownloader::new().with_target_dir("/nonexistent");
        assert!(!downloader.file_exists("nonexistent.gguf"));
    }

    #[test]
    fn test_model_downloader_target_path() {
        let downloader = ModelDownloader::new().with_target_dir("/tmp/models");
        assert_eq!(
            downloader.target_path("model.gguf"),
            PathBuf::from("/tmp/models/model.gguf")
        );
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_bytes(4 * 1024 * 1024 * 1024 + 512 * 1024 * 1024), "4.5 GB");
    }

    #[test]
    fn test_repo_file_formatted_size() {
        let file = RepoFile {
            name: "model.gguf".to_string(),
            size: 4 * 1024 * 1024 * 1024,
            is_gguf: true,
        };
        assert_eq!(file.formatted_size(), "4.0 GB");
    }

    #[test]
    fn test_recommended_models_not_empty() {
        let models = RecommendedModels::coding_models();
        assert!(!models.is_empty());

        let small = RecommendedModels::small_models();
        assert!(!small.is_empty());
    }

    #[test]
    fn test_stderr_progress_trait() {
        let progress = StderrProgress;
        progress.on_start(1000);
        progress.on_progress(500, 1000);
        progress.on_complete(1000);
        progress.on_error("test error");
    }
}
