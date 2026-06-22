//! Model management: discovery, memory estimation, GPU recommendation, and configuration.
//!
//! This module provides:
//! - Priority-based model discovery (local `.chatvcode/models/` > global `~/.chatvcode/models/`)
//! - Memory usage estimation from GGUF metadata
//! - GPU layer offload recommendations based on available VRAM
//! - Persistent configuration file support (`~/.chatvcode/config.json`)

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{LlmError, LlmResult};
use crate::gguf::{
    GgufMetadata, discover_gguf_models, format_file_size, format_param_count,
    pre_validate_model,
};
use crate::service::default_model_dir;

// ---------------------------------------------------------------------------
// Model discovery
// ---------------------------------------------------------------------------

/// Where a model was discovered from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSource {
    /// Found in the local project's `.chatvcode/models/` directory.
    Local,
    /// Found in the global `~/.chatvcode/models/` directory.
    Global,
    /// Found at a user-specified path.
    Custom(PathBuf),
}

impl std::fmt::Display for ModelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local => write!(f, "local"),
            Self::Global => write!(f, "global"),
            Self::Custom(p) => write!(f, "custom ({})", p.display()),
        }
    }
}

/// A discovered model with its metadata and source location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredModel {
    /// Absolute path to the GGUF file.
    pub path: PathBuf,
    /// Where this model was found.
    pub source: ModelSource,
    /// Human-readable file name.
    pub name: String,
    /// File size in bytes.
    pub file_size: u64,
    /// Model architecture (e.g., "llama", "qwen2").
    pub architecture: Option<String>,
    /// Parameter count.
    pub n_params: Option<u64>,
    /// Training context length.
    pub context_length: Option<i32>,
    /// Quantization / file type.
    pub file_type: Option<String>,
}

impl DiscoveredModel {
    /// Format a human-readable summary line for this model.
    #[must_use]
    pub fn summary_line(&self) -> String {
        let mut parts = vec![self.name.clone()];
        if let Some(ref arch) = self.architecture {
            parts.push(format!("arch={arch}"));
        }
        if let Some(params) = self.n_params {
            parts.push(format!("params={}", format_param_count(params)));
        }
        if let Some(ctx) = self.context_length {
            parts.push(format!("ctx={ctx}"));
        }
        if let Some(ref ft) = self.file_type {
            parts.push(format!("quant={ft}"));
        }
        parts.push(format!("size={}", format_file_size(&self.path)));
        parts.push(format!("source={}", self.source));
        parts.join(", ")
    }
}

/// Returns the local model directory: `<cwd>/.chatvcode/models/`.
#[must_use]
pub fn local_model_dir() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".chatvcode")
        .join("models")
}

/// Returns all model search directories in priority order.
///
/// 1. `<cwd>/.chatvcode/models/` (local project)
/// 2. `~/.chatvcode/models/` (global user)
#[must_use]
pub fn model_search_dirs() -> Vec<(PathBuf, ModelSource)> {
    let mut dirs = Vec::new();

    let local = local_model_dir();
    if local.exists() {
        dirs.push((local, ModelSource::Local));
    }

    let global = default_model_dir();
    if global.exists() {
        dirs.push((global, ModelSource::Global));
    }

    dirs
}

/// Discover all available GGUF models across all search directories.
///
/// Models are returned in priority order:
/// 1. Local project models (`.chatvcode/models/`)
/// 2. Global user models (`~/.chatvcode/models/`)
///
/// Duplicate files (same canonical path) are de-duplicated.
/// Invalid GGUF files are skipped with a warning.
#[must_use]
pub fn list_models() -> Vec<DiscoveredModel> {
    let mut models = Vec::new();
    let mut seen_paths = HashSet::new();

    for (dir, source) in model_search_dirs() {
        let discovered = discover_gguf_models(&dir);
        for (path, _header, meta_result) in discovered {
            // Canonicalize to de-duplicate symlinks / relative paths
            let canon = path.canonicalize().unwrap_or_else(|_| path.clone());
            if !seen_paths.insert(canon) {
                continue;
            }

            let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());

            let (architecture, n_params, context_length, file_type) = match &meta_result {
                Ok(meta) => (
                    meta.architecture.clone(),
                    meta.parameter_count,
                    meta.context_length,
                    meta.file_type.clone(),
                ),
                Err(_) => (None, None, None, None),
            };

            models.push(DiscoveredModel {
                path,
                source: source.clone(),
                name,
                file_size,
                architecture,
                n_params,
                context_length,
                file_type,
            });
        }
    }

    models
}

/// Discover models from a specific directory (for user-specified paths).
pub fn list_models_in_dir(dir: &Path) -> Vec<DiscoveredModel> {
    let mut models = Vec::new();

    if !dir.exists() {
        return models;
    }

    let discovered = discover_gguf_models(dir);
    for (path, _header, meta_result) in discovered {
        let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());

        let (architecture, n_params, context_length, file_type) = match &meta_result {
            Ok(meta) => (
                meta.architecture.clone(),
                meta.parameter_count,
                meta.context_length,
                meta.file_type.clone(),
            ),
            Err(_) => (None, None, None, None),
        };

        models.push(DiscoveredModel {
            path,
            source: ModelSource::Custom(dir.to_path_buf()),
            name,
            file_size,
            architecture,
            n_params,
            context_length,
            file_type,
        });
    }

    models
}

// ---------------------------------------------------------------------------
// Memory estimation
// ---------------------------------------------------------------------------

/// Memory usage estimation for a model at a given context size.
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Model weights size in bytes (approximately the file size).
    pub model_bytes: u64,
    /// Estimated KV cache size in bytes for the given context length.
    pub kv_cache_bytes: u64,
    /// Estimated runtime overhead (activations, scratch buffers, etc.).
    pub overhead_bytes: u64,
    /// Total estimated RAM usage in bytes.
    pub total_bytes: u64,
    /// Whether the model is likely to fit in available system RAM.
    pub fits_in_ram: bool,
    /// Human-readable warning messages (e.g., "model exceeds available RAM").
    pub warnings: Vec<String>,
}

impl MemoryEstimate {
    /// Format a human-readable summary of the memory estimate.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "  Model weights : {}",
            format_bytes(self.model_bytes)
        ));
        lines.push(format!(
            "  KV cache      : {}",
            format_bytes(self.kv_cache_bytes)
        ));
        lines.push(format!(
            "  Overhead      : {}",
            format_bytes(self.overhead_bytes)
        ));
        lines.push(format!(
            "  Total (est.)  : {}",
            format_bytes(self.total_bytes)
        ));
        if self.fits_in_ram {
            lines.push("  Status        : ✓ Expected to fit in available RAM".into());
        } else {
            lines.push("  Status        : ⚠ May exceed available RAM".into());
        }
        for w in &self.warnings {
            lines.push(format!("  Warning       : {w}"));
        }
        lines.join("\n")
    }
}

/// Estimate memory usage for a model file at a given context length.
///
/// Uses GGUF metadata to estimate:
/// - Model weights size (from file size)
/// - KV cache size (from architecture parameters and context length)
/// - Runtime overhead (heuristic)
///
/// # Arguments
///
/// * `model_path` — Path to the GGUF model file.
/// * `n_ctx` — Desired context window size (tokens).
///
/// # Errors
///
/// Returns an error if the model file cannot be read or validated.
pub fn estimate_memory(model_path: &Path, n_ctx: u32) -> LlmResult<MemoryEstimate> {
    let meta = pre_validate_model(model_path)?;
    let file_size = std::fs::metadata(model_path)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(estimate_memory_from_metadata(&meta, file_size, n_ctx))
}

/// Estimate memory from already-parsed GGUF metadata.
///
/// This avoids re-reading the file when metadata is already available.
#[must_use]
pub fn estimate_memory_from_metadata(
    meta: &GgufMetadata,
    file_size: u64,
    n_ctx: u32,
) -> MemoryEstimate {
    let mut warnings = Vec::new();

    // Model weights ≈ file size (GGUF files are memory-mapped)
    let model_bytes = file_size;

    // KV cache estimate:
    //   Per layer: 2 (K + V) * n_head_kv * head_dim * n_ctx * sizeof(f16)
    //   head_dim is typically embedding_length / num_heads
    //   Total: num_layers * per_layer
    let n_embd = meta.embedding_length.unwrap_or(0) as u64;
    let n_layer = meta.num_layers.unwrap_or(0) as u64;
    let n_head_kv = meta.num_kv_heads.unwrap_or(0) as u64;
    let n_head = meta.num_heads.unwrap_or(1).max(1) as u64;

    let head_dim = if n_head > 0 { n_embd / n_head } else { 0 };
    let kv_per_layer = 2 * n_head_kv * head_dim * (n_ctx as u64) * 2; // 2 bytes for f16
    let kv_cache_bytes = n_layer * kv_per_layer;

    // Overhead: activations, scratch buffers, thread stacks, etc.
    // Heuristic: ~10% of model size, minimum 256MB, capped at 2GB
    let overhead_bytes = (model_bytes / 10).clamp(256 * 1024 * 1024, 2 * 1024 * 1024 * 1024);

    let total_bytes = model_bytes + kv_cache_bytes + overhead_bytes;

    // Check against available system RAM
    let available_ram = get_available_ram();
    let fits_in_ram = if let Some(ram) = available_ram {
        if total_bytes > ram {
            warnings.push(format!(
                "Estimated usage ({}) exceeds available RAM ({})",
                format_bytes(total_bytes),
                format_bytes(ram)
            ));
        }
        total_bytes <= ram
    } else {
        // Can't determine RAM; assume it fits but note the uncertainty
        true
    };

    // Additional warnings
    if total_bytes > 16 * 1024 * 1024 * 1024 {
        warnings.push("Model may require more than 16 GB of memory".into());
    }
    if n_ctx as i32 > meta.context_length.unwrap_or(i32::MAX) {
        warnings.push(format!(
            "Requested context ({n_ctx}) exceeds model training context ({})",
            meta.context_length.unwrap_or(0)
        ));
    }

    MemoryEstimate {
        model_bytes,
        kv_cache_bytes,
        overhead_bytes,
        total_bytes,
        fits_in_ram,
        warnings,
    }
}

// ---------------------------------------------------------------------------
// GPU layer recommendation
// ---------------------------------------------------------------------------

/// Recommendation for GPU layer offloading.
#[derive(Debug, Clone)]
pub struct GpuRecommendation {
    /// Recommended number of GPU layers to offload (-1 = all).
    pub recommended_layers: i32,
    /// Description of the recommendation.
    pub description: String,
    /// Total number of transformer layers in the model.
    pub total_layers: i32,
    /// Estimated VRAM usage for the recommended layer count.
    pub estimated_vram_bytes: u64,
    /// Whether GPU offload is supported in the current build.
    pub gpu_available: bool,
}

impl GpuRecommendation {
    /// Format a human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        if !self.gpu_available {
            lines.push("GPU offload: not available (CPU-only build)".into());
            lines.push(format!(
                "  Recommendation: use --n-gpu-layers 0 (all layers on CPU)"
            ));
            return lines.join("\n");
        }

        lines.push(format!("  Total layers     : {}", self.total_layers));
        lines.push(format!(
            "  Recommended      : {} layers",
            if self.recommended_layers < 0 {
                "all (-1)".to_string()
            } else {
                self.recommended_layers.to_string()
            }
        ));
        lines.push(format!(
            "  Est. VRAM usage  : {}",
            format_bytes(self.estimated_vram_bytes)
        ));
        lines.push(format!("  Note             : {}", self.description));
        lines.join("\n")
    }
}

/// Recommend GPU layer offloading based on model metadata and available VRAM.
///
/// # Arguments
///
/// * `model_path` — Path to the GGUF model file.
/// * `available_vram_bytes` — Available GPU VRAM in bytes. If `None`, a heuristic is used.
///
/// # Errors
///
/// Returns an error if the model file cannot be validated.
pub fn recommend_gpu_layers(
    model_path: &Path,
    available_vram_bytes: Option<u64>,
) -> LlmResult<GpuRecommendation> {
    let meta = pre_validate_model(model_path)?;
    let file_size = std::fs::metadata(model_path)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(recommend_gpu_layers_from_metadata(&meta, file_size, available_vram_bytes))
}

/// Recommend GPU layers from already-parsed metadata.
#[must_use]
pub fn recommend_gpu_layers_from_metadata(
    meta: &GgufMetadata,
    file_size: u64,
    available_vram_bytes: Option<u64>,
) -> GpuRecommendation {
    let gpu_available = crate::supports_gpu_offload();
    let total_layers = meta.num_layers.unwrap_or(0);

    if !gpu_available || total_layers == 0 {
        return GpuRecommendation {
            recommended_layers: 0,
            description: "GPU offload not available or model has no layers".into(),
            total_layers,
            estimated_vram_bytes: 0,
            gpu_available,
        };
    }

    // Estimate VRAM per layer:
    //   weight_bytes_per_layer ≈ file_size / total_layers
    //   KV cache per layer depends on context size (we use a default 8192)
    let default_ctx: u64 = 8192;
    let n_embd = meta.embedding_length.unwrap_or(0) as u64;
    let n_head_kv = meta.num_kv_heads.unwrap_or(0) as u64;
    let n_head = meta.num_heads.unwrap_or(1).max(1) as u64;
    let head_dim = if n_head > 0 { n_embd / n_head } else { 0 };

    let weight_per_layer = if total_layers > 0 {
        file_size / total_layers as u64
    } else {
        0
    };
    let kv_per_layer = 2 * n_head_kv * head_dim * default_ctx * 2; // f16
    let vram_per_layer = weight_per_layer + kv_per_layer;

    // Overhead: model metadata, compute buffers, etc. (~512MB heuristic)
    let gpu_overhead: u64 = 512 * 1024 * 1024;

    let vram = available_vram_bytes.unwrap_or_else(|| estimate_default_vram());

    let usable_vram = vram.saturating_sub(gpu_overhead);
    let max_layers_by_vram = if vram_per_layer > 0 {
        (usable_vram / vram_per_layer) as i32
    } else {
        total_layers
    };

    let recommended = if max_layers_by_vram >= total_layers {
        -1 // all layers fit
    } else {
        // Leave a small margin (10%) for safety
        let layers = (max_layers_by_vram as f64 * 0.9) as i32;
        layers.max(0)
    };
    let est_vram = if recommended < 0 {
        vram_per_layer * total_layers as u64 + gpu_overhead
    } else {
        vram_per_layer * recommended as u64 + gpu_overhead
    };

    let description = if recommended < 0 {
        format!(
            "All {total_layers} layers fit in available VRAM ({})",
            format_bytes(vram)
        )
    } else if recommended == 0 {
        format!(
            "Model too large for available VRAM ({}); use CPU-only mode",
            format_bytes(vram)
        )
    } else {
        format!(
            "{recommended}/{total_layers} layers fit in available VRAM ({})",
            format_bytes(vram)
        )
    };

    GpuRecommendation {
        recommended_layers: recommended,
        description,
        total_layers,
        estimated_vram_bytes: est_vram,
        gpu_available,
    }
}

// ---------------------------------------------------------------------------
// Configuration file
// ---------------------------------------------------------------------------

/// Persistent configuration for `chatvcode`.
///
/// Stored as JSON at `~/.chatvcode/config.json`.
/// Settings priority: CLI arguments > config file > built-in defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatvcodeConfig {
    /// Model-related settings.
    #[serde(default)]
    pub model: ModelConfig,

    /// Generation parameters.
    #[serde(default)]
    pub generation: GenerationConfig,

    /// Chat / RAG settings.
    #[serde(default)]
    pub chat: ChatConfig,
}

/// Model-related configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Default model path (or name to auto-discover).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,

    /// Number of GPU layers to offload (-1 = all, 0 = CPU only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_gpu_layers: Option<i32>,

    /// Context window size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_ctx: Option<u32>,

    /// Number of inference threads (0 = auto).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_threads: Option<i32>,

    /// Chat template override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,

    /// Use memory-mapped I/O.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_mmap: Option<bool>,

    /// Verbose llama.cpp logging.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verbose_log: Option<bool>,
}

/// Generation parameter configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p (nucleus) sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-k sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,

    /// Maximum tokens to generate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
}

/// Chat and RAG configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatConfig {
    /// Default system prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,

    /// Enable streaming output by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Enable RAG retrieval by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval: Option<bool>,

    /// Number of context snippets to retrieve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k_retrieval: Option<usize>,

    /// Context token budget (0 = unlimited).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_token_budget: Option<usize>,
}

/// Global configuration file path: `~/.chatvcode/config.json`.
#[must_use]
pub fn default_config_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".chatvcode")
        .join("config.json")
}

/// Local (project-level) configuration file path: `<cwd>/.chatvcode/config.json`.
///
/// This config takes priority over the global config when both exist.
#[must_use]
pub fn local_config_path() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".chatvcode")
        .join("config.json")
}

impl ChatvcodeConfig {
    /// Load configuration with priority: local > global > built-in default.
    ///
    /// Search order:
    /// 1. `<cwd>/.chatvcode/config.json` (local/project-level)
    /// 2. `~/.chatvcode/config.json` (global/user-level)
    /// 3. Built-in defaults
    ///
    /// When both local and global configs exist, they are merged with
    /// local values taking priority over global values for each field.
    ///
    /// Returns `Ok(default)` if no config files exist.
    /// Returns an error if a config file exists but cannot be parsed.
    pub fn load_default() -> LlmResult<Self> {
        let global_path = default_config_path();
        let local_path = local_config_path();

        let global_exists = global_path.exists();
        let local_exists = local_path.exists();

        // De-duplicate: if local and global resolve to the same file,
        // only load once (treat as global).
        let same_file = global_exists
            && local_exists
            && global_path.canonicalize().ok() == local_path.canonicalize().ok();

        match (global_exists, local_exists) {
            (false, false) => {
                log::info!("No configuration files found, using built-in defaults");
                Ok(Self::default())
            }
            (true, false) => {
                log::info!("Loading global configuration from {}", global_path.display());
                Self::load_from(&global_path)
            }
            (false, true) => {
                log::info!("Loading local configuration from {}", local_path.display());
                Self::load_from(&local_path)
            }
            (true, true) if same_file => {
                log::info!(
                    "Local and global config resolve to the same file: {}",
                    global_path.display()
                );
                Self::load_from(&global_path)
            }
            (true, true) => {
                // Both exist and are different files — merge with local priority
                log::info!(
                    "Merging configurations: local ({}) > global ({})",
                    local_path.display(),
                    global_path.display()
                );
                let global = Self::load_from(&global_path)?;
                let local = Self::load_from(&local_path)?;
                Ok(local.merge_over(global))
            }
        }
    }

    /// Merge this config over a base config.
    ///
    /// For each field, `self`'s value takes priority when it is `Some`.
    /// If `self`'s value is `None`, the base value is used.
    ///
    /// This implements the merge semantics for:
    /// `local config > global config > built-in defaults`.
    #[must_use]
    pub fn merge_over(self, base: Self) -> Self {
        Self {
            model: ModelConfig {
                path: self.model.path.or(base.model.path),
                n_gpu_layers: self.model.n_gpu_layers.or(base.model.n_gpu_layers),
                n_ctx: self.model.n_ctx.or(base.model.n_ctx),
                n_threads: self.model.n_threads.or(base.model.n_threads),
                template: self.model.template.or(base.model.template),
                use_mmap: self.model.use_mmap.or(base.model.use_mmap),
                verbose_log: self.model.verbose_log.or(base.model.verbose_log),
            },
            generation: GenerationConfig {
                temperature: self.generation.temperature.or(base.generation.temperature),
                top_p: self.generation.top_p.or(base.generation.top_p),
                top_k: self.generation.top_k.or(base.generation.top_k),
                max_tokens: self.generation.max_tokens.or(base.generation.max_tokens),
            },
            chat: ChatConfig {
                system_prompt: self.chat.system_prompt.or(base.chat.system_prompt),
                stream: self.chat.stream.or(base.chat.stream),
                retrieval: self.chat.retrieval.or(base.chat.retrieval),
                top_k_retrieval: self.chat.top_k_retrieval.or(base.chat.top_k_retrieval),
                context_token_budget: self
                    .chat
                    .context_token_budget
                    .or(base.chat.context_token_budget),
            },
        }
    }

    /// Load configuration from a specific path.
    pub fn load_from(path: &Path) -> LlmResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            LlmError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to read config file '{}': {e}", path.display()),
            ))
        })?;

        let config: Self = serde_json::from_str(&content).map_err(|e| {
            LlmError::InvalidParameter(format!(
                "Failed to parse config file '{}': {e}",
                path.display()
            ))
        })?;

        log::info!("Loaded configuration from {}", path.display());
        Ok(config)
    }

    /// Save configuration to the default path.
    pub fn save_default(&self) -> LlmResult<()> {
        let path = default_config_path();
        self.save_to(&path)
    }

    /// Save configuration to a specific path.
    pub fn save_to(&self, path: &Path) -> LlmResult<()> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                LlmError::Io(std::io::Error::new(
                    e.kind(),
                    format!(
                        "Failed to create config directory '{}': {e}",
                        parent.display()
                    ),
                ))
            })?;
        }

        let json = serde_json::to_string_pretty(self).map_err(|e| {
            LlmError::Internal(format!("Failed to serialize config: {e}"))
        })?;

        std::fs::write(path, json).map_err(|e| {
            LlmError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to write config file '{}': {e}", path.display()),
            ))
        })?;

        log::info!("Saved configuration to {}", path.display());
        Ok(())
    }

    /// Apply config file values as defaults for any `None` CLI arguments.
    ///
    /// This implements the priority: CLI > config > built-in default.
    /// Each CLI argument is passed as `Option<T>`: if `Some`, the CLI value
    /// is used; if `None`, the config value (or built-in default) is used.
    #[must_use]
    pub fn resolve_model_path(&self, cli_value: Option<&str>) -> Option<String> {
        cli_value
            .map(String::from)
            .or_else(|| self.model.path.clone())
    }

    /// Resolve `n_gpu_layers` with priority: CLI > config > default (0).
    #[must_use]
    pub fn resolve_n_gpu_layers(&self, cli_value: Option<i32>) -> i32 {
        cli_value
            .or(self.model.n_gpu_layers)
            .unwrap_or(0)
    }

    /// Resolve `n_ctx` with priority: CLI > config > default (8192).
    #[must_use]
    pub fn resolve_n_ctx(&self, cli_value: Option<u32>) -> u32 {
        cli_value
            .or(self.model.n_ctx)
            .unwrap_or(8192)
    }

    /// Resolve `n_threads` with priority: CLI > config > default (auto).
    #[must_use]
    pub fn resolve_n_threads(&self, cli_value: Option<i32>) -> i32 {
        cli_value
            .or(self.model.n_threads)
            .unwrap_or_else(|| num_cpus::get() as i32)
    }

    /// Resolve `template` with priority: CLI > config > default ("auto").
    #[must_use]
    pub fn resolve_template(&self, cli_value: Option<&str>) -> String {
        cli_value
            .map(String::from)
            .or_else(|| self.model.template.clone())
            .unwrap_or_else(|| "auto".to_string())
    }

    /// Resolve `temperature` with priority: CLI > config > default (0.7).
    #[must_use]
    pub fn resolve_temperature(&self, cli_value: Option<f32>) -> f32 {
        cli_value
            .or(self.generation.temperature)
            .unwrap_or(0.7)
    }

    /// Resolve `top_p` with priority: CLI > config > default (0.9).
    #[must_use]
    pub fn resolve_top_p(&self, cli_value: Option<f32>) -> f32 {
        cli_value
            .or(self.generation.top_p)
            .unwrap_or(0.9)
    }

    /// Resolve `top_k` with priority: CLI > config > default (40).
    #[must_use]
    pub fn resolve_top_k(&self, cli_value: Option<i32>) -> i32 {
        cli_value
            .or(self.generation.top_k)
            .unwrap_or(40)
    }

    /// Resolve `max_tokens` with priority: CLI > config > default (2048).
    #[must_use]
    pub fn resolve_max_tokens(&self, cli_value: Option<i32>) -> i32 {
        cli_value
            .or(self.generation.max_tokens)
            .unwrap_or(2048)
    }

    /// Resolve `system_prompt` with priority: CLI > config > default.
    #[must_use]
    pub fn resolve_system_prompt(&self, cli_value: Option<&str>) -> Option<String> {
        cli_value
            .map(String::from)
            .or_else(|| self.chat.system_prompt.clone())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Format bytes as a human-readable string.
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;
    const KB: u64 = 1024;

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

/// Attempt to get available system RAM in bytes.
///
/// Returns `None` if the information cannot be determined.
fn get_available_ram() -> Option<u64> {
    // Try reading from /proc/meminfo on Linux/WSL
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return Some(kb * 1024);
                        }
                    }
                }
            }
        }
    }

    // Windows: use GlobalMemoryStatusEx via FFI
    #[cfg(target_os = "windows")]
    {
        // Simplified: return a reasonable default for modern systems
        // A proper implementation would use kernel32::GlobalMemoryStatusEx
        return Some(16 * 1024 * 1024 * 1024); // 16 GB default
    }

    // Fallback: assume 16 GB
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        Some(16 * 1024 * 1024 * 1024)
    }

    #[cfg(target_os = "linux")]
    None
}

/// Estimate default VRAM for GPU recommendation when not specified.
fn estimate_default_vram() -> u64 {
    // Check for CUDA_VISIBLE_DEVICES or common VRAM sizes
    // Default to 8 GB (common for consumer GPUs)
    8 * 1024 * 1024 * 1024
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufMetadata;
    use std::collections::HashMap;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_bytes(4_294_967_296), "4.0 GB");
    }

    #[test]
    fn test_model_source_display() {
        assert_eq!(format!("{}", ModelSource::Local), "local");
        assert_eq!(format!("{}", ModelSource::Global), "global");
        assert_eq!(
            format!("{}", ModelSource::Custom(PathBuf::from("/tmp"))),
            "custom (/tmp)"
        );
    }

    #[test]
    fn test_discovered_model_summary_line() {
        let model = DiscoveredModel {
            path: PathBuf::from("/models/test.gguf"),
            source: ModelSource::Global,
            name: "test.gguf".to_string(),
            file_size: 4_000_000_000,
            architecture: Some("llama".to_string()),
            n_params: Some(7_000_000_000),
            context_length: Some(8192),
            file_type: Some("Q4_K_M".to_string()),
        };
        let summary = model.summary_line();
        assert!(summary.contains("test.gguf"));
        assert!(summary.contains("llama"));
        assert!(summary.contains("7.00B"));
        assert!(summary.contains("8192"));
        assert!(summary.contains("Q4_K_M"));
        assert!(summary.contains("global"));
    }

    #[test]
    fn test_model_search_dirs() {
        // This test depends on the filesystem, so just verify it doesn't panic
        let dirs = model_search_dirs();
        // Directories may or may not exist depending on test environment
        for (dir, source) in &dirs {
            assert!(dir.exists(), "Search dir should exist: {}", dir.display());
            assert!(
                matches!(source, ModelSource::Local | ModelSource::Global),
                "Unexpected source: {source:?}"
            );
        }
    }

    #[test]
    fn test_list_models_no_crash() {
        // Should not crash even if no models exist
        let _models = list_models();
    }

    #[test]
    fn test_estimate_memory_from_metadata() {
        let meta = GgufMetadata {
            raw: HashMap::new(),
            architecture: Some("llama".into()),
            description: None,
            name: Some("Test-7B".into()),
            file_type: Some("Q4_K_M".into()),
            parameter_count: Some(7_000_000_000),
            context_length: Some(8192),
            embedding_length: Some(4096),
            num_layers: Some(32),
            num_heads: Some(32),
            num_kv_heads: Some(32),
            chat_template: None,
            tokenizer_type: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let estimate = estimate_memory_from_metadata(&meta, 4_000_000_000, 8192);

        // Model weights should match file size
        assert_eq!(estimate.model_bytes, 4_000_000_000);

        // KV cache should be positive
        assert!(estimate.kv_cache_bytes > 0);

        // Overhead should be at least 256MB
        assert!(estimate.overhead_bytes >= 256 * 1024 * 1024);

        // Total should be sum of parts
        assert_eq!(
            estimate.total_bytes,
            estimate.model_bytes + estimate.kv_cache_bytes + estimate.overhead_bytes
        );
    }

    #[test]
    fn test_estimate_memory_warns_on_ctx_overflow() {
        let meta = GgufMetadata {
            context_length: Some(4096),
            num_layers: Some(32),
            num_heads: Some(32),
            num_kv_heads: Some(32),
            embedding_length: Some(4096),
            ..GgufMetadata::default()
        };

        let estimate = estimate_memory_from_metadata(&meta, 1_000_000_000, 8192);
        assert!(
            estimate
                .warnings
                .iter()
                .any(|w| w.contains("exceeds model training context")),
            "Expected context overflow warning, got: {:?}",
            estimate.warnings
        );
    }

    #[test]
    fn test_recommend_gpu_layers_from_metadata() {
        let meta = GgufMetadata {
            num_layers: Some(32),
            num_heads: Some(32),
            num_kv_heads: Some(32),
            embedding_length: Some(4096),
            ..GgufMetadata::default()
        };

        // Simulate 8GB VRAM
        let rec = recommend_gpu_layers_from_metadata(&meta, 4_000_000_000, Some(8 * 1024 * 1024 * 1024));
        assert!(rec.total_layers == 32);

        // With 0 VRAM, should recommend 0 layers
        let rec_zero = recommend_gpu_layers_from_metadata(&meta, 4_000_000_000, Some(0));
        assert_eq!(rec_zero.recommended_layers, 0);

        // With huge VRAM, should recommend all layers (-1)
        let rec_huge = recommend_gpu_layers_from_metadata(&meta, 4_000_000_000, Some(128 * 1024 * 1024 * 1024));
        assert_eq!(rec_huge.recommended_layers, -1);
    }

    #[test]
    fn test_gpu_recommendation_summary() {
        let rec = GpuRecommendation {
            recommended_layers: 28,
            description: "28/32 layers fit".into(),
            total_layers: 32,
            estimated_vram_bytes: 7 * 1024 * 1024 * 1024,
            gpu_available: true,
        };
        let summary = rec.summary();
        assert!(summary.contains("32"));
        assert!(summary.contains("28"));
    }

    #[test]
    fn test_config_default() {
        let config = ChatvcodeConfig::default();
        assert!(config.model.path.is_none());
        assert!(config.model.n_gpu_layers.is_none());
        assert!(config.generation.temperature.is_none());
        assert!(config.chat.system_prompt.is_none());
    }

    #[test]
    fn test_config_serialize_deserialize() {
        let config = ChatvcodeConfig {
            model: ModelConfig {
                path: Some("/models/test.gguf".into()),
                n_gpu_layers: Some(32),
                n_ctx: Some(4096),
                n_threads: None,
                template: Some("chatml".into()),
                use_mmap: Some(true),
                verbose_log: None,
            },
            generation: GenerationConfig {
                temperature: Some(0.5),
                top_p: None,
                top_k: Some(50),
                max_tokens: None,
            },
            chat: ChatConfig {
                system_prompt: Some("You are helpful.".into()),
                stream: Some(true),
                retrieval: None,
                top_k_retrieval: None,
                context_token_budget: None,
            },
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        let parsed: ChatvcodeConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.model.path.as_deref(), Some("/models/test.gguf"));
        assert_eq!(parsed.model.n_gpu_layers, Some(32));
        assert_eq!(parsed.model.n_ctx, Some(4096));
        assert!(parsed.model.n_threads.is_none());
        assert_eq!(parsed.model.template.as_deref(), Some("chatml"));
        assert_eq!(parsed.generation.temperature, Some(0.5));
        assert_eq!(parsed.generation.top_k, Some(50));
        assert!(parsed.generation.top_p.is_none());
        assert_eq!(
            parsed.chat.system_prompt.as_deref(),
            Some("You are helpful.")
        );
    }

    #[test]
    fn test_config_resolve_priority() {
        let config = ChatvcodeConfig {
            model: ModelConfig {
                path: Some("/config/model.gguf".into()),
                n_gpu_layers: Some(16),
                n_ctx: Some(4096),
                ..ModelConfig::default()
            },
            generation: GenerationConfig {
                temperature: Some(0.3),
                top_p: Some(0.8),
                ..GenerationConfig::default()
            },
            ..ChatvcodeConfig::default()
        };

        // CLI overrides config
        assert_eq!(
            config.resolve_model_path(Some("/cli/model.gguf")),
            Some("/cli/model.gguf".to_string())
        );
        assert_eq!(config.resolve_n_gpu_layers(Some(32)), 32);
        assert_eq!(config.resolve_n_ctx(Some(8192)), 8192);
        assert!((config.resolve_temperature(Some(0.9)) - 0.9).abs() < f32::EPSILON);

        // Config fills in when CLI is None
        assert_eq!(
            config.resolve_model_path(None),
            Some("/config/model.gguf".to_string())
        );
        assert_eq!(config.resolve_n_gpu_layers(None), 16);
        assert_eq!(config.resolve_n_ctx(None), 4096);
        assert!((config.resolve_temperature(None) - 0.3).abs() < f32::EPSILON);
        assert!((config.resolve_top_p(None) - 0.8).abs() < f32::EPSILON);

        // Built-in default when both CLI and config are None
        assert_eq!(config.resolve_n_threads(None), num_cpus::get() as i32);
        assert!((config.resolve_top_k(None) - 40) == 0);
        assert_eq!(config.resolve_max_tokens(None), 2048);
        assert_eq!(config.resolve_template(None), "auto");
    }

    #[test]
    fn test_config_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");

        let config = ChatvcodeConfig {
            model: ModelConfig {
                path: Some("/test/model.gguf".into()),
                n_gpu_layers: Some(-1),
                ..ModelConfig::default()
            },
            generation: GenerationConfig {
                temperature: Some(0.42),
                ..GenerationConfig::default()
            },
            chat: ChatConfig {
                system_prompt: Some("Be concise.".into()),
                ..ChatConfig::default()
            },
        };

        config.save_to(&path).unwrap();
        assert!(path.exists());

        let loaded = ChatvcodeConfig::load_from(&path).unwrap();
        assert_eq!(loaded.model.path.as_deref(), Some("/test/model.gguf"));
        assert_eq!(loaded.model.n_gpu_layers, Some(-1));
        assert_eq!(loaded.generation.temperature, Some(0.42));
        assert_eq!(loaded.chat.system_prompt.as_deref(), Some("Be concise."));
    }

    #[test]
    fn test_config_load_nonexistent_returns_default() {
        let path = PathBuf::from("/nonexistent/path/config.json");
        let result = ChatvcodeConfig::load_from(&path);
        assert!(result.is_err()); // Should error on missing file when explicitly requested
    }

    #[test]
    fn test_config_resolve_system_prompt() {
        let config = ChatvcodeConfig {
            chat: ChatConfig {
                system_prompt: Some("Config prompt".into()),
                ..ChatConfig::default()
            },
            ..ChatvcodeConfig::default()
        };

        assert_eq!(
            config.resolve_system_prompt(Some("CLI prompt")),
            Some("CLI prompt".to_string())
        );
        assert_eq!(
            config.resolve_system_prompt(None),
            Some("Config prompt".to_string())
        );
    }

    #[test]
    fn test_default_config_path() {
        let path = default_config_path();
        assert!(path.to_string_lossy().contains(".chatvcode"));
        assert!(path.to_string_lossy().contains("config.json"));
    }

    #[test]
    fn test_local_config_path() {
        let path = local_config_path();
        assert!(path.to_string_lossy().contains(".chatvcode"));
        assert!(path.to_string_lossy().contains("config.json"));
        // Local config should be under current working directory
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        assert!(path.starts_with(&cwd));
    }

    #[test]
    fn test_config_merge_over_local_priority() {
        let global = ChatvcodeConfig {
            model: ModelConfig {
                path: Some("/global/model.gguf".into()),
                n_gpu_layers: Some(16),
                n_ctx: Some(4096),
                n_threads: Some(4),
                ..ModelConfig::default()
            },
            generation: GenerationConfig {
                temperature: Some(0.5),
                top_p: Some(0.8),
                ..GenerationConfig::default()
            },
            chat: ChatConfig {
                system_prompt: Some("Global prompt".into()),
                ..ChatConfig::default()
            },
        };

        let local = ChatvcodeConfig {
            model: ModelConfig {
                path: Some("/local/model.gguf".into()),
                n_ctx: Some(8192),
                // n_gpu_layers and n_threads are None in local
                ..ModelConfig::default()
            },
            generation: GenerationConfig {
                temperature: Some(0.3),
                // top_p is None in local
                ..GenerationConfig::default()
            },
            chat: ChatConfig {
                system_prompt: Some("Local prompt".into()),
                ..ChatConfig::default()
            },
        };

        // local.merge_over(global): local values override global where Some
        let merged = local.merge_over(global);

        // Local overrides global
        assert_eq!(merged.model.path.as_deref(), Some("/local/model.gguf"));
        assert_eq!(merged.model.n_ctx, Some(8192));
        assert!((merged.generation.temperature.unwrap() - 0.3).abs() < f32::EPSILON);
        assert_eq!(merged.chat.system_prompt.as_deref(), Some("Local prompt"));

        // Global fills in where local is None
        assert_eq!(merged.model.n_gpu_layers, Some(16));
        assert_eq!(merged.model.n_threads, Some(4));
        assert!((merged.generation.top_p.unwrap() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_merge_over_empty_local() {
        let global = ChatvcodeConfig {
            model: ModelConfig {
                path: Some("/global/model.gguf".into()),
                n_gpu_layers: Some(32),
                ..ModelConfig::default()
            },
            generation: GenerationConfig {
                temperature: Some(0.7),
                ..GenerationConfig::default()
            },
            ..ChatvcodeConfig::default()
        };

        let local = ChatvcodeConfig::default();

        // Empty local should not override any global values
        let merged = local.merge_over(global);
        assert_eq!(merged.model.path.as_deref(), Some("/global/model.gguf"));
        assert_eq!(merged.model.n_gpu_layers, Some(32));
        assert!((merged.generation.temperature.unwrap() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_merge_over_empty_global() {
        let global = ChatvcodeConfig::default();
        let local = ChatvcodeConfig {
            model: ModelConfig {
                path: Some("/local/model.gguf".into()),
                ..ModelConfig::default()
            },
            ..ChatvcodeConfig::default()
        };

        let merged = local.merge_over(global);
        assert_eq!(merged.model.path.as_deref(), Some("/local/model.gguf"));
        assert!(merged.model.n_gpu_layers.is_none());
        assert!(merged.generation.temperature.is_none());
    }

    #[test]
    fn test_config_load_default_with_both_files() {
        // Test the merge behavior when both local and global config files exist
        let tmp = tempfile::tempdir().unwrap();
        let global_dir = tmp.path().join("global");
        let local_dir = tmp.path().join("local");
        std::fs::create_dir_all(&global_dir).unwrap();
        std::fs::create_dir_all(&local_dir).unwrap();

        let global_path = global_dir.join("config.json");
        let local_path = local_dir.join("config.json");

        let global_config = ChatvcodeConfig {
            model: ModelConfig {
                path: Some("/global/model.gguf".into()),
                n_gpu_layers: Some(16),
                ..ModelConfig::default()
            },
            ..ChatvcodeConfig::default()
        };
        global_config.save_to(&global_path).unwrap();

        let local_config = ChatvcodeConfig {
            model: ModelConfig {
                path: Some("/local/model.gguf".into()),
                ..ModelConfig::default()
            },
            ..ChatvcodeConfig::default()
        };
        local_config.save_to(&local_path).unwrap();

        // Load both and merge manually (simulating load_default behavior)
        let loaded_global = ChatvcodeConfig::load_from(&global_path).unwrap();
        let loaded_local = ChatvcodeConfig::load_from(&local_path).unwrap();
        let merged = loaded_local.merge_over(loaded_global);

        // Local overrides global
        assert_eq!(merged.model.path.as_deref(), Some("/local/model.gguf"));
        // Global fills in where local is None
        assert_eq!(merged.model.n_gpu_layers, Some(16));
    }

    #[test]
    fn test_memory_estimate_summary() {
        let est = MemoryEstimate {
            model_bytes: 4_000_000_000,
            kv_cache_bytes: 1_000_000_000,
            overhead_bytes: 256_000_000,
            total_bytes: 5_256_000_000,
            fits_in_ram: true,
            warnings: vec![],
        };
        let summary = est.summary();
        assert!(summary.contains("Model weights"));
        assert!(summary.contains("KV cache"));
        assert!(summary.contains("Total"));
        assert!(summary.contains("✓"));
    }

    #[test]
    fn test_memory_estimate_summary_with_warnings() {
        let est = MemoryEstimate {
            model_bytes: 4_000_000_000,
            kv_cache_bytes: 1_000_000_000,
            overhead_bytes: 256_000_000,
            total_bytes: 5_256_000_000,
            fits_in_ram: false,
            warnings: vec!["Not enough RAM".into()],
        };
        let summary = est.summary();
        assert!(summary.contains("⚠"));
        assert!(summary.contains("Not enough RAM"));
    }

    #[test]
    fn test_list_models_in_dir_empty() {
        let dir = tempfile::tempdir().unwrap();
        let models = list_models_in_dir(dir.path());
        assert!(models.is_empty());
    }

    #[test]
    fn test_list_models_in_dir_nonexistent() {
        let models = list_models_in_dir(Path::new("/nonexistent/dir"));
        assert!(models.is_empty());
    }
}
