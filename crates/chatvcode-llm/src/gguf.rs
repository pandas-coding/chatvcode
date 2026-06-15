//! GGUF file format parsing, validation, and metadata extraction.
//!
//! GGUF (GGML Unified Format) is the model file format used by `llama.cpp`.
//! This module provides lightweight utilities to inspect GGUF files before
//! passing them to the llama.cpp C API, enabling early validation, metadata
//! preview, and chat template auto-detection.
//!
//! # Format Overview (simplified)
//!
//! ```text
//! Offset  Size    Field
//! 0       4       Magic: "GGUF" (0x47 0x47 0x55 0x46)
//! 4       4       Version (u32, little-endian)
//! 8       8       Tensor count (u64)
//! 16      8       Metadata KV count (u64)
//! 24      *       Metadata key-value pairs
//! ...     *       Tensor info entries
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::error::{LlmError, LlmResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Expected magic bytes for GGUF files.
pub const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];

/// Supported GGUF versions.
pub const SUPPORTED_VERSIONS: &[u32] = &[2, 3];

// ---------------------------------------------------------------------------
// GGUF value types (for metadata parsing)
// ---------------------------------------------------------------------------

/// GGUF metadata value types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum GgufValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl GgufValueType {
    const fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::U8),
            1 => Some(Self::I8),
            2 => Some(Self::U16),
            3 => Some(Self::I16),
            4 => Some(Self::U32),
            5 => Some(Self::I32),
            6 => Some(Self::F32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::U64),
            11 => Some(Self::I64),
            12 => Some(Self::F64),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// GGUF header
// ---------------------------------------------------------------------------

/// Parsed GGUF file header.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    /// Magic bytes (always `GGUF_MAGIC` if valid).
    pub magic: [u8; 4],
    /// GGUF format version.
    pub version: u32,
    /// Number of tensors in the file.
    pub tensor_count: u64,
    /// Number of metadata key-value pairs.
    pub metadata_kv_count: u64,
    /// Total file size in bytes.
    pub file_size: u64,
}

// ---------------------------------------------------------------------------
// GGUF metadata (extracted keys of interest)
// ---------------------------------------------------------------------------

/// Extracted GGUF metadata relevant for model discovery and templating.
#[derive(Debug, Clone, Default)]
pub struct GgufMetadata {
    /// All raw metadata key-value pairs (stringified).
    pub raw: HashMap<String, String>,

    /// Model architecture (e.g., "llama", "mistral", "gemma").
    pub architecture: Option<String>,

    /// Human-readable model description.
    pub description: Option<String>,

    /// Model name.
    pub name: Option<String>,

    /// Quantization / file type string.
    pub file_type: Option<String>,

    /// Estimated parameter count (read from metadata if present).
    pub parameter_count: Option<u64>,

    /// Context length the model was trained with.
    pub context_length: Option<i32>,

    /// Embedding dimension.
    pub embedding_length: Option<i32>,

    /// Number of layers.
    pub num_layers: Option<i32>,

    /// Number of attention heads.
    pub num_heads: Option<i32>,

    /// Number of key/value heads.
    pub num_kv_heads: Option<i32>,

    /// Chat template string (jinja format).
    pub chat_template: Option<String>,

    /// Tokenizer type (e.g., "bpe", "sentencepiece").
    pub tokenizer_type: Option<String>,

    /// BOS token id.
    pub bos_token_id: Option<i32>,

    /// EOS token id.
    pub eos_token_id: Option<i32>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Validate that a file is a well-formed GGUF file.
///
/// Checks the magic bytes and version. Returns the parsed header on success.
/// This is a lightweight check — it reads only the first 24+ bytes.
pub fn validate_gguf(path: &Path) -> LlmResult<GgufHeader> {
    let file_size = fs::metadata(path)
        .map_err(|e| {
            LlmError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("cannot access file '{}': {e}", path.display()),
            ))
        })?
        .len();

    if file_size < 24 {
        return Err(LlmError::ModelLoadFailed(format!(
            "file '{}' is too small ({file_size} bytes) to be a valid GGUF file",
            path.display()
        )));
    }

    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if magic != GGUF_MAGIC {
        return Err(LlmError::ModelLoadFailed(format!(
            "file '{}' is not a valid GGUF file (invalid magic bytes: {magic:?}, expected {:?})",
            path.display(),
            GGUF_MAGIC
        )));
    }

    // Read version
    let version = read_u32_le(&mut reader)?;
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(LlmError::Unsupported(format!(
            "GGUF version {version} is not supported (supported: {SUPPORTED_VERSIONS:?}). \
             The model at '{}' may require a newer version of llama.cpp.",
            path.display()
        )));
    }

    // Read tensor count and metadata count
    let tensor_count = read_u64_le(&mut reader)?;
    let metadata_kv_count = read_u64_le(&mut reader)?;

    log::debug!(
        "GGUF header validated: path={}, version={version}, tensors={tensor_count}, metadata_kvs={metadata_kv_count}, size={file_size}",
        path.display()
    );

    Ok(GgufHeader { magic, version, tensor_count, metadata_kv_count, file_size })
}

/// Check if a file appears to be a GGUF file (quick magic-byte check).
#[must_use]
pub fn is_gguf_file(path: &Path) -> bool {
    if let Ok(mut file) = fs::File::open(path) {
        let mut magic = [0u8; 4];
        if file.read_exact(&mut magic).is_ok() {
            return magic == GGUF_MAGIC;
        }
    }
    false
}

/// Read GGUF metadata from a file without loading the full model.
///
/// This parses the metadata key-value section of the GGUF file,
/// extracting fields relevant for model discovery and template selection.
pub fn read_gguf_metadata(path: &Path) -> LlmResult<GgufMetadata> {
    let _start = Instant::now();

    // Open the file once and read everything in a single pass
    let file_size = fs::metadata(path)
        .map_err(|e| {
            LlmError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("cannot access file '{}': {e}", path.display()),
            ))
        })?
        .len();

    if file_size < 24 {
        return Err(LlmError::ModelLoadFailed(format!(
            "file '{}' is too small ({file_size} bytes) to be a valid GGUF file",
            path.display()
        )));
    }

    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read header: magic + version + tensor_count + metadata_kv_count
    let mut magic_buf = [0u8; 4];
    reader.read_exact(&mut magic_buf)?;
    if magic_buf != GGUF_MAGIC {
        return Err(LlmError::ModelLoadFailed(format!(
            "file '{}' is not a valid GGUF file (invalid magic bytes: {magic_buf:?})",
            path.display()
        )));
    }
    let version = read_u32_le(&mut reader)?;
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(LlmError::Unsupported(format!(
            "GGUF version {version} is not supported (supported: {SUPPORTED_VERSIONS:?})",
        )));
    }
    let _tensor_count = read_u64_le(&mut reader)?;
    let metadata_kv_count = read_u64_le(&mut reader)?;

    let mut raw = HashMap::new();
    let mut meta = GgufMetadata::default();

    for _ in 0..metadata_kv_count {
        // Read key string
        let key = read_gguf_string(&mut reader)?;

        // Read value type
        let val_type_raw = read_u32_le(&mut reader)?;
        let val_type = GgufValueType::from_u32(val_type_raw).unwrap_or_else(|| {
            log::warn!(
                "Unknown GGUF value type {val_type_raw} for key '{key}', treating as string"
            );
            GgufValueType::String
        });

        // Read and store value
        let val_str = match val_type {
            GgufValueType::U8 => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                format!("{}", buf[0])
            }
            GgufValueType::I8 => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                format!("{}", buf[0] as i8)
            }
            GgufValueType::U16 => {
                let v = read_u16_le(&mut reader)?;
                format!("{v}")
            }
            GgufValueType::I16 => {
                let v = read_i16_le(&mut reader)?;
                format!("{v}")
            }
            GgufValueType::U32 => {
                let v = read_u32_le(&mut reader)?;
                format!("{v}")
            }
            GgufValueType::I32 => {
                let v = read_i32_le(&mut reader)?;
                format!("{v}")
            }
            GgufValueType::F32 => {
                let v = read_f32_le(&mut reader)?;
                format!("{v}")
            }
            GgufValueType::U64 => {
                let v = read_u64_le(&mut reader)?;
                format!("{v}")
            }
            GgufValueType::I64 => {
                let v = read_i64_le(&mut reader)?;
                format!("{v}")
            }
            GgufValueType::F64 => {
                let v = read_f64_le(&mut reader)?;
                format!("{v}")
            }
            GgufValueType::Bool => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                format!("{}", buf[0] != 0)
            }
            GgufValueType::String => read_gguf_string(&mut reader)?,
            GgufValueType::Array => {
                // Skip arrays for now — read item type and length, then skip data
                let arr_item_type = read_u32_le(&mut reader)?;
                let arr_len = read_u64_le(&mut reader)?;
                let item_size = gguf_value_type_size(arr_item_type);
                if let Some(size) = item_size {
                    let skip_bytes = arr_len as u64 * size;
                    let current_pos = reader.stream_position()?;
                    reader.seek(SeekFrom::Start(current_pos + skip_bytes))?;
                } else {
                    // Variable-size items (strings) — skip by reading string lengths
                    for _ in 0..arr_len {
                        let _ = read_gguf_string(&mut reader)?;
                    }
                }
                format!("[array:{arr_len}]")
            }
        };

        raw.insert(key.clone(), val_str.clone());

        // Extract general metadata fields
        match key.as_str() {
            "general.architecture" => meta.architecture = Some(val_str.clone()),
            "general.name" => meta.name = Some(val_str.clone()),
            "general.description" => meta.description = Some(val_str.clone()),
            "general.file_type" => meta.file_type = Some(val_str.clone()),
            "tokenizer.ggml.model" => meta.tokenizer_type = Some(val_str.clone()),
            "tokenizer.ggml.bos_token_id" => {
                meta.bos_token_id = val_str.parse::<i32>().ok();
            }
            "tokenizer.ggml.eos_token_id" => {
                meta.eos_token_id = val_str.parse::<i32>().ok();
            }
            "tokenizer.chat_template" => {
                meta.chat_template = Some(val_str.clone());
            }
            _ => {}
        }

        // Handle architecture-specific numeric fields dynamically.
        // In GGUF, these keys are prefixed with the architecture name,
        // e.g., "llama.context_length" or "mistral.context_length".
        // We parse the prefix from the detected architecture and match accordingly.
        if let Some(ref arch) = meta.architecture
            && key.starts_with(arch)
            && key.len() > arch.len() + 1
        {
            let suffix = &key[arch.len() + 1..]; // skip the dot
            match suffix {
                "context_length" => meta.context_length = val_str.parse::<i32>().ok(),
                "embedding_length" => meta.embedding_length = val_str.parse::<i32>().ok(),
                "block_count" => meta.num_layers = val_str.parse::<i32>().ok(),
                "attention.head_count" => meta.num_heads = val_str.parse::<i32>().ok(),
                "attention.head_count_kv" => meta.num_kv_heads = val_str.parse::<i32>().ok(),
                _ => {}
            }
        }
    }

    log::debug!(
        "GGUF metadata read: arch={:?}, params={:?}, ctx={:?}, template={}, elapsed={:?}",
        meta.architecture,
        meta.parameter_count,
        meta.context_length,
        if meta.chat_template.is_some() { "yes" } else { "no" },
        _start.elapsed()
    );

    meta.raw = raw;
    Ok(meta)
}

// ---------------------------------------------------------------------------
// Chat template auto-detection
// ---------------------------------------------------------------------------

/// Suggested chat template based on model architecture.
///
/// Uses the architecture name and available metadata to recommend
/// a [`crate::types::ChatTemplate`] variant.
///
/// # Returns
///
/// - `Some(template_name)` when a template can be inferred from the model
///   metadata or a known architecture.
/// - `None` when the architecture is unknown and no metadata template is
///   present. In this case callers should fall back to a portable default
///   such as `ChatML` (or `Raw` if no formatting is desired).
#[must_use]
pub fn infer_chat_template(meta: &GgufMetadata) -> Option<String> {
    // If the model already has a chat template in metadata, use it
    if let Some(tmpl) = &meta.chat_template
        && !tmpl.is_empty()
    {
        return Some(tmpl.clone());
    }

    // Otherwise, infer from architecture name
    let arch = meta.architecture.as_deref().unwrap_or("").to_lowercase();

    match arch.as_str() {
        "llama" => Some("llama3".to_string()),
        "mistral" => Some("chatml".to_string()),
        "gemma" | "gemma2" => Some("gemma".to_string()),
        "phi" | "phi3" | "phi4" => Some("chatml".to_string()),
        "qwen2" | "qwen2.5" => Some("chatml".to_string()),
        "deepseek" | "deepseek2" | "deepseek3" => Some("deepseek".to_string()),
        "command-r" | "c4ai" => Some("chatml".to_string()),
        "chatglm" => Some("chatglm3".to_string()),
        "falcon" => Some("chatml".to_string()),
        "starcoder" | "starcoder2" => Some("chatml".to_string()),
        "codellama" => Some("llama3".to_string()),
        "stablelm" | "stablelm2" => Some("chatml".to_string()),
        "grok-1" => Some("chatml".to_string()),
        "dbrx" => Some("chatml".to_string()),
        "exaone" => Some("chatml".to_string()),
        "orion" => Some("chatml".to_string()),
        "olmo" | "olmoe" => Some("chatml".to_string()),
        "arctic" => Some("chatml".to_string()),
        "granite" | "granite-3" => Some("chatml".to_string()),
        "nemotron" | "nemotron3" | "nemotron4" => Some("chatml".to_string()),
        "minicpm" | "minicpm3" => Some("chatml".to_string()),
        "mamba" | "mamba2" => Some("chatml".to_string()),
        "cohere" | "cohere2" => Some("chatml".to_string()),
        "bitnet" => Some("chatml".to_string()),
        "jamba" => Some("chatml".to_string()),
        "t5" | "t5encoder" => Some("chatml".to_string()),
        "openelm" => Some("chatml".to_string()),
        "chameleon" => Some("chatml".to_string()),
        "" => {
            log::info!(
                "No architecture detected in GGUF metadata; cannot infer chat template"
            );
            None
        }
        _ => {
            log::info!(
                "Unknown architecture '{}', cannot infer chat template; \
                 caller should fall back to ChatML or Raw",
                meta.architecture.as_deref().unwrap_or("unknown")
            );
            None
        }
    }
}

/// Format a human-readable summary of GGUF metadata.
#[must_use]
pub fn format_gguf_summary(path: &Path, meta: &GgufMetadata) -> String {
    let mut lines = Vec::new();
    lines.push(format!("📄 Model: {}", path.display()));

    if let Some(arch) = &meta.architecture {
        lines.push(format!("   Architecture: {arch}"));
    }
    if let Some(name) = &meta.name {
        lines.push(format!("   Name: {name}"));
    }
    if let Some(desc) = &meta.description {
        lines.push(format!("   Description: {desc}"));
    }
    if let Some(ft) = &meta.file_type {
        lines.push(format!("   Quantization: {ft}"));
    }
    if let Some(ctx) = meta.context_length {
        lines.push(format!("   Context size: {ctx}"));
    }
    if let Some(embd) = meta.embedding_length {
        lines.push(format!("   Embedding dim: {embd}"));
    }
    if let Some(layers) = meta.num_layers {
        lines.push(format!("   Layers: {layers}"));
    }
    if let Some(heads) = meta.num_heads {
        lines.push(format!("   Attention heads: {heads}"));
    }
    if let Some(tok) = &meta.tokenizer_type {
        lines.push(format!("   Tokenizer: {tok}"));
    }
    let template_status = if meta.chat_template.is_some() { "embedded" } else { "auto-detected" };
    lines.push(format!("   Chat template: {template_status}"));

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Model discovery with validation
// ---------------------------------------------------------------------------

/// Discover GGUF models in a directory, with validation.
///
/// Returns a list of (path, header, metadata) for valid GGUF files.
/// Invalid files are logged as warnings but do not cause errors.
#[must_use]
pub fn discover_gguf_models(
    dir: &Path,
) -> Vec<(PathBuf, GgufHeader, Result<GgufMetadata, LlmError>)> {
    let mut results = Vec::new();

    if !dir.exists() {
        return results;
    }

    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            log::warn!("Cannot read model directory '{}': {}", dir.display(), e);
            return results;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();

        // Skip non-files and non-GGUF extensions
        if !path.is_file() {
            continue;
        }

        let is_gguf = path.extension().is_some_and(|ext| ext == "gguf") || is_gguf_file(&path);

        if !is_gguf {
            continue;
        }

        match validate_gguf(&path) {
            Ok(header) => {
                let metadata = read_gguf_metadata(&path);
                results.push((path, header, metadata));
            }
            Err(e) => {
                log::warn!("Skipping invalid GGUF file '{}': {}", path.display(), e);
            }
        }
    }

    results
}

/// Validate a model file before attempting to load it into memory.
///
/// Checks:
/// - File exists and is a regular file
/// - Has valid GGUF magic bytes
/// - Has a supported GGUF version
/// - Is not trivially empty or corrupted
///
/// Returns a [`GgufMetadata`] preview if successful.
pub fn pre_validate_model(path: &Path) -> LlmResult<GgufMetadata> {
    if !path.exists() {
        return Err(LlmError::ModelNotFound(path.display().to_string()));
    }

    if !path.is_file() {
        return Err(LlmError::ModelNotFound(format!("'{}' is not a regular file", path.display())));
    }

    // Validate GGUF format
    let _header = validate_gguf(path)?;

    // Read metadata
    let metadata = read_gguf_metadata(path)?;

    Ok(metadata)
}

// ---------------------------------------------------------------------------
// Helper: safe model loading with enhanced error messages
// ---------------------------------------------------------------------------

/// Load a model with comprehensive pre-flight checks and error messages.
///
/// This function:
/// 1. Validates that the file exists and is a valid GGUF file
/// 2. Reads metadata before loading (for friendlier error messages)
/// 3. Loads the model using llama.cpp
/// 4. Checks for OOM or other loading failures with helpful diagnostics
pub fn load_model_safe(
    path: &Path,
    n_gpu_layers: i32,
    use_mmap: bool,
    use_mlock: bool,
) -> LlmResult<super::context::LlamaModel> {
    // Pre-flight validation
    let meta = pre_validate_model(path).map_err(|e| {
        match &e {
            LlmError::ModelNotFound(_) => {
                let help = format!(
                    "Model file not found. Please download a GGUF model and place it at:\n  {}\n\n\
                     Recommended models:\n  - CodeLlama: https://huggingface.co/TheBloke/CodeLlama-7B-GGUF\n  - DeepSeek-Coder: https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF\n  - Qwen2.5-Coder: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF\n\n\
                     Place the downloaded .gguf file in ~/.chatvcode/models/ for auto-discovery.",
                    crate::service::default_model_dir().display()
                );
                LlmError::ModelNotFound(format!("{e}\n\n{help}"))
            }
            LlmError::ModelLoadFailed(msg) if msg.contains("not a valid GGUF") => {
                LlmError::ModelLoadFailed(format!(
                    "The file at '{}' does not appear to be a valid GGUF model.\n\
                     Make sure you downloaded a .gguf file (not a .safetensors or .bin file).\n\
                     Look for files with 'GGUF' in the filename on HuggingFace.",
                    path.display()
                ))
            }
            LlmError::Unsupported(msg) if msg.contains("GGUF version") => {
                LlmError::Unsupported(format!(
                    "{msg}.\nPlease re-download the model in a supported format or update llama.cpp."
                ))
            }
            _ => e,
        }
    })?;

    log::info!(
        "Pre-validated model: arch={:?}, ctx={:?}, size={}",
        meta.architecture,
        meta.context_length,
        format_file_size(path)
    );

    // Attempt model loading
    let model = super::context::LlamaModel::load(path, n_gpu_layers, use_mmap, use_mlock)
        .map_err(|e| {
            match &e {
                LlmError::ModelLoadFailed(_) => {
                    // Enhance OOM errors
                    let file_size = format_file_size(path);
                    LlmError::ModelLoadFailed(format!(
                        "Failed to load model '{}' ({file_size}).\n\n\
                         Possible causes:\n\
                         1. Out of memory — try reducing GPU layers (--n-gpu-layers 0 for CPU-only)\n\
                         2. Corrupted file — try re-downloading the model\n\
                         3. Insufficient RAM/VRAM — the model may be too large for your system\n\
                         4. Incompatible GGUF version — the model may require a newer llama.cpp\n\n\
                         Error details: {e}",
                        path.display()
                    ))
                }
                _ => e,
            }
        })?;

    log::info!(
        "Model loaded: {} ({} params, {} ctx)",
        path.display(),
        format_param_count(model.info().n_params),
        model.info().n_ctx_train
    );

    Ok(model)
}

// ---------------------------------------------------------------------------
// Validation-only scan (no loading)
// ---------------------------------------------------------------------------

/// Scan a model file and print a summary to the logger.
///
/// Useful for CLI commands like `chatvcode model info <path>`.
/// Does not load the model into memory.
pub fn scan_model(path: &Path) -> LlmResult<GgufMetadata> {
    let meta = pre_validate_model(path)?;
    let summary = format_gguf_summary(path, &meta);
    log::info!("Model scan:\n{summary}");
    Ok(meta)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn read_u8<R: Read>(r: &mut R) -> std::io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16_le<R: Read>(r: &mut R) -> std::io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16_le<R: Read>(r: &mut R) -> std::io::Result<i16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32_le<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32_le<R: Read>(r: &mut R) -> std::io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64_le<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64_le<R: Read>(r: &mut R) -> std::io::Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32_le<R: Read>(r: &mut R) -> std::io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64_le<R: Read>(r: &mut R) -> std::io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

/// Read a GGUF string: 8-byte length (u64) followed by UTF-8 bytes.
fn read_gguf_string<R: Read>(reader: &mut R) -> std::io::Result<String> {
    let len = read_u64_le(reader)? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Returns the byte size of a GGUF value type for array skipping.
const fn gguf_value_type_size(vt: u32) -> Option<u64> {
    match vt {
        0 => Some(1),  // u8
        1 => Some(1),  // i8
        2 => Some(2),  // u16
        3 => Some(2),  // i16
        4 => Some(4),  // u32
        5 => Some(4),  // i32
        6 => Some(4),  // f32
        7 => Some(1),  // bool
        8 => None,     // string (variable)
        9 => None,     // array (variable)
        10 => Some(8), // u64
        11 => Some(8), // i64
        12 => Some(8), // f64
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Display helpers (shared)
// ---------------------------------------------------------------------------

/// Format a file size in human-readable format.
#[must_use]
pub fn format_file_size(path: &Path) -> String {
    match fs::metadata(path) {
        Ok(meta) => {
            let bytes = meta.len();
            if bytes >= 1_073_741_824 {
                format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
            } else if bytes >= 1_048_576 {
                format!("{:.2} MB", bytes as f64 / 1_048_576.0)
            } else if bytes >= 1024 {
                format!("{:.2} KB", bytes as f64 / 1024.0)
            } else {
                format!("{bytes} B")
            }
        }
        Err(_) => "unknown".to_string(),
    }
}

/// Format a parameter count in human-readable form.
#[must_use]
pub fn format_param_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{n}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tempfile::NamedTempFile;

    // -----------------------------------------------------------------------
    // Unit tests for binary reading helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_u32_le() {
        let data = [0x78, 0x56, 0x34, 0x12];
        let mut cursor = Cursor::new(data);
        assert_eq!(read_u32_le(&mut cursor).unwrap(), 0x12345678);
    }

    #[test]
    fn test_read_u64_le() {
        let data = [0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01];
        let mut cursor = Cursor::new(data);
        assert_eq!(read_u64_le(&mut cursor).unwrap(), 0x0123456789ABCDEF);
    }

    #[test]
    fn test_read_gguf_string() {
        let data: Vec<u8> = {
            let mut v = Vec::new();
            v.extend_from_slice(&5u64.to_le_bytes()); // length
            v.extend_from_slice(b"hello"); // content
            v
        };
        let mut cursor = Cursor::new(data);
        assert_eq!(read_gguf_string(&mut cursor).unwrap(), "hello");
    }

    #[test]
    fn test_read_gguf_string_empty() {
        let data = 0u64.to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        assert_eq!(read_gguf_string(&mut cursor).unwrap(), "");
    }

    // -----------------------------------------------------------------------
    // GGUF validation tests
    // -----------------------------------------------------------------------

    fn make_gguf_file(version: u32, tensor_count: u64, kv_count: u64, body: &[u8]) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC);
        data.extend_from_slice(&version.to_le_bytes());
        data.extend_from_slice(&tensor_count.to_le_bytes());
        data.extend_from_slice(&kv_count.to_le_bytes());
        data.extend_from_slice(body);
        data
    }

    #[test]
    fn test_validate_gguf_valid_v3() -> LlmResult<()> {
        let data = make_gguf_file(3, 0, 0, &[]);
        let file = NamedTempFile::new()?;
        fs::write(file.path(), &data)?;

        let header = validate_gguf(file.path())?;
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 0);
        assert_eq!(header.metadata_kv_count, 0);
        Ok(())
    }

    #[test]
    fn test_validate_gguf_valid_v2() -> LlmResult<()> {
        let data = make_gguf_file(2, 1, 5, &[]);
        let file = NamedTempFile::new()?;
        fs::write(file.path(), &data)?;

        let header = validate_gguf(file.path())?;
        assert_eq!(header.version, 2);
        assert_eq!(header.tensor_count, 1);
        assert_eq!(header.metadata_kv_count, 5);
        Ok(())
    }

    #[test]
    fn test_validate_gguf_bad_magic() -> LlmResult<()> {
        let mut data = make_gguf_file(3, 0, 0, &[]);
        data[0] = b'X'; // corrupt magic
        let file = NamedTempFile::new()?;
        fs::write(file.path(), &data)?;

        let result = validate_gguf(file.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not a valid GGUF"), "unexpected error: {err}");
        Ok(())
    }

    #[test]
    fn test_validate_gguf_unsupported_version() -> LlmResult<()> {
        let data = make_gguf_file(99, 0, 0, &[]);
        let file = NamedTempFile::new()?;
        fs::write(file.path(), &data)?;

        let result = validate_gguf(file.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not supported"), "unexpected error: {err}");
        Ok(())
    }

    #[test]
    fn test_validate_gguf_too_small() -> LlmResult<()> {
        let data = vec![0u8; 10]; // too small for header
        let file = NamedTempFile::new()?;
        fs::write(file.path(), &data)?;

        let result = validate_gguf(file.path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("too small"), "unexpected error: {err}");
        Ok(())
    }

    #[test]
    fn test_is_gguf_file_positive() -> LlmResult<()> {
        let data = make_gguf_file(3, 0, 0, &[]);
        let file = NamedTempFile::new()?;
        fs::write(file.path(), &data)?;

        assert!(is_gguf_file(file.path()));
        Ok(())
    }

    #[test]
    fn test_is_gguf_file_negative() -> LlmResult<()> {
        let file = NamedTempFile::new()?;
        fs::write(file.path(), b"not a gguf file")?;

        assert!(!is_gguf_file(file.path()));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // GGUF metadata parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_gguf_metadata_minimal() -> LlmResult<()> {
        let mut body = Vec::new();

        // Key: "general.architecture" (string, 21 bytes)
        let key = b"general.architecture";
        body.extend_from_slice(&(key.len() as u64).to_le_bytes()); // key length = 21
        body.extend_from_slice(key);
        body.extend_from_slice(&8u32.to_le_bytes()); // type: string
        body.extend_from_slice(&5u64.to_le_bytes()); // value length
        body.extend_from_slice(b"llama");

        let data = make_gguf_file(3, 0, 1, &body);
        let file = NamedTempFile::new()?;
        fs::write(file.path(), &data)?;

        let meta = read_gguf_metadata(file.path())?;
        assert_eq!(meta.architecture.as_deref(), Some("llama"));
        Ok(())
    }

    #[test]
    fn test_read_gguf_metadata_with_chat_template() -> LlmResult<()> {
        let mut body = Vec::new();

        // Key: "tokenizer.chat_template"
        let tmpl =
            "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}";
        body.extend_from_slice(&"tokenizer.chat_template".len().to_le_bytes());
        body.extend_from_slice(b"tokenizer.chat_template");
        body.extend_from_slice(&8u32.to_le_bytes()); // string
        body.extend_from_slice(&(tmpl.len() as u64).to_le_bytes());
        body.extend_from_slice(tmpl.as_bytes());

        let data = make_gguf_file(3, 0, 1, &body);
        let file = NamedTempFile::new()?;
        fs::write(file.path(), &data)?;

        let meta = read_gguf_metadata(file.path())?;
        assert!(meta.chat_template.is_some());
        assert!(meta.chat_template.unwrap().contains("bos_token"));
        Ok(())
    }

    #[test]
    fn test_read_gguf_metadata_multiple_keys() -> LlmResult<()> {
        let mut body = Vec::new();

        // Key 1: "general.architecture" = "mistral"
        let key1 = b"general.architecture";
        body.extend_from_slice(&(key1.len() as u64).to_le_bytes()); // 21
        body.extend_from_slice(key1);
        body.extend_from_slice(&8u32.to_le_bytes());
        body.extend_from_slice(&7u64.to_le_bytes());
        body.extend_from_slice(b"mistral");

        // Key 2: "general.name" = "Mistral-7B"
        let key2 = b"general.name";
        body.extend_from_slice(&(key2.len() as u64).to_le_bytes()); // 12
        body.extend_from_slice(key2);
        body.extend_from_slice(&8u32.to_le_bytes());
        body.extend_from_slice(&10u64.to_le_bytes());
        body.extend_from_slice(b"Mistral-7B");

        let data = make_gguf_file(3, 0, 2, &body);
        let file = NamedTempFile::new()?;
        fs::write(file.path(), &data)?;

        let meta = read_gguf_metadata(file.path())?;
        assert_eq!(meta.architecture.as_deref(), Some("mistral"));
        assert_eq!(meta.name.as_deref(), Some("Mistral-7B"));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Chat template inference tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_infer_chat_template_from_metadata() {
        let meta = GgufMetadata {
            chat_template: Some("custom jinja template".into()),
            ..GgufMetadata::default()
        };
        assert_eq!(infer_chat_template(&meta), Some("custom jinja template".into()));
    }

    #[test]
    fn test_infer_chat_template_llama() {
        let meta = GgufMetadata { architecture: Some("llama".into()), ..GgufMetadata::default() };
        assert_eq!(infer_chat_template(&meta), Some("llama3".into()));
    }

    #[test]
    fn test_infer_chat_template_mistral() {
        let meta = GgufMetadata { architecture: Some("mistral".into()), ..Default::default() };
        assert_eq!(infer_chat_template(&meta), Some("chatml".into()));
    }

    #[test]
    fn test_infer_chat_template_gemma() {
        let meta = GgufMetadata { architecture: Some("gemma".into()), ..Default::default() };
        assert_eq!(infer_chat_template(&meta), Some("gemma".into()));
    }

    #[test]
    fn test_infer_chat_template_codellama() {
        let meta = GgufMetadata { architecture: Some("codellama".into()), ..Default::default() };
        assert_eq!(infer_chat_template(&meta), Some("llama3".into()));
    }

    #[test]
    fn test_infer_chat_template_deepseek() {
        let meta = GgufMetadata { architecture: Some("deepseek".into()), ..Default::default() };
        assert_eq!(infer_chat_template(&meta), Some("deepseek".into()));

        let meta = GgufMetadata { architecture: Some("deepseek3".into()), ..Default::default() };
        assert_eq!(infer_chat_template(&meta), Some("deepseek".into()));
    }

    #[test]
    fn test_infer_chat_template_unknown_returns_none() {
        let meta = GgufMetadata { architecture: Some("unknown-arch".into()), ..Default::default() };
        assert_eq!(infer_chat_template(&meta), None);
    }

    #[test]
    fn test_infer_chat_template_no_architecture_returns_none() {
        let meta = GgufMetadata::default();
        assert_eq!(infer_chat_template(&meta), None);
    }

    // -----------------------------------------------------------------------
    // Format helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_file_size() -> LlmResult<()> {
        let file = NamedTempFile::new()?;
        fs::write(file.path(), vec![0u8; 100])?;
        let result = format_file_size(file.path());
        assert!(result.contains('B'), "unexpected: {result}");
        Ok(())
    }

    #[test]
    fn test_format_param_count() {
        assert_eq!(format_param_count(0), "0");
        assert_eq!(format_param_count(1_500_000_000), "1.50B");
        assert_eq!(format_param_count(7_000_000), "7.0M");
        assert_eq!(format_param_count(3000), "3.0K");
        assert_eq!(format_param_count(42), "42");
    }

    #[test]
    fn test_format_gguf_summary() {
        let meta = GgufMetadata {
            architecture: Some("llama".into()),
            name: Some("Llama-3-8B".into()),
            file_type: Some("Q4_K_M".into()),
            context_length: Some(8192),
            chat_template: Some("jinja".into()),
            ..Default::default()
        };

        let summary = format_gguf_summary(Path::new("/test/model.gguf"), &meta);
        assert!(summary.contains("llama"));
        assert!(summary.contains("Llama-3-8B"));
        assert!(summary.contains("Q4_K_M"));
        assert!(summary.contains("8192"));
        assert!(summary.contains("embedded"));
    }
}
