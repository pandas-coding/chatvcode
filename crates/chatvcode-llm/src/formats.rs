//! Non-GGUF model format support and conversion utilities.
//!
//! Provides detection and metadata reading for alternative model formats
//! like safetensors, along with guidance for converting to GGUF.
//!
//! # Supported Formats
//!
//! - **safetensors** — HuggingFace's safe tensor format
//! - **PyTorch (.bin/.pt)** — Legacy PyTorch format (detection only)
//!
//! # Conversion
//!
//! To use non-GGUF models with llama.cpp, they must be converted to GGUF.
//! This module provides guidance and helper functions for the conversion process.
//!
//! ## Example: Converting safetensors to GGUF
//!
//! ```bash
//! # Using llama.cpp's convert script
//! python convert_hf_to_gguf.py /path/to/model --outtype q4_K_M
//! ```

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{LlmError, LlmResult};

/// Supported model file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// GGUF format (llama.cpp native).
    Gguf,
    /// safetensors format (HuggingFace).
    SafeTensors,
    /// PyTorch format (.bin, .pt).
    PyTorch,
    /// Unknown or unsupported format.
    Unknown,
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf => write!(f, "GGUF"),
            Self::SafeTensors => write!(f, "safetensors"),
            Self::PyTorch => write!(f, "PyTorch"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Detect the format of a model file.
pub fn detect_format(path: &Path) -> ModelFormat {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    match extension.as_str() {
        "gguf" => ModelFormat::Gguf,
        "safetensors" => ModelFormat::SafeTensors,
        "bin" | "pt" | "pth" => {
            // Could be PyTorch, but verify by checking magic bytes
            if is_safetensors_file(path) {
                ModelFormat::SafeTensors
            } else {
                ModelFormat::PyTorch
            }
        }
        _ => {
            // Try to detect by magic bytes
            if is_gguf_file(path) {
                ModelFormat::Gguf
            } else if is_safetensors_file(path) {
                ModelFormat::SafeTensors
            } else {
                ModelFormat::Unknown
            }
        }
    }
}

/// Check if a file is in GGUF format.
fn is_gguf_file(path: &Path) -> bool {
    let Ok(mut file) = File::open(path) else {
        return false;
    };

    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() {
        return false;
    }

    magic == [b'G', b'G', b'U', b'F']
}

/// Check if a file is in safetensors format.
///
/// safetensors files start with an 8-byte little-endian header size,
/// followed by a JSON header.
fn is_safetensors_file(path: &Path) -> bool {
    let Ok(mut file) = File::open(path) else {
        return false;
    };

    // Read header size (8 bytes, little-endian u64)
    let mut size_bytes = [0u8; 8];
    if file.read_exact(&mut size_bytes).is_err() {
        return false;
    }

    let header_size = u64::from_le_bytes(size_bytes);

    // Sanity check: header should be reasonable size (< 100MB)
    if header_size > 100 * 1024 * 1024 {
        return false;
    }

    // Try to read and parse the JSON header
    let mut header_bytes = vec![0u8; header_size as usize];
    if file.read_exact(&mut header_bytes).is_err() {
        return false;
    }

    // Check if it's valid JSON
    serde_json::from_slice::<Value>(&header_bytes).is_ok()
}

/// Metadata from a safetensors file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTensorsMetadata {
    /// Tensor names and their shapes.
    pub tensors: Vec<TensorInfo>,
    /// Format version.
    pub format: Option<String>,
    /// Model architecture hints.
    pub architecture: Option<String>,
    /// Total parameter count.
    pub n_params: u64,
    /// File size in bytes.
    pub file_size: u64,
}

/// Information about a single tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<u64>,
    /// Data type (e.g., "F32", "F16", "BF16").
    pub dtype: String,
    /// Data offset in file.
    pub data_offsets: (u64, u64),
}

/// Read metadata from a safetensors file.
pub fn read_safetensors_metadata(path: &Path) -> LlmResult<SafeTensorsMetadata> {
    let mut file = BufReader::new(File::open(path).map_err(|e| {
        LlmError::Io(std::io::Error::new(
            e.kind(),
            format!("Failed to open {}: {e}", path.display()),
        ))
    })?);

    let file_size = file.get_ref().metadata().map(|m| m.len()).unwrap_or(0);

    // Read header size
    let mut size_bytes = [0u8; 8];
    file.read_exact(&mut size_bytes).map_err(|e| {
        LlmError::Internal(format!("Failed to read header size: {e}"))
    })?;

    let header_size = u64::from_le_bytes(size_bytes);

    // Read header JSON
    let mut header_bytes = vec![0u8; header_size as usize];
    file.read_exact(&mut header_bytes).map_err(|e| {
        LlmError::Internal(format!("Failed to read header: {e}"))
    })?;

    let header: Value = serde_json::from_slice(&header_bytes).map_err(|e| {
        LlmError::Internal(format!("Failed to parse header JSON: {e}"))
    })?;

    let mut tensors = Vec::new();
    let mut n_params: u64 = 0;

    // Parse tensor information
    if let Some(obj) = header.as_object() {
        for (name, value) in obj {
            if name == "__metadata__" {
                continue;
            }

            if let Some(tensor_obj) = value.as_object() {
                let dtype = tensor_obj
                    .get("dtype")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                let shape: Vec<u64> = tensor_obj
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                    .unwrap_or_default();

                let data_offsets = tensor_obj
                    .get("data_offsets")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| {
                        if arr.len() >= 2 {
                            Some((
                                arr[0].as_u64().unwrap_or(0),
                                arr[1].as_u64().unwrap_or(0),
                            ))
                        } else {
                            None
                        }
                    })
                    .unwrap_or((0, 0));

                // Calculate parameter count for this tensor
                let tensor_params: u64 = shape.iter().product();
                n_params += tensor_params;

                tensors.push(TensorInfo {
                    name: name.clone(),
                    shape,
                    dtype,
                    data_offsets,
                });
            }
        }
    }

    // Try to infer architecture from tensor names
    let architecture = infer_architecture(&tensors);

    // Try to get format from metadata
    let format = header
        .get("__metadata__")
        .and_then(|m| m.get("format"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Ok(SafeTensorsMetadata {
        tensors,
        format,
        architecture,
        n_params,
        file_size,
    })
}

/// Infer model architecture from tensor names.
fn infer_architecture(tensors: &[TensorInfo]) -> Option<String> {
    let names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
    let joined = names.join(" ");

    if joined.contains("llama") || joined.contains("Llama") {
        Some("llama".to_string())
    } else if joined.contains("gpt") || joined.contains("GPT") {
        Some("gpt".to_string())
    } else if joined.contains("bert") || joined.contains("BERT") {
        Some("bert".to_string())
    } else if joined.contains("qwen") || joined.contains("Qwen") {
        Some("qwen".to_string())
    } else if joined.contains("mistral") || joined.contains("Mistral") {
        Some("mistral".to_string())
    } else {
        None
    }
}

/// Conversion options for safetensors to GGUF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionOptions {
    /// Output quantization type (e.g., "q4_K_M", "f16").
    pub outtype: String,
    /// Output file path.
    pub output_path: Option<PathBuf>,
    /// Vocabulary file path (if separate).
    pub vocab_file: Option<PathBuf>,
    /// Whether to use legacy quantization.
    pub legacy_quant: bool,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            outtype: "q4_K_M".to_string(),
            output_path: None,
            vocab_file: None,
            legacy_quant: false,
        }
    }
}

/// Generate a conversion command for llama.cpp's convert script.
pub fn generate_convert_command(
    input_path: &Path,
    options: &ConversionOptions,
) -> String {
    let output = options
        .output_path
        .clone()
        .unwrap_or_else(|| input_path.with_extension("gguf"));

    let mut cmd = format!(
        "python convert_hf_to_gguf.py \"{}\" --outfile \"{}\" --outtype {}",
        input_path.display(),
        output.display(),
        options.outtype
    );

    if let Some(ref vocab) = options.vocab_file {
        cmd.push_str(&format!(" --vocab \"{}\"", vocab.display()));
    }

    if options.legacy_quant {
        cmd.push_str(" --use-alibi");
    }

    cmd
}

/// Check if a model directory contains a HuggingFace model.
pub fn is_huggingface_model_dir(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }

    // Check for config.json
    let config_path = path.join("config.json");
    if !config_path.exists() {
        return false;
    }

    // Check for model files
    let has_safetensors = path.join("model.safetensors").exists()
        || path.join("model.safetensors.index.json").exists();
    let has_pytorch = path.join("pytorch_model.bin").exists()
        || path.join("pytorch_model.bin.index.json").exists();

    has_safetensors || has_pytorch
}

/// Find all model files in a directory.
pub fn find_model_files(path: &Path) -> Vec<(PathBuf, ModelFormat)> {
    let mut files = Vec::new();

    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let file_path = entry.path();
            if file_path.is_file() {
                let format = detect_format(&file_path);
                if format != ModelFormat::Unknown {
                    files.push((file_path, format));
                }
            }
        }
    }

    files
}

/// Get conversion guidance for a model format.
pub fn conversion_guidance(format: ModelFormat) -> &'static str {
    match format {
        ModelFormat::Gguf => "This model is already in GGUF format and can be used directly.",
        ModelFormat::SafeTensors => {
            "To convert safetensors to GGUF:\n\
             1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp\n\
             2. Install requirements: pip install -r requirements.txt\n\
             3. Run conversion: python convert_hf_to_gguf.py /path/to/model\n\
             4. Use the generated .gguf file with chatvcode"
        }
        ModelFormat::PyTorch => {
            "To convert PyTorch models to GGUF:\n\
             1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp\n\
             2. Install requirements: pip install -r requirements.txt\n\
             3. Run conversion: python convert_hf_to_gguf.py /path/to/model\n\
             Note: Some PyTorch models may require specific conversion scripts."
        }
        ModelFormat::Unknown => {
            "Unknown model format. Please check the model source for conversion instructions.\n\
             chatvcode currently supports GGUF format models."
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_format_display() {
        assert_eq!(ModelFormat::Gguf.to_string(), "GGUF");
        assert_eq!(ModelFormat::SafeTensors.to_string(), "safetensors");
        assert_eq!(ModelFormat::PyTorch.to_string(), "PyTorch");
        assert_eq!(ModelFormat::Unknown.to_string(), "unknown");
    }

    #[test]
    fn test_detect_format_by_extension() {
        assert_eq!(detect_format(Path::new("model.gguf")), ModelFormat::Gguf);
        assert_eq!(detect_format(Path::new("model.safetensors")), ModelFormat::SafeTensors);
    }

    #[test]
    fn test_detect_format_unknown() {
        assert_eq!(detect_format(Path::new("model.xyz")), ModelFormat::Unknown);
    }

    #[test]
    fn test_conversion_options_default() {
        let opts = ConversionOptions::default();
        assert_eq!(opts.outtype, "q4_K_M");
        assert!(opts.output_path.is_none());
        assert!(opts.vocab_file.is_none());
        assert!(!opts.legacy_quant);
    }

    #[test]
    fn test_generate_convert_command() {
        let opts = ConversionOptions::default();
        let cmd = generate_convert_command(Path::new("/path/to/model"), &opts);
        assert!(cmd.contains("convert_hf_to_gguf.py"));
        assert!(cmd.contains("/path/to/model"));
        assert!(cmd.contains("q4_K_M"));
    }

    #[test]
    fn test_generate_convert_command_with_options() {
        let opts = ConversionOptions {
            outtype: "f16".to_string(),
            output_path: Some(PathBuf::from("/output/model.gguf")),
            vocab_file: Some(PathBuf::from("/path/vocab.json")),
            legacy_quant: true,
        };
        let cmd = generate_convert_command(Path::new("/input"), &opts);
        assert!(cmd.contains("f16"));
        assert!(cmd.contains("/output/model.gguf"));
        assert!(cmd.contains("/path/vocab.json"));
        assert!(cmd.contains("--use-alibi"));
    }

    #[test]
    fn test_conversion_guidance() {
        let gguf = conversion_guidance(ModelFormat::Gguf);
        assert!(gguf.contains("already in GGUF"));

        let safe = conversion_guidance(ModelFormat::SafeTensors);
        assert!(safe.contains("convert"));
        assert!(safe.contains("llama.cpp"));
    }

    #[test]
    fn test_infer_architecture() {
        let tensors = vec![
            TensorInfo {
                name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                shape: vec![4096, 4096],
                dtype: "F16".to_string(),
                data_offsets: (0, 0),
            },
        ];
        // This doesn't match any known pattern, so should return None
        let arch = infer_architecture(&tensors);
        assert!(arch.is_none());

        let tensors = vec![
            TensorInfo {
                name: "llama.model.layers.0.weight".to_string(),
                shape: vec![4096, 4096],
                dtype: "F16".to_string(),
                data_offsets: (0, 0),
            },
        ];
        let arch = infer_architecture(&tensors);
        assert_eq!(arch, Some("llama".to_string()));
    }

    #[test]
    fn test_is_huggingface_model_dir_nonexistent() {
        assert!(!is_huggingface_model_dir(Path::new("/nonexistent/path")));
    }

    #[test]
    fn test_find_model_files_empty_dir() {
        let files = find_model_files(Path::new("/nonexistent"));
        assert!(files.is_empty());
    }
}
