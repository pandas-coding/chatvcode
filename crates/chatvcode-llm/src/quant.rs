//! Quantization type information and compatibility matrix.
//!
//! Provides information about GGUF quantization types, their characteristics,
//! and compatibility with different hardware configurations.
//!
//! # Example
//!
//! ```ignore
//! use chatvcode_llm::quant::{QuantType, all_quant_types, recommend_quant};
//!
//! // Get info about a specific quantization type
//! let q4km = QuantType::from_name("Q4_K_M").unwrap();
//! println!("{}: {} bpw, quality={}", q4km.name, q4km.bits_per_weight, q4km.quality);
//!
//! // Get recommendation based on available memory
//! let rec = recommend_quant(7_000_000_000, 8 * 1024 * 1024 * 1024);
//! println!("Recommended: {}", rec.name);
//! ```

use serde::{Deserialize, Serialize};

/// GGUF quantization type information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantType {
    /// Quantization type name (e.g., "Q4_K_M").
    pub name: &'static str,
    /// Bits per weight (approximate).
    pub bits_per_weight: f32,
    /// Quality rating (1-10, higher is better).
    pub quality: u8,
    /// Speed rating (1-10, higher is faster).
    pub speed: u8,
    /// Memory usage relative to FP16 (percentage, e.g., 25 = 25% of FP16 size).
    pub memory_percent: u8,
    /// Whether this quantization supports GPU offload.
    pub gpu_supported: bool,
    /// Description of the quantization type.
    pub description: &'static str,
}

impl QuantType {
    /// Look up a quantization type by name.
    pub fn from_name(name: &str) -> Option<Self> {
        all_quant_types().into_iter().find(|q| q.name.eq_ignore_ascii_case(name))
    }

    /// Estimate the model file size for a given parameter count.
    pub fn estimate_size(&self, n_params: u64) -> u64 {
        // FP16: 2 bytes per parameter
        let fp16_size = n_params * 2;
        (fp16_size * self.memory_percent as u64) / 100
    }

    /// Format the estimated size in human-readable form.
    pub fn formatted_estimate(&self, n_params: u64) -> String {
        format_bytes(self.estimate_size(n_params))
    }
}

/// Get all supported quantization types.
pub fn all_quant_types() -> Vec<QuantType> {
    vec![
        QuantType {
            name: "F16",
            bits_per_weight: 16.0,
            quality: 10,
            speed: 5,
            memory_percent: 100,
            gpu_supported: true,
            description: "Full 16-bit precision. Highest quality but largest size.",
        },
        QuantType {
            name: "Q8_0",
            bits_per_weight: 8.5,
            quality: 9,
            speed: 7,
            memory_percent: 53,
            gpu_supported: true,
            description: "8-bit quantization. Very high quality, minimal loss from FP16.",
        },
        QuantType {
            name: "Q6_K",
            bits_per_weight: 6.5,
            quality: 8,
            speed: 7,
            memory_percent: 41,
            gpu_supported: true,
            description: "6-bit k-quant. Good balance of quality and size.",
        },
        QuantType {
            name: "Q5_K_M",
            bits_per_weight: 5.7,
            quality: 8,
            speed: 7,
            memory_percent: 36,
            gpu_supported: true,
            description: "5-bit k-quant medium. Recommended for quality-focused use.",
        },
        QuantType {
            name: "Q5_K_S",
            bits_per_weight: 5.5,
            quality: 7,
            speed: 8,
            memory_percent: 34,
            gpu_supported: true,
            description: "5-bit k-quant small. Slightly smaller than Q5_K_M.",
        },
        QuantType {
            name: "Q4_K_M",
            bits_per_weight: 4.8,
            quality: 7,
            speed: 8,
            memory_percent: 30,
            gpu_supported: true,
            description: "4-bit k-quant medium. Best balance of quality and size for most users.",
        },
        QuantType {
            name: "Q4_K_S",
            bits_per_weight: 4.6,
            quality: 6,
            speed: 9,
            memory_percent: 29,
            gpu_supported: true,
            description: "4-bit k-quant small. Smaller than Q4_K_M with slight quality loss.",
        },
        QuantType {
            name: "Q4_0",
            bits_per_weight: 4.5,
            quality: 6,
            speed: 9,
            memory_percent: 28,
            gpu_supported: true,
            description: "4-bit legacy quant. Fast but lower quality than k-quants.",
        },
        QuantType {
            name: "Q3_K_M",
            bits_per_weight: 3.9,
            quality: 5,
            speed: 8,
            memory_percent: 24,
            gpu_supported: true,
            description: "3-bit k-quant medium. For memory-constrained systems.",
        },
        QuantType {
            name: "Q3_K_S",
            bits_per_weight: 3.5,
            quality: 4,
            speed: 9,
            memory_percent: 22,
            gpu_supported: true,
            description: "3-bit k-quant small. Significant quality loss.",
        },
        QuantType {
            name: "Q2_K",
            bits_per_weight: 3.4,
            quality: 3,
            speed: 8,
            memory_percent: 21,
            gpu_supported: true,
            description: "2-bit k-quant. Severe quality loss, only for extreme memory constraints.",
        },
        QuantType {
            name: "IQ4_XS",
            bits_per_weight: 4.25,
            quality: 7,
            speed: 8,
            memory_percent: 27,
            gpu_supported: true,
            description: "4-bit importance quant. Better quality than Q4_0 at similar size.",
        },
        QuantType {
            name: "IQ3_M",
            bits_per_weight: 3.5,
            quality: 5,
            speed: 8,
            memory_percent: 22,
            gpu_supported: true,
            description: "3-bit importance quant. Better quality than Q3_K at similar size.",
        },
    ]
}

/// Recommend a quantization type based on model size and available memory.
///
/// # Arguments
///
/// * `n_params` — Number of model parameters.
/// * `available_memory` — Available system memory in bytes.
///
/// # Returns
///
/// The recommended quantization type, or `None` if no suitable type found.
pub fn recommend_quant(n_params: u64, available_memory: u64) -> Option<QuantType> {
    let types = all_quant_types();

    // Find the highest quality quant that fits in memory
    // Leave 20% headroom for KV cache and system use
    let usable_memory = (available_memory * 80) / 100;

    let mut candidates: Vec<&QuantType> = types
        .iter()
        .filter(|q| q.estimate_size(n_params) <= usable_memory)
        .collect();

    // Sort by quality (descending), then by speed (descending)
    candidates.sort_by(|a, b| {
        b.quality.cmp(&a.quality).then(b.speed.cmp(&a.speed))
    });

    candidates.first().cloned().cloned()
}

/// Get quantization types suitable for a given memory budget.
pub fn quant_for_memory(n_params: u64, available_memory: u64) -> Vec<QuantType> {
    let types = all_quant_types();
    let usable_memory = (available_memory * 80) / 100;

    let mut suitable: Vec<QuantType> = types
        .into_iter()
        .filter(|q| q.estimate_size(n_params) <= usable_memory)
        .collect();

    suitable.sort_by(|a, b| b.quality.cmp(&a.quality));
    suitable
}

/// Compare two quantization types.
pub fn compare_quant(name1: &str, name2: &str) -> Option<(QuantType, QuantType)> {
    let q1 = QuantType::from_name(name1)?;
    let q2 = QuantType::from_name(name2)?;
    Some((q1, q2))
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

/// Generate a compatibility matrix as a formatted table.
pub fn compatibility_matrix(n_params: u64) -> String {
    let types = all_quant_types();
    let mut output = String::new();

    output.push_str(&format!(
        "{:<12} {:>8} {:>8} {:>8} {:>12} {:>5}\n",
        "Type", "BPW", "Quality", "Speed", "Est. Size", "GPU"
    ));
    output.push_str(&"-".repeat(60));
    output.push('\n');

    for q in &types {
        output.push_str(&format!(
            "{:<12} {:>8.1} {:>8}/10 {:>8}/10 {:>12} {:>5}\n",
            q.name,
            q.bits_per_weight,
            q.quality,
            q.speed,
            q.formatted_estimate(n_params),
            if q.gpu_supported { "Yes" } else { "No" }
        ));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_quant_types_not_empty() {
        let types = all_quant_types();
        assert!(!types.is_empty());
        assert!(types.len() >= 10);
    }

    #[test]
    fn test_quant_type_from_name() {
        let q4km = QuantType::from_name("Q4_K_M");
        assert!(q4km.is_some());
        let q4km = q4km.unwrap();
        assert_eq!(q4km.name, "Q4_K_M");
        assert!(q4km.bits_per_weight > 4.0);
        assert!(q4km.bits_per_weight < 5.0);
    }

    #[test]
    fn test_quant_type_from_name_case_insensitive() {
        let q1 = QuantType::from_name("q4_k_m");
        let q2 = QuantType::from_name("Q4_K_M");
        assert!(q1.is_some());
        assert!(q2.is_some());
        assert_eq!(q1.unwrap().name, q2.unwrap().name);
    }

    #[test]
    fn test_quant_type_from_name_unknown() {
        let q = QuantType::from_name("UNKNOWN_QUANT");
        assert!(q.is_none());
    }

    #[test]
    fn test_estimate_size() {
        let q4km = QuantType::from_name("Q4_K_M").unwrap();
        // 7B model at Q4_K_M should be around 4GB
        let size = q4km.estimate_size(7_000_000_000);
        assert!(size > 3_000_000_000);
        assert!(size < 5_000_000_000);
    }

    #[test]
    fn test_recommend_quant() {
        // 7B model with 16GB RAM
        let rec = recommend_quant(7_000_000_000, 16 * 1024 * 1024 * 1024);
        assert!(rec.is_some());
        let rec = rec.unwrap();
        // Should recommend something with good quality
        assert!(rec.quality >= 6);
    }

    #[test]
    fn test_recommend_quant_limited_memory() {
        // 7B model with only 4GB RAM
        let rec = recommend_quant(7_000_000_000, 4 * 1024 * 1024 * 1024);
        assert!(rec.is_some());
        let rec = rec.unwrap();
        // Should recommend a smaller quant
        assert!(rec.memory_percent <= 30);
    }

    #[test]
    fn test_quant_for_memory() {
        let suitable = quant_for_memory(7_000_000_000, 8 * 1024 * 1024 * 1024);
        assert!(!suitable.is_empty());
        // Should be sorted by quality descending
        for window in suitable.windows(2) {
            assert!(window[0].quality >= window[1].quality);
        }
    }

    #[test]
    fn test_compare_quant() {
        let result = compare_quant("Q4_K_M", "Q8_0");
        assert!(result.is_some());
        let (q1, q2) = result.unwrap();
        assert_eq!(q1.name, "Q4_K_M");
        assert_eq!(q2.name, "Q8_0");
        assert!(q2.quality > q1.quality);
        assert!(q2.memory_percent > q1.memory_percent);
    }

    #[test]
    fn test_compatibility_matrix() {
        let matrix = compatibility_matrix(7_000_000_000);
        assert!(matrix.contains("Q4_K_M"));
        assert!(matrix.contains("Q8_0"));
        assert!(matrix.contains("Quality"));
    }

    #[test]
    fn test_all_quants_have_gpu_support() {
        for q in all_quant_types() {
            assert!(q.gpu_supported, "{} should support GPU", q.name);
        }
    }

    #[test]
    fn test_quality_ordering() {
        // Higher bit quants should generally have higher quality
        let q8 = QuantType::from_name("Q8_0").unwrap();
        let q4 = QuantType::from_name("Q4_K_M").unwrap();
        let q2 = QuantType::from_name("Q2_K").unwrap();
        assert!(q8.quality > q4.quality);
        assert!(q4.quality > q2.quality);
    }
}
