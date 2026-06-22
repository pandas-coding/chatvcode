//! Backend initialization and GPU backend discovery.
//!
//! Wraps the low-level `llama.cpp` / `ggml` backend lifecycle:
//!
//! - [`init`] / [`shutdown`] — one-time backend setup and teardown
//! - [`supports_gpu_offload`] — runtime GPU offload capability check
//! - [`available_backends`] — enumerate registered compute backends (CPU, CUDA, ...)
//! - [`detect_gpu_acceleration`] — auto-detect available GPU acceleration
//! - [`recommend_gpu_config`] — recommend optimal GPU configuration
//!
//! Upper-layer code (`chatvcode-core`, `chatvcode-cli`) should call [`init`]
//! once at startup and [`shutdown`] at exit. The remaining helpers are useful
//! for diagnostics and UI display.

use crate::ffi;
use serde::{Deserialize, Serialize};

/// Registered backend information discovered from ggml/llama.cpp.
///
/// Each entry describes one compute backend (e.g. "CPU", "CUDA") together
/// with the list of devices it exposes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendInfo {
    /// Backend name as reported by ggml (e.g. "CPU", "CUDA", "Vulkan").
    pub name: String,
    /// Human-readable device descriptions (e.g. "NVIDIA GeForce RTX 4090").
    pub devices: Vec<String>,
}

/// Detected GPU acceleration type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuAcceleration {
    /// No GPU acceleration available.
    None,
    /// NVIDIA CUDA acceleration.
    Cuda,
    /// AMD/NVIDIA Vulkan acceleration.
    Vulkan,
    /// Apple Metal acceleration (macOS only).
    Metal,
    /// Multiple GPU backends available.
    Multiple,
}

impl std::fmt::Display for GpuAcceleration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Cuda => write!(f, "CUDA"),
            Self::Vulkan => write!(f, "Vulkan"),
            Self::Metal => write!(f, "Metal"),
            Self::Multiple => write!(f, "multiple"),
        }
    }
}

/// GPU device information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// Device name (e.g., "NVIDIA GeForce RTX 4090").
    pub name: String,
    /// Backend type (CUDA, Vulkan, Metal).
    pub backend: String,
    /// Total VRAM in bytes (if available).
    pub vram_bytes: Option<u64>,
    /// Device index within the backend.
    pub device_index: usize,
}

/// GPU configuration recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfigRecommendation {
    /// Recommended number of GPU layers to offload.
    pub n_gpu_layers: i32,
    /// Explanation of the recommendation.
    pub reason: String,
    /// Estimated VRAM usage in bytes.
    pub estimated_vram_bytes: u64,
    /// Whether full offload is possible.
    pub full_offload_possible: bool,
    /// Detected GPU acceleration type.
    pub acceleration: GpuAcceleration,
    /// Available GPU devices.
    pub devices: Vec<GpuDeviceInfo>,
}

/// Initialize the llama.cpp backend.
///
/// Must be called once before any other LLM operations.
/// It is safe to call this multiple times (idempotent).
pub fn init() {
    crate::log::setup_ggml_logging(false);
    unsafe { ffi::llama_backend_init() };
}

/// Shut down the llama.cpp backend.
///
/// Should be called once when the program exits.
pub fn shutdown() {
    unsafe { ffi::llama_backend_free() };
}

/// Returns whether llama.cpp reports GPU offload support at runtime.
#[must_use]
pub fn supports_gpu_offload() -> bool {
    unsafe { ffi::llama_supports_gpu_offload() }
}

/// Enumerate registered ggml backends and their visible devices.
///
/// Returns one [`BackendInfo`] per registered backend. The CPU backend is
/// always present; GPU backends (CUDA, Vulkan, Metal) appear only when the
/// corresponding feature flag was enabled at compile time.
#[must_use]
pub fn available_backends() -> Vec<BackendInfo> {
    let mut backends = Vec::new();
    let count = unsafe { ffi::ggml_backend_reg_count() };

    for i in 0..count {
        let reg = unsafe { ffi::ggml_backend_reg_get(i) };
        if reg.is_null() {
            continue;
        }

        let name = unsafe { cstr_to_string(ffi::ggml_backend_reg_name(reg)) }
            .unwrap_or_else(|| "unknown".to_string());

        let mut devices = Vec::new();
        let dev_count = unsafe { ffi::ggml_backend_reg_dev_count(reg) };
        for j in 0..dev_count {
            let dev = unsafe { ffi::ggml_backend_reg_dev_get(reg, j) };
            if dev.is_null() {
                continue;
            }

            let dev_name = unsafe { cstr_to_string(ffi::ggml_backend_dev_name(dev)) }
                .unwrap_or_else(|| "unknown".to_string());
            let dev_desc = unsafe { cstr_to_string(ffi::ggml_backend_dev_description(dev)) }
                .unwrap_or_default();

            if dev_desc.is_empty() || dev_desc == dev_name {
                devices.push(dev_name);
            } else {
                devices.push(format!("{dev_name} ({dev_desc})"));
            }
        }

        backends.push(BackendInfo { name, devices });
    }

    backends
}

/// Detect available GPU acceleration.
///
/// Examines the registered backends and returns the best available
/// GPU acceleration type. Returns `GpuAcceleration::None` if only
/// CPU is available.
#[must_use]
pub fn detect_gpu_acceleration() -> GpuAcceleration {
    let backends = available_backends();
    let mut has_cuda = false;
    let mut has_vulkan = false;
    let mut has_metal = false;

    for backend in &backends {
        let name_lower = backend.name.to_lowercase();
        if name_lower.contains("cuda") {
            has_cuda = true;
        } else if name_lower.contains("vulkan") {
            has_vulkan = true;
        } else if name_lower.contains("metal") {
            has_metal = true;
        }
    }

    let gpu_count = has_cuda as u8 + has_vulkan as u8 + has_metal as u8;
    if gpu_count > 1 {
        GpuAcceleration::Multiple
    } else if has_cuda {
        GpuAcceleration::Cuda
    } else if has_vulkan {
        GpuAcceleration::Vulkan
    } else if has_metal {
        GpuAcceleration::Metal
    } else {
        GpuAcceleration::None
    }
}

/// List all available GPU devices.
#[must_use]
pub fn list_gpu_devices() -> Vec<GpuDeviceInfo> {
    let backends = available_backends();
    let mut devices = Vec::new();
    let mut device_index = 0;

    for backend in &backends {
        let name_lower = backend.name.to_lowercase();
        if name_lower.contains("cpu") {
            continue;
        }

        for device_name in &backend.devices {
            devices.push(GpuDeviceInfo {
                name: device_name.clone(),
                backend: backend.name.clone(),
                vram_bytes: None, // VRAM detection requires backend-specific APIs
                device_index,
            });
            device_index += 1;
        }
    }

    devices
}

/// Recommend GPU configuration based on model size and available hardware.
///
/// # Arguments
///
/// * `model_size_bytes` — Estimated model size in bytes.
/// * `n_layers` — Number of transformer layers in the model.
/// * `available_vram_bytes` — Available VRAM in bytes (0 = auto-detect).
///
/// # Returns
///
/// A [`GpuConfigRecommendation`] with optimal settings.
#[must_use]
pub fn recommend_gpu_config(
    model_size_bytes: u64,
    n_layers: i32,
    available_vram_bytes: u64,
) -> GpuConfigRecommendation {
    let acceleration = detect_gpu_acceleration();
    let devices = list_gpu_devices();

    if acceleration == GpuAcceleration::None {
        return GpuConfigRecommendation {
            n_gpu_layers: 0,
            reason: "No GPU acceleration available".to_string(),
            estimated_vram_bytes: 0,
            full_offload_possible: false,
            acceleration,
            devices,
        };
    }

    // Estimate VRAM per layer (rough heuristic)
    // Each layer typically requires: model_size / n_layers + overhead
    let overhead_per_layer = 100 * 1024 * 1024; // 100MB overhead per layer for KV cache
    let model_per_layer = if n_layers > 0 { model_size_bytes / n_layers as u64 } else { 0 };
    let vram_per_layer = model_per_layer + overhead_per_layer;

    // Determine available VRAM
    let effective_vram = if available_vram_bytes > 0 {
        available_vram_bytes
    } else {
        // Try to estimate from devices
        devices.first().and_then(|d| d.vram_bytes).unwrap_or(8 * 1024 * 1024 * 1024) // Default 8GB
    };

    // Reserve some VRAM for system use
    let usable_vram = effective_vram.saturating_sub(512 * 1024 * 1024);

    // Calculate max layers that fit
    let max_layers = if vram_per_layer > 0 {
        (usable_vram / vram_per_layer) as i32
    } else {
        0
    };

    let recommended_layers = max_layers.min(n_layers).max(0);
    let full_offload = recommended_layers >= n_layers;

    let reason = if full_offload {
        format!(
            "Full offload possible: {} layers fit in {} VRAM",
            n_layers,
            format_bytes(effective_vram)
        )
    } else if recommended_layers > 0 {
        format!(
            "Partial offload: {} of {} layers fit in {} VRAM",
            recommended_layers,
            n_layers,
            format_bytes(effective_vram)
        )
    } else {
        "Insufficient VRAM for GPU offload".to_string()
    };

    let estimated_vram = if recommended_layers > 0 {
        vram_per_layer * recommended_layers as u64
    } else {
        0
    };

    GpuConfigRecommendation {
        n_gpu_layers: recommended_layers,
        reason,
        estimated_vram_bytes: estimated_vram,
        full_offload_possible: full_offload,
        acceleration,
        devices,
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

/// Helper to convert C string to Rust String.
fn cstr_to_string(ptr: *const std::ffi::c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_str()
        .ok()
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_shutdown_idempotent() {
        init();
        init();
        shutdown();
        shutdown();
    }

    #[test]
    fn test_available_backends() {
        init();

        let backends = available_backends();
        assert!(!backends.is_empty(), "no ggml backends registered");
        assert!(backends.iter().any(|b| b.name.eq_ignore_ascii_case("cpu")));

        if std::env::var("LLAMA_CUDA").ok().as_deref() == Some("1") {
            assert!(
                backends.iter().any(|b| b.name.eq_ignore_ascii_case("cuda")),
                "CUDA requested but backend not registered: {backends:#?}"
            );
        }

        if std::env::var("LLAMA_VULKAN").ok().as_deref() == Some("1") {
            assert!(
                backends
                    .iter()
                    .any(|b| b.name.eq_ignore_ascii_case("vulkan")),
                "Vulkan requested but backend not registered: {backends:#?}"
            );
        }
    }

    #[test]
    fn test_detect_gpu_acceleration() {
        init();
        let accel = detect_gpu_acceleration();
        // Should return some value (None if no GPU, or specific type)
        let _ = accel.to_string();
    }

    #[test]
    fn test_list_gpu_devices() {
        init();
        let devices = list_gpu_devices();
        // May be empty on CPU-only systems
        for device in &devices {
            assert!(!device.name.is_empty());
            assert!(!device.backend.is_empty());
        }
    }

    #[test]
    fn test_recommend_gpu_config_no_gpu() {
        // Test with a model that would need GPU
        let rec = recommend_gpu_config(4_000_000_000, 32, 0);
        // Should return a valid recommendation
        assert!(rec.n_gpu_layers >= 0);
        assert!(!rec.reason.is_empty());
    }

    #[test]
    fn test_recommend_gpu_config_with_vram() {
        init();
        // 4GB model, 32 layers, 16GB VRAM
        let rec = recommend_gpu_config(4_000_000_000, 32, 16 * 1024 * 1024 * 1024);
        assert!(rec.n_gpu_layers > 0 || rec.acceleration == GpuAcceleration::None);
    }

    #[test]
    fn test_gpu_acceleration_display() {
        assert_eq!(GpuAcceleration::None.to_string(), "none");
        assert_eq!(GpuAcceleration::Cuda.to_string(), "CUDA");
        assert_eq!(GpuAcceleration::Vulkan.to_string(), "Vulkan");
        assert_eq!(GpuAcceleration::Metal.to_string(), "Metal");
        assert_eq!(GpuAcceleration::Multiple.to_string(), "multiple");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_bytes(8 * 1024 * 1024 * 1024), "8.0 GB");
    }
}
