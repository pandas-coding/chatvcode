//! Backend initialization and GPU backend discovery.
//!
//! Wraps the low-level `llama.cpp` / `ggml` backend lifecycle:
//!
//! - [`init`] / [`shutdown`] — one-time backend setup and teardown
//! - [`supports_gpu_offload`] — runtime GPU offload capability check
//! - [`available_backends`] — enumerate registered compute backends (CPU, CUDA, ...)
//!
//! Upper-layer code (`chatvcode-core`, `chatvcode-cli`) should call [`init`]
//! once at startup and [`shutdown`] at exit. The remaining helpers are useful
//! for diagnostics and UI display.

use crate::error::cstr_to_string;
use crate::ffi;

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
}
