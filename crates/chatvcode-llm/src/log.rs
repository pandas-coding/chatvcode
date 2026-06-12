//! Bridge between llama.cpp/ggml C logging and Rust's `log` crate.
//!
//! llama.cpp and ggml both write diagnostic output through a shared C callback
//! (`ggml_log_callback`). By default this callback prints everything to stderr,
//! which produces extremely noisy output during model loading (hundreds of
//! lines of `llama_model_loader`, `print_info`, `load_tensors`, `create_tensor`,
//! etc.).
//!
//! This module installs a custom callback via `llama_log_set()` (which sets
//! both the ggml-level and llama-level logger states) that:
//! - In **quiet mode** (default): forwards only `WARN` and `ERROR` messages
//!   to Rust's `log` crate at the appropriate level. `INFO`, `DEBUG`, and
//!   `CONT` messages are silently dropped.
//! - In **verbose mode** (`LlmConfig::verbose_log = true`): forwards all
//!   messages to Rust's `log` crate, preserving the original levels.
//!
//! # Thread Safety
//!
//! The callback may be invoked from multiple threads during parallel tensor
//! loading. It uses lock-free atomics and is safe to call concurrently.

use std::ffi::CStr;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ffi;

/// Controls whether verbose (INFO/DEBUG) ggml log messages are forwarded.
static GGML_VERBOSE: AtomicBool = AtomicBool::new(false);

/// Install the custom log callback and set verbosity mode.
///
/// Must be called **before** `llama_backend_init()` and any model
/// loading to ensure all C-level log output is captured.
///
/// # Arguments
///
/// * `verbose` — When `true`, forward all messages (DEBUG, INFO, WARN, ERROR).
///   When `false` (default), only forward WARN and ERROR.
pub fn setup_ggml_logging(verbose: bool) {
    GGML_VERBOSE.store(verbose, Ordering::Relaxed);

    unsafe {
        ffi::llama_log_set(Some(ggml_log_bridge), std::ptr::null_mut());
    }
}

/// The C callback installed into ggml.
///
/// This is called for every log message emitted by ggml and llama.cpp.
/// We inspect the log level and decide whether to forward the message
/// to Rust's `log` crate.
unsafe extern "C" fn ggml_log_bridge(
    level: ffi::GgmlLogLevel,
    text: *const std::ffi::c_char,
    _user_data: *mut std::ffi::c_void,
) {
    let text_str = if text.is_null() {
        "(null)"
    } else {
        // Use `CStr::from_ptr` which is the safe-ish way in C callback context
        unsafe { CStr::from_ptr(text) }
            .to_str()
            .unwrap_or("(invalid utf-8)")
    };

    let text_trimmed = text_str.trim_end();

    if text_trimmed.is_empty() {
        return;
    }

    let verbose = GGML_VERBOSE.load(Ordering::Relaxed);

    match level {
        ffi::GgmlLogLevel::Error => {
            log::error!("[ggml] {}", text_trimmed);
        }
        ffi::GgmlLogLevel::Warn => {
            log::warn!("[ggml] {}", text_trimmed);
        }
        ffi::GgmlLogLevel::Info => {
            if verbose {
                log::info!("[ggml] {}", text_trimmed);
            }
        }
        ffi::GgmlLogLevel::Debug => {
            if verbose {
                log::debug!("[ggml] {}", text_trimmed);
            }
        }
        ffi::GgmlLogLevel::Cont => {
            // CONT ("continue previous log") messages only make sense when
            // INFO/DEBUG are enabled, so only forward in verbose mode.
            if verbose {
                log::info!("[ggml] {}", text_trimmed);
            }
        }
        ffi::GgmlLogLevel::None => {
            // NONE — typically not emitted, ignore unless verbose.
            if verbose {
                log::trace!("[ggml] {}", text_trimmed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_setup_ggml_logging_quiet() {
        // Should not panic
        setup_ggml_logging(false);
        assert!(!GGML_VERBOSE.load(Ordering::Relaxed));
    }

    #[test]
    fn test_setup_ggml_logging_verbose() {
        setup_ggml_logging(true);
        assert!(GGML_VERBOSE.load(Ordering::Relaxed));
        // Reset to quiet
        setup_ggml_logging(false);
    }
}
