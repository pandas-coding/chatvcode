//! Error types for the chatvcode-llm crate.

use std::fmt;

/// Result type alias for LLM operations.
pub type LlmResult<T> = Result<T, LlmError>;

/// Errors that can occur during LLM operations.
#[derive(Debug)]
pub enum LlmError {
    /// Model file not found.
    ModelNotFound(String),

    /// Failed to load the model.
    ModelLoadFailed(String),

    /// Model is not loaded.
    ModelNotLoaded,

    /// Tokenization failed.
    TokenizeFailed(String),

    /// Inference / decode failed.
    InferenceFailed(String),

    /// Invalid parameter.
    InvalidParameter(String),

    /// Context overflow — input is too long.
    ContextOverflow { n_ctx: u32, n_tokens: i32 },

    /// Generation was cancelled.
    Cancelled,

    /// Unsupported model architecture or feature.
    Unsupported(String),

    /// Internal error (unexpected state).
    Internal(String),

    /// I/O error.
    Io(std::io::Error),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotFound(path) => write!(f, "model not found: {path}"),
            Self::ModelLoadFailed(msg) => write!(f, "failed to load model: {msg}"),
            Self::ModelNotLoaded => write!(f, "model is not loaded"),
            Self::TokenizeFailed(msg) => write!(f, "tokenization failed: {msg}"),
            Self::InferenceFailed(msg) => write!(f, "inference failed: {msg}"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::ContextOverflow { n_ctx, n_tokens } => {
                write!(f, "context overflow: {n_tokens} tokens exceeds context size {n_ctx}")
            }
            Self::Cancelled => write!(f, "generation cancelled"),
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl std::error::Error for LlmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for LlmError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

// ---------------------------------------------------------------------------
// Helper for converting C string results
// ---------------------------------------------------------------------------

/// Read a null-terminated C string into a Rust `String`.
///
/// # Safety
///
/// `ptr` must be a valid, null-terminated C string or null.
pub(crate) unsafe fn cstr_to_string(ptr: *const std::ffi::c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: caller guarantees ptr is a valid null-terminated C string or null
    let cstr = unsafe { std::ffi::CStr::from_ptr(ptr) };
    Some(cstr.to_string_lossy().into_owned())
}
