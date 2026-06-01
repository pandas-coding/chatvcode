use std::fmt;
use std::path::PathBuf;

/// Classification of error types in the VDB layer.
///
/// Each kind maps to a default severity via [`default_severity`](VdbErrorKind::default_severity).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VdbErrorKind {
    /// I/O error (e.g., file read/write failure).
    Io,
    /// Failed to load the ONNX model.
    ModelLoad,
    /// Failed to load the tokenizer.
    TokenizerLoad,
    /// Error during model inference.
    Inference,
    /// Invalid input (e.g., dimension mismatch, empty text).
    InvalidInput,
    /// Vector storage error (e.g., file format mismatch).
    Storage,
    /// Serialization or deserialization error.
    Serialization,
}

/// Error severity indicating whether the error is recoverable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VdbErrorSeverity {
    /// The operation can continue despite this error (e.g., per-chunk inference failure).
    Recoverable,
    /// The operation cannot continue (e.g., model file missing).
    Unrecoverable,
}

impl fmt::Display for VdbErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Recoverable => write!(f, "recoverable"),
            Self::Unrecoverable => write!(f, "unrecoverable"),
        }
    }
}

impl VdbErrorKind {
    #[must_use]
    pub const fn default_severity(self) -> VdbErrorSeverity {
        match self {
            Self::Io => VdbErrorSeverity::Recoverable,
            Self::ModelLoad => VdbErrorSeverity::Unrecoverable,
            Self::TokenizerLoad => VdbErrorSeverity::Unrecoverable,
            Self::Inference => VdbErrorSeverity::Recoverable,
            Self::InvalidInput => VdbErrorSeverity::Unrecoverable,
            Self::Storage => VdbErrorSeverity::Recoverable,
            Self::Serialization => VdbErrorSeverity::Recoverable,
        }
    }
}

/// Contextual information attached to a [`VdbError`].
///
/// Stores the file path and operation name where the error occurred,
/// which is included in the error's display output.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VdbContext {
    pub path: Option<PathBuf>,
    pub operation: Option<&'static str>,
}

impl VdbContext {
    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }

    #[must_use]
    pub const fn with_operation(mut self, operation: &'static str) -> Self {
        self.operation = Some(operation);
        self
    }
}

/// A structured error type for the VDB layer.
///
/// Includes error kind, message, context (path + operation), source chain,
/// and severity. Use the convenience constructors (`model_load`, `inference`,
/// etc.) or the builder pattern (`with_context`, `with_source`, `with_severity`)
/// to construct errors.
///
/// # Examples
///
/// ```
/// use atlas_vdb::{VdbError, VdbErrorKind, VdbContext};
///
/// let err = VdbError::model_load("Model file not found")
///     .with_context(
///         VdbContext::default()
///             .with_path("model.onnx")
///             .with_operation("model_load"),
///     );
///
/// assert_eq!(err.kind, VdbErrorKind::ModelLoad);
/// assert!(!err.is_recoverable());
/// ```
#[derive(Debug, Clone)]
pub struct VdbError {
    /// The classification of this error.
    pub kind: VdbErrorKind,
    /// A human-readable error message.
    pub message: String,
    /// Contextual information (path, operation) where the error occurred.
    pub context: VdbContext,
    /// Optional source/cause description.
    pub source: Option<String>,
    /// Whether the error is recoverable or unrecoverable.
    pub severity: VdbErrorSeverity,
}

impl VdbError {
    /// Creates a new error with the given kind and message. Severity is derived from the kind.
    pub fn new(kind: VdbErrorKind, message: impl Into<String>) -> Self {
        let severity = kind.default_severity();
        Self {
            kind,
            message: message.into(),
            context: VdbContext::default(),
            source: None,
            severity,
        }
    }

    /// Attaches context (path and operation) to this error.
    #[must_use]
    pub fn with_context(mut self, context: VdbContext) -> Self {
        self.context = context;
        self
    }

    /// Attaches a source/cause description to this error.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Overrides the default severity for this error.
    #[must_use]
    pub const fn with_severity(mut self, severity: VdbErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Returns `true` if this error is recoverable (operation may continue).
    #[must_use]
    pub fn is_recoverable(&self) -> bool {
        self.severity == VdbErrorSeverity::Recoverable
    }

    pub fn io(message: impl Into<String>) -> Self {
        Self::new(VdbErrorKind::Io, message)
    }

    pub fn model_load(message: impl Into<String>) -> Self {
        Self::new(VdbErrorKind::ModelLoad, message)
    }

    pub fn tokenizer_load(message: impl Into<String>) -> Self {
        Self::new(VdbErrorKind::TokenizerLoad, message)
    }

    pub fn inference(message: impl Into<String>) -> Self {
        Self::new(VdbErrorKind::Inference, message)
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::new(VdbErrorKind::InvalidInput, message)
    }

    pub fn storage(message: impl Into<String>) -> Self {
        Self::new(VdbErrorKind::Storage, message)
    }

    pub fn serialization(message: impl Into<String>) -> Self {
        Self::new(VdbErrorKind::Serialization, message)
    }
}

impl fmt::Display for VdbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;

        if let Some(operation) = self.context.operation {
            write!(f, " [operation={operation}]")?;
        }

        if let Some(path) = &self.context.path {
            write!(f, " [path={}]", path.display())?;
        }

        if let Some(source) = &self.source {
            write!(f, " [caused by: {source}]")?;
        }

        Ok(())
    }
}

impl std::error::Error for VdbError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl From<std::io::Error> for VdbError {
    fn from(value: std::io::Error) -> Self {
        Self::io(value.to_string()).with_source(value.to_string())
    }
}

/// Convenience type alias for `Result<T, VdbError>`.
pub type VdbResult<T> = Result<T, VdbError>;
