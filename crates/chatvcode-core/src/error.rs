use std::fmt;
use std::path::PathBuf;

/// Classification of error types.
///
/// Each variant represents a distinct category of failure. The default
/// severity for each kind is defined in [`ErrorKind::default_severity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// I/O errors (file not found, permission denied, etc.).
    Io,
    /// Attempted to process an unsupported language.
    UnsupportedLanguage,
    /// Syntax or parsing errors.
    Parse,
    /// Invalid user input (e.g., nonexistent path).
    InvalidInput,
    /// Unexpected internal errors.
    Internal,
}

/// Severity level indicating whether an error is recoverable.
///
/// Recoverable errors (e.g., a single file failing to parse) allow the
/// pipeline to continue processing other files. Unrecoverable errors
/// (e.g., invalid input path) halt the entire operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// The operation can continue despite this error.
    Recoverable,
    /// The operation must be aborted.
    Unrecoverable,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Recoverable => write!(f, "recoverable"),
            Self::Unrecoverable => write!(f, "unrecoverable"),
        }
    }
}

impl ErrorKind {
    /// Returns the default severity for this error kind.
    #[must_use]
    pub const fn default_severity(self) -> ErrorSeverity {
        match self {
            Self::Io => ErrorSeverity::Recoverable,
            Self::UnsupportedLanguage => ErrorSeverity::Recoverable,
            Self::Parse => ErrorSeverity::Recoverable,
            Self::InvalidInput => ErrorSeverity::Unrecoverable,
            Self::Internal => ErrorSeverity::Unrecoverable,
        }
    }
}

/// Contextual information attached to an error.
///
/// Provides optional metadata (file path, language, operation) to aid
/// debugging and error reporting. Uses a builder pattern for construction.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ErrorContext {
    /// File path associated with the error, if applicable.
    pub path: Option<PathBuf>,
    /// Programming language associated with the error, if applicable.
    pub language: Option<crate::model::FileLanguage>,
    /// Name of the operation that produced the error.
    pub operation: Option<&'static str>,
}

impl ErrorContext {
    /// Attaches a file path to this context.
    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Attaches a language to this context.
    #[must_use]
    pub const fn with_language(mut self, language: crate::model::FileLanguage) -> Self {
        self.language = Some(language);
        self
    }

    /// Attaches an operation name to this context.
    #[must_use]
    pub const fn with_operation(mut self, operation: &'static str) -> Self {
        self.operation = Some(operation);
        self
    }
}

/// A structured error with context and severity information.
///
/// `ChatVCodeError` is the primary error type used throughout the ChatVCode pipeline.
/// It supports optional context (file path, language, operation), an optional
/// source error chain, and configurable severity.
#[derive(Debug, Clone)]
pub struct ChatVCodeError {
    /// The category of this error.
    pub kind: ErrorKind,
    /// Human-readable error message.
    pub message: String,
    /// Optional contextual metadata.
    pub context: ErrorContext,
    /// Optional source error description (for error chaining).
    pub source: Option<String>,
    /// Whether this error is recoverable or fatal.
    pub severity: ErrorSeverity,
}

impl ChatVCodeError {
    /// Creates a new error with the given kind and message.
    ///
    /// Severity is set to the default for the error kind.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        let severity = kind.default_severity();
        Self {
            kind,
            message: message.into(),
            context: ErrorContext::default(),
            source: None,
            severity,
        }
    }

    /// Attaches contextual metadata to this error.
    #[must_use]
    pub fn with_context(mut self, context: ErrorContext) -> Self {
        self.context = context;
        self
    }

    /// Attaches a source error description for error chaining.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Overrides the default severity for this error.
    #[must_use]
    pub const fn with_severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Returns `true` if this error is recoverable (operation can continue).
    #[must_use]
    pub fn is_recoverable(&self) -> bool {
        self.severity == ErrorSeverity::Recoverable
    }

    /// Creates an I/O error with the given message.
    pub fn io(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Io, message)
    }

    /// Creates a parse error with the given message.
    pub fn parse(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Parse, message)
    }

    /// Creates an unsupported language error with the given message.
    pub fn unsupported_language(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::UnsupportedLanguage, message)
    }

    /// Creates an invalid input error with the given message.
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidInput, message)
    }

    /// Creates an internal error with the given message.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Internal, message)
    }
}

impl fmt::Display for ChatVCodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;

        if let Some(operation) = self.context.operation {
            write!(f, " [operation={operation}]")?;
        }

        if let Some(path) = &self.context.path {
            write!(f, " [path={}]", path.display())?;
        }

        if let Some(language) = self.context.language {
            write!(f, " [language={language}]")?;
        }

        if let Some(source) = &self.source {
            write!(f, " [caused by: {source}]")?;
        }

        Ok(())
    }
}

impl std::error::Error for ChatVCodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl From<std::io::Error> for ChatVCodeError {
    fn from(value: std::io::Error) -> Self {
        let path_context = ErrorContext::default();
        Self::io(value.to_string())
            .with_source(value.to_string())
            .with_context(path_context)
    }
}

/// Convenience type alias for `Result<T, ChatVCodeError>`.
pub type ChatVCodeResult<T> = Result<T, ChatVCodeError>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FileLanguage;

    #[test]
    fn error_kind_default_severity() {
        assert_eq!(ErrorKind::Io.default_severity(), ErrorSeverity::Recoverable);
        assert_eq!(ErrorKind::UnsupportedLanguage.default_severity(), ErrorSeverity::Recoverable);
        assert_eq!(ErrorKind::Parse.default_severity(), ErrorSeverity::Recoverable);
        assert_eq!(ErrorKind::InvalidInput.default_severity(), ErrorSeverity::Unrecoverable);
        assert_eq!(ErrorKind::Internal.default_severity(), ErrorSeverity::Unrecoverable);
    }

    #[test]
    fn is_recoverable() {
        assert!(ChatVCodeError::io("test").is_recoverable());
        assert!(ChatVCodeError::parse("test").is_recoverable());
        assert!(ChatVCodeError::unsupported_language("test").is_recoverable());
        assert!(!ChatVCodeError::invalid_input("test").is_recoverable());
        assert!(!ChatVCodeError::internal("test").is_recoverable());
    }

    #[test]
    fn with_severity_overrides_default() {
        let err = ChatVCodeError::io("test").with_severity(ErrorSeverity::Unrecoverable);
        assert!(!err.is_recoverable());
        assert_eq!(err.severity, ErrorSeverity::Unrecoverable);

        let err = ChatVCodeError::invalid_input("test").with_severity(ErrorSeverity::Recoverable);
        assert!(err.is_recoverable());
    }

    #[test]
    fn with_source_chain() {
        let err = ChatVCodeError::io("file read failed").with_source("permission denied");
        assert_eq!(err.source, Some("permission denied".to_string()));
        let display = err.to_string();
        assert!(display.contains("caused by: permission denied"));
    }

    #[test]
    fn display_includes_context_and_source() {
        let err = ChatVCodeError::io("read failure")
            .with_context(
                ErrorContext::default()
                    .with_operation("scan")
                    .with_path("/project/src/main.rs")
                    .with_language(FileLanguage::Rust),
            )
            .with_source("No such file");
        let display = err.to_string();
        assert!(display.contains("read failure"));
        assert!(display.contains("[operation=scan]"));
        assert!(display.contains("[path=/project/src/main.rs]"));
        assert!(display.contains("[language=rust]"));
        assert!(display.contains("[caused by: No such file]"));
    }

    #[test]
    fn from_io_error_preserves_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let chatvcode_err: ChatVCodeError = io_err.into();
        assert_eq!(chatvcode_err.kind, ErrorKind::Io);
        assert!(chatvcode_err.source.is_some());
        assert!(
            chatvcode_err
                .source
                .as_ref()
                .unwrap()
                .contains("file not found")
        );
    }

    #[test]
    fn error_severity_display() {
        assert_eq!(ErrorSeverity::Recoverable.to_string(), "recoverable");
        assert_eq!(ErrorSeverity::Unrecoverable.to_string(), "unrecoverable");
    }
}
