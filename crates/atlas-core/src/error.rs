use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Io,
    UnsupportedLanguage,
    Parse,
    InvalidInput,
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Recoverable,
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
    pub fn default_severity(self) -> ErrorSeverity {
        match self {
            Self::Io => ErrorSeverity::Recoverable,
            Self::UnsupportedLanguage => ErrorSeverity::Recoverable,
            Self::Parse => ErrorSeverity::Recoverable,
            Self::InvalidInput => ErrorSeverity::Unrecoverable,
            Self::Internal => ErrorSeverity::Unrecoverable,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ErrorContext {
    pub path: Option<PathBuf>,
    pub language: Option<crate::model::FileLanguage>,
    pub operation: Option<&'static str>,
}

impl ErrorContext {
    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn with_language(mut self, language: crate::model::FileLanguage) -> Self {
        self.language = Some(language);
        self
    }

    pub fn with_operation(mut self, operation: &'static str) -> Self {
        self.operation = Some(operation);
        self
    }
}

#[derive(Debug, Clone)]
pub struct AtlasError {
    pub kind: ErrorKind,
    pub message: String,
    pub context: ErrorContext,
    pub source: Option<String>,
    pub severity: ErrorSeverity,
}

impl AtlasError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        let severity = kind.default_severity();
        Self { kind, message: message.into(), context: ErrorContext::default(), source: None, severity }
    }

    pub fn with_context(mut self, context: ErrorContext) -> Self {
        self.context = context;
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn with_severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    pub fn is_recoverable(&self) -> bool {
        self.severity == ErrorSeverity::Recoverable
    }

    pub fn io(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Io, message)
    }

    pub fn parse(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Parse, message)
    }

    pub fn unsupported_language(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::UnsupportedLanguage, message)
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidInput, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Internal, message)
    }
}

impl fmt::Display for AtlasError {
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

impl std::error::Error for AtlasError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl From<std::io::Error> for AtlasError {
    fn from(value: std::io::Error) -> Self {
        let path_context = ErrorContext::default();
        AtlasError::io(value.to_string()).with_source(value.to_string()).with_context(path_context)
    }
}

pub type AtlasResult<T> = Result<T, AtlasError>;

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
        assert!(AtlasError::io("test").is_recoverable());
        assert!(AtlasError::parse("test").is_recoverable());
        assert!(AtlasError::unsupported_language("test").is_recoverable());
        assert!(!AtlasError::invalid_input("test").is_recoverable());
        assert!(!AtlasError::internal("test").is_recoverable());
    }

    #[test]
    fn with_severity_overrides_default() {
        let err = AtlasError::io("test").with_severity(ErrorSeverity::Unrecoverable);
        assert!(!err.is_recoverable());
        assert_eq!(err.severity, ErrorSeverity::Unrecoverable);

        let err = AtlasError::invalid_input("test").with_severity(ErrorSeverity::Recoverable);
        assert!(err.is_recoverable());
    }

    #[test]
    fn with_source_chain() {
        let err = AtlasError::io("file read failed").with_source("permission denied");
        assert_eq!(err.source, Some("permission denied".to_string()));
        let display = err.to_string();
        assert!(display.contains("caused by: permission denied"));
    }

    #[test]
    fn display_includes_context_and_source() {
        let err = AtlasError::io("read failure")
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
        let atlas_err: AtlasError = io_err.into();
        assert_eq!(atlas_err.kind, ErrorKind::Io);
        assert!(atlas_err.source.is_some());
        assert!(atlas_err.source.as_ref().unwrap().contains("file not found"));
    }

    #[test]
    fn error_severity_display() {
        assert_eq!(ErrorSeverity::Recoverable.to_string(), "recoverable");
        assert_eq!(ErrorSeverity::Unrecoverable.to_string(), "unrecoverable");
    }
}
