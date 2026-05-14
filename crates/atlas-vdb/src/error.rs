use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VdbErrorKind {
    Io,
    ModelLoad,
    TokenizerLoad,
    Inference,
    InvalidInput,
    Storage,
    Serialization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VdbErrorSeverity {
    Recoverable,
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
    pub fn default_severity(self) -> VdbErrorSeverity {
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

    pub fn with_operation(mut self, operation: &'static str) -> Self {
        self.operation = Some(operation);
        self
    }
}

#[derive(Debug, Clone)]
pub struct VdbError {
    pub kind: VdbErrorKind,
    pub message: String,
    pub context: VdbContext,
    pub source: Option<String>,
    pub severity: VdbErrorSeverity,
}

impl VdbError {
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

    pub fn with_context(mut self, context: VdbContext) -> Self {
        self.context = context;
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn with_severity(mut self, severity: VdbErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

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
        VdbError::io(value.to_string()).with_source(value.to_string())
    }
}

pub type VdbResult<T> = Result<T, VdbError>;
