use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileLanguage {
    Rust,
    JavaScript,
    TypeScript,
    Jsx,
    Tsx,
    Unknown,
}

impl FileLanguage {
    pub fn from_extension(extension: &str) -> Self {
        match extension {
            "rs" => Self::Rust,
            "js" => Self::JavaScript,
            "jsx" => Self::Jsx,
            "ts" => Self::TypeScript,
            "tsx" => Self::Tsx,
            _ => Self::Unknown,
        }
    }

    pub fn all_supported() -> &'static [FileLanguage] {
        &[Self::Rust, Self::JavaScript, Self::TypeScript, Self::Jsx, Self::Tsx]
    }

    pub fn from_path(path: &Path) -> Self {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(Self::from_extension)
            .unwrap_or(Self::Unknown)
    }

    pub fn is_supported(self) -> bool {
        !matches!(self, Self::Unknown)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Jsx => "jsx",
            Self::Tsx => "tsx",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for FileLanguage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFile {
    pub path: PathBuf,
    pub language: FileLanguage,
    pub source_text: String,
}

impl SourceFile {
    pub fn new(path: impl Into<PathBuf>, source_text: impl Into<String>) -> Self {
        let path = path.into();
        let language = FileLanguage::from_path(&path);

        Self { path, language, source_text: source_text.into() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkSpan {
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
}

impl ChunkSpan {
    pub fn new(start_byte: usize, end_byte: usize, start_line: usize, end_line: usize) -> Self {
        Self { start_byte, end_byte, start_line, end_line }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Class,
    Interface,
    TypeAlias,
    Module,
    Constant,
    Method,
    Unknown,
}

impl fmt::Display for ChunkKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Function => "function",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Impl => "impl",
            Self::Class => "class",
            Self::Interface => "interface",
            Self::TypeAlias => "type_alias",
            Self::Module => "module",
            Self::Constant => "constant",
            Self::Method => "method",
            Self::Unknown => "unknown",
        };

        write!(f, "{value}")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeChunk {
    pub file_path: PathBuf,
    pub language: FileLanguage,
    pub kind: ChunkKind,
    pub symbol_name: Option<String>,
    pub span: ChunkSpan,
    pub source_text: String,
}

#[derive(Debug, Clone)]
pub struct ParseResult {
    pub file: SourceFile,
    pub chunks: Vec<CodeChunk>,
    pub errors: Vec<crate::error::AtlasError>,
}

impl ParseResult {
    pub fn success(file: SourceFile, chunks: Vec<CodeChunk>) -> Self {
        Self { file, chunks, errors: Vec::new() }
    }

    pub fn with_errors(mut self, errors: Vec<crate::error::AtlasError>) -> Self {
        self.errors = errors;
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexStats {
    pub total_files: usize,
    pub parsed_files: usize,
    pub skipped_files: usize,
    pub total_chunks: usize,
    pub total_errors: usize,
}

#[derive(Debug, Clone, Default)]
pub struct IndexResult {
    pub files: Vec<ParseResult>,
    pub errors: Vec<crate::error::AtlasError>,
    pub stats: IndexStats,
}


impl IndexResult {
    pub fn from_parse_results(
        files: Vec<ParseResult>,
        errors: Vec<crate::error::AtlasError>,
    ) -> Self {
        let parsed_files = files.len();
        let total_chunks = files.iter().map(|file| file.chunks.len()).sum();
        let file_errors: usize = files.iter().map(|file| file.errors.len()).sum();
        let total_errors = errors.len() + file_errors;
        let total_files = parsed_files + errors.len();

        Self {
            stats: IndexStats {
                total_files,
                parsed_files,
                skipped_files: total_files.saturating_sub(parsed_files),
                total_chunks,
                total_errors,
            },
            files,
            errors,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_extension_rust() {
        assert_eq!(FileLanguage::from_extension("rs"), FileLanguage::Rust);
    }

    #[test]
    fn from_extension_javascript() {
        assert_eq!(FileLanguage::from_extension("js"), FileLanguage::JavaScript);
    }

    #[test]
    fn from_extension_jsx() {
        assert_eq!(FileLanguage::from_extension("jsx"), FileLanguage::Jsx);
    }

    #[test]
    fn from_extension_typescript() {
        assert_eq!(FileLanguage::from_extension("ts"), FileLanguage::TypeScript);
    }

    #[test]
    fn from_extension_tsx() {
        assert_eq!(FileLanguage::from_extension("tsx"), FileLanguage::Tsx);
    }

    #[test]
    fn from_extension_unknown() {
        assert_eq!(FileLanguage::from_extension("py"), FileLanguage::Unknown);
        assert_eq!(FileLanguage::from_extension("java"), FileLanguage::Unknown);
        assert_eq!(FileLanguage::from_extension("go"), FileLanguage::Unknown);
        assert_eq!(FileLanguage::from_extension(""), FileLanguage::Unknown);
    }

    #[test]
    fn from_path_simple_filenames() {
        assert_eq!(FileLanguage::from_path(Path::new("main.rs")), FileLanguage::Rust);
        assert_eq!(FileLanguage::from_path(Path::new("index.js")), FileLanguage::JavaScript);
        assert_eq!(FileLanguage::from_path(Path::new("App.tsx")), FileLanguage::Tsx);
        assert_eq!(FileLanguage::from_path(Path::new("Component.jsx")), FileLanguage::Jsx);
        assert_eq!(FileLanguage::from_path(Path::new("types.ts")), FileLanguage::TypeScript);
    }

    #[test]
    fn from_path_with_directory() {
        assert_eq!(
            FileLanguage::from_path(Path::new("src/lib.rs")),
            FileLanguage::Rust
        );
        assert_eq!(
            FileLanguage::from_path(Path::new("project/src/index.js")),
            FileLanguage::JavaScript
        );
    }

    #[test]
    fn from_path_no_extension() {
        assert_eq!(FileLanguage::from_path(Path::new("Makefile")), FileLanguage::Unknown);
        assert_eq!(FileLanguage::from_path(Path::new("README")), FileLanguage::Unknown);
    }

    #[test]
    fn from_path_unsupported_extension() {
        assert_eq!(FileLanguage::from_path(Path::new("style.css")), FileLanguage::Unknown);
        assert_eq!(FileLanguage::from_path(Path::new("data.json")), FileLanguage::Unknown);
        assert_eq!(FileLanguage::from_path(Path::new("doc.md")), FileLanguage::Unknown);
    }

    #[test]
    fn is_supported() {
        assert!(FileLanguage::Rust.is_supported());
        assert!(FileLanguage::JavaScript.is_supported());
        assert!(FileLanguage::TypeScript.is_supported());
        assert!(FileLanguage::Jsx.is_supported());
        assert!(FileLanguage::Tsx.is_supported());
        assert!(!FileLanguage::Unknown.is_supported());
    }

    #[test]
    fn as_str_roundtrip() {
        for lang in FileLanguage::all_supported() {
            assert_eq!(lang.as_str(), lang.to_string());
        }
    }

    #[test]
    fn display_format() {
        assert_eq!(FileLanguage::Rust.to_string(), "rust");
        assert_eq!(FileLanguage::JavaScript.to_string(), "javascript");
        assert_eq!(FileLanguage::TypeScript.to_string(), "typescript");
        assert_eq!(FileLanguage::Jsx.to_string(), "jsx");
        assert_eq!(FileLanguage::Tsx.to_string(), "tsx");
        assert_eq!(FileLanguage::Unknown.to_string(), "unknown");
    }

    #[test]
    fn source_file_detects_language_from_path() {
        let sf = SourceFile::new("lib.rs", "fn main() {}");
        assert_eq!(sf.language, FileLanguage::Rust);

        let sf = SourceFile::new("app.tsx", "export default function App() {}");
        assert_eq!(sf.language, FileLanguage::Tsx);
    }

    #[test]
    fn source_file_unknown_language() {
        let sf = SourceFile::new("style.css", "body {}");
        assert_eq!(sf.language, FileLanguage::Unknown);
    }

    #[test]
    fn chunk_span_fields() {
        let span = ChunkSpan::new(10, 50, 2, 5);
        assert_eq!(span.start_byte, 10);
        assert_eq!(span.end_byte, 50);
        assert_eq!(span.start_line, 2);
        assert_eq!(span.end_line, 5);
    }

    #[test]
    fn chunk_kind_display() {
        assert_eq!(ChunkKind::Function.to_string(), "function");
        assert_eq!(ChunkKind::Struct.to_string(), "struct");
        assert_eq!(ChunkKind::Enum.to_string(), "enum");
        assert_eq!(ChunkKind::Trait.to_string(), "trait");
        assert_eq!(ChunkKind::Impl.to_string(), "impl");
        assert_eq!(ChunkKind::Class.to_string(), "class");
        assert_eq!(ChunkKind::Interface.to_string(), "interface");
        assert_eq!(ChunkKind::TypeAlias.to_string(), "type_alias");
        assert_eq!(ChunkKind::Module.to_string(), "module");
        assert_eq!(ChunkKind::Constant.to_string(), "constant");
        assert_eq!(ChunkKind::Method.to_string(), "method");
        assert_eq!(ChunkKind::Unknown.to_string(), "unknown");
    }

    #[test]
    fn index_stats_default() {
        let stats = IndexStats::default();
        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.parsed_files, 0);
        assert_eq!(stats.skipped_files, 0);
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_errors, 0);
    }

    #[test]
    fn index_result_from_parse_results_counts() {
        let file1 = SourceFile::new("a.rs", "fn a() {}");
        let chunk1 = CodeChunk {
            file_path: PathBuf::from("a.rs"),
            language: FileLanguage::Rust,
            kind: ChunkKind::Function,
            symbol_name: Some("a".into()),
            span: ChunkSpan::new(0, 10, 0, 0),
            source_text: "fn a() {}".into(),
        };
        let pr1 = ParseResult::success(file1, vec![chunk1]);

        let file2 = SourceFile::new("b.rs", "fn b() {}");
        let chunk2 = CodeChunk {
            file_path: PathBuf::from("b.rs"),
            language: FileLanguage::Rust,
            kind: ChunkKind::Function,
            symbol_name: Some("b".into()),
            span: ChunkSpan::new(0, 10, 0, 0),
            source_text: "fn b() {}".into(),
        };
        let pr2 = ParseResult::success(file2, vec![chunk2]);

        let result = IndexResult::from_parse_results(vec![pr1, pr2], vec![]);
        assert_eq!(result.stats.total_files, 2);
        assert_eq!(result.stats.parsed_files, 2);
        assert_eq!(result.stats.total_chunks, 2);
        assert_eq!(result.stats.total_errors, 0);
    }
}
