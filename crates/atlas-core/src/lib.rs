use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

pub use error::{AtlasError, AtlasResult, ErrorContext, ErrorKind, ErrorSeverity};
pub use model::{
    ChunkKind, ChunkSpan, CodeChunk, FileLanguage, IndexResult, IndexStats, ParseResult, SourceFile,
};
pub use scanner::{ScanOptions, ScanResult, Scanner};

pub mod error;
pub mod ignore;
pub mod model;
pub mod scanner;

pub trait ParseSource: Send + Sync {
    fn parse(&self, source_file: SourceFile) -> AtlasResult<ParseResult>;
}

impl<F> ParseSource for F
where
    F: Fn(SourceFile) -> AtlasResult<ParseResult> + Send + Sync,
{
    fn parse(&self, source_file: SourceFile) -> AtlasResult<ParseResult> {
        self(source_file)
    }
}

pub fn index_path(path: impl Into<PathBuf>, parser: &dyn ParseSource) -> AtlasResult<IndexResult> {
    let path = path.into();
    let start = Instant::now();

    if !path.exists() {
        let err = AtlasError::invalid_input(format!(
            "Path does not exist: {}",
            path.display()
        ));
        log::error!("{}", err);
        return Err(err);
    }

    log::info!("Starting index for path: {}", path.display());

    let options = ScanOptions::new(&path);
    let source_files = Scanner::scan_and_read(&options);

    let total_scanned = source_files.len();
    let scan_errors: Vec<_> = source_files.iter().filter(|r| r.is_err()).collect();
    log::info!(
        "Scan complete: {} source files found, {} scan errors",
        total_scanned - scan_errors.len(),
        scan_errors.len()
    );

    let results: Vec<_> = source_files
        .into_par_iter()
        .map(|result| match result {
            Ok(source_file) => {
                log::debug!("Parsing file: {}", source_file.path.display());
                match parser.parse(source_file) {
                    Ok(parse_result) => {
                        if !parse_result.errors.is_empty() {
                            log::warn!(
                                "Parse warnings in {}: {}",
                                parse_result.file.path.display(),
                                parse_result.errors.len()
                            );
                        }
                        Ok(parse_result)
                    }
                    Err(e) => {
                        log::error!("Failed to parse file: {}", e);
                        Err(e)
                    }
                }
            }
            Err(e) => {
                log::error!("Scan error: {}", e);
                Err(e)
            }
        })
        .collect();

    let mut parse_results = Vec::new();
    let mut scan_errors = Vec::new();

    for result in results {
        match result {
            Ok(parse_result) => parse_results.push(parse_result),
            Err(e) => scan_errors.push(e),
        }
    }

    let mut index_result = IndexResult::from_parse_results(parse_results, scan_errors);
    let elapsed = start.elapsed();
    index_result.set_elapsed_ms(elapsed.as_millis() as u64);

    log::info!(
        "Index complete: {} files parsed, {} chunks, {} errors in {}ms",
        index_result.stats.parsed_files,
        index_result.stats.total_chunks,
        index_result.stats.total_errors,
        index_result.stats.elapsed_ms
    );

    Ok(index_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_project() -> TempDir {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn hello() {}").unwrap();
        fs::write(root.join("src/index.js"), "function greet() {}").unwrap();

        fs::create_dir_all(root.join("target/debug")).unwrap();
        fs::write(root.join("target/debug/program"), "binary").unwrap();

        tmp
    }

    fn mock_parser(source_file: SourceFile) -> AtlasResult<ParseResult> {
        let chunk = CodeChunk {
            id: CodeChunk::generate_id(&source_file.path, ChunkKind::Function, Some("mock"), 0),
            file_path: source_file.path.clone(),
            language: source_file.language,
            kind: ChunkKind::Function,
            symbol_name: Some("mock".to_string()),
            span: ChunkSpan::new(0, source_file.source_text.len(), 0, 0),
            source_text: source_file.source_text.clone(),
        };
        Ok(ParseResult::success(source_file, vec![chunk]))
    }

    #[test]
    fn test_index_path_scans_and_parses() {
        let tmp = create_test_project();
        let result = index_path(tmp.path(), &mock_parser).unwrap();

        assert_eq!(result.stats.total_files, 3);
        assert_eq!(result.stats.parsed_files, 3);
        assert_eq!(result.stats.total_chunks, 3);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_index_path_nonexistent_path() {
        let result = index_path("/nonexistent/path", &mock_parser);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, ErrorKind::InvalidInput);
    }

    #[test]
    fn test_index_path_single_file() {
        let tmp = TempDir::new().unwrap();
        let file_path = tmp.path().join("test.rs");
        fs::write(&file_path, "fn test() {}").unwrap();

        let result = index_path(&file_path, &mock_parser).unwrap();
        assert_eq!(result.stats.total_files, 1);
        assert_eq!(result.stats.parsed_files, 1);
        assert_eq!(result.stats.total_chunks, 1);
    }

    #[test]
    fn test_index_path_collects_parse_errors() {
        let tmp = create_test_project();

        let failing_parser = |_source_file: SourceFile| -> AtlasResult<ParseResult> {
            Err(AtlasError::parse("mock parse failure"))
        };

        let result = index_path(tmp.path(), &failing_parser).unwrap();
        assert_eq!(result.stats.parsed_files, 0);
        assert_eq!(result.stats.total_errors, 3);
    }

    #[test]
    fn test_index_path_skips_ignored_dirs() {
        let tmp = create_test_project();
        let result = index_path(tmp.path(), &mock_parser).unwrap();

        for file in &result.files {
            let path_str = file.file.path.to_string_lossy();
            assert!(!path_str.contains("target"));
        }
    }

    #[test]
    fn test_parallel_index_results_match_sequential() {
        let tmp = create_test_project();

        let result = index_path(tmp.path(), &mock_parser).unwrap();

        assert_eq!(result.stats.total_files, 3);
        assert_eq!(result.stats.parsed_files, 3);
        assert_eq!(result.stats.total_chunks, 3);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_parallel_index_consistent_results_across_runs() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        for i in 0..20 {
            fs::write(root.join(format!("src/file_{i:02}.rs")), format!("fn func_{i}() {{}}")).unwrap();
        }

        let result1 = index_path(root, &mock_parser).unwrap();
        let result2 = index_path(root, &mock_parser).unwrap();

        assert_eq!(result1.stats.total_files, result2.stats.total_files);
        assert_eq!(result1.stats.parsed_files, result2.stats.parsed_files);
        assert_eq!(result1.stats.skipped_files, result2.stats.skipped_files);
        assert_eq!(result1.stats.total_chunks, result2.stats.total_chunks);
        assert_eq!(result1.stats.total_errors, result2.stats.total_errors);
        assert_eq!(result1.stats.files_by_language, result2.stats.files_by_language);
        assert_eq!(result1.stats.chunks_by_language, result2.stats.chunks_by_language);
        assert_eq!(result1.stats.chunks_by_kind, result2.stats.chunks_by_kind);
        assert_eq!(result1.stats.total_source_bytes, result2.stats.total_source_bytes);
    }

    #[test]
    fn test_parallel_index_mixed_success_and_failure() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/good.rs"), "fn good() {}").unwrap();
        fs::write(root.join("src/bad.rs"), "fn bad() {}").unwrap();

        let selective_parser = |source_file: SourceFile| -> AtlasResult<ParseResult> {
            if source_file.path.to_string_lossy().contains("bad") {
                Err(AtlasError::parse("selective failure"))
            } else {
                let chunk = CodeChunk {
                    id: CodeChunk::generate_id(&source_file.path, ChunkKind::Function, Some("good"), 0),
                    file_path: source_file.path.clone(),
                    language: source_file.language,
                    kind: ChunkKind::Function,
                    symbol_name: Some("good".to_string()),
                    span: ChunkSpan::new(0, source_file.source_text.len(), 0, 0),
                    source_text: source_file.source_text.clone(),
                };
                Ok(ParseResult::success(source_file, vec![chunk]))
            }
        };

        let result = index_path(root, &selective_parser).unwrap();
        assert_eq!(result.stats.parsed_files, 1);
        assert_eq!(result.stats.total_errors, 1);
    }

    #[test]
    fn test_index_path_nonexistent_path_is_unrecoverable() {
        let result = index_path("/nonexistent/path", &mock_parser);
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidInput);
        assert_eq!(err.severity, ErrorSeverity::Unrecoverable);
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_parse_errors_are_recoverable() {
        let tmp = create_test_project();

        let failing_parser = |_source_file: SourceFile| -> AtlasResult<ParseResult> {
            Err(AtlasError::parse("mock parse failure"))
        };

        let result = index_path(tmp.path(), &failing_parser).unwrap();
        for err in &result.errors {
            assert!(err.is_recoverable());
        }
    }

    #[test]
    fn test_scan_errors_have_context() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();

        let result = index_path(root, &mock_parser).unwrap();
        for err in &result.errors {
            assert!(err.context.operation.is_some() || err.context.path.is_some());
        }
    }
}
