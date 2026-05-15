use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Component, Path, PathBuf};

use rayon::prelude::*;

use crate::error::{AtlasError, AtlasResult, ErrorContext};
use crate::ignore;
use crate::model::{FileLanguage, SourceFile};

/// Options for configuring the file scanner.
#[derive(Debug, Clone)]
pub struct ScanOptions {
    /// The root directory to scan.
    pub root: PathBuf,
    /// Maximum directory depth to scan. None means unlimited.
    pub max_depth: Option<usize>,
    /// File size threshold in bytes above which a file is treated as "large".
    pub large_file_threshold: Option<usize>,
    /// Maximum lines to read from a large file. None means read all.
    pub large_file_max_lines: Option<usize>,
    /// Whether to respect `.gitignore` rules found during scanning.
    pub respect_gitignore: bool,
}

impl ScanOptions {
    /// Creates scan options with the given root path and no depth limit.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            max_depth: None,
            large_file_threshold: None,
            large_file_max_lines: None,
            respect_gitignore: true,
        }
    }

    /// Sets the maximum directory depth to scan.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Sets the large-file threshold in bytes.
    pub fn with_large_file_threshold(mut self, threshold: usize) -> Self {
        self.large_file_threshold = Some(threshold);
        self
    }

    /// Sets the maximum lines to read from large files.
    pub fn with_large_file_max_lines(mut self, max_lines: usize) -> Self {
        self.large_file_max_lines = Some(max_lines);
        self
    }
}

/// Result of a directory scan operation.
#[derive(Debug, Clone)]
pub struct ScanResult {
    /// Files that were successfully identified as source files.
    pub source_files: Vec<PathBuf>,
    /// Files that were skipped (unsupported language, binary, etc.).
    pub skipped_files: Vec<PathBuf>,
    /// Errors encountered during scanning.
    pub errors: Vec<AtlasError>,
}

impl ScanResult {
    /// Creates an empty scan result.
    pub fn new() -> Self {
        Self { source_files: Vec::new(), skipped_files: Vec::new(), errors: Vec::new() }
    }
}

impl Default for ScanResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Scanner that walks directories and identifies source files.
///
/// The scanner implements the following logic:
/// 1. Recursively walk through directories (using `ignore::WalkBuilder`
///    for `.gitignore` support when enabled)
/// 2. Skip ignored directories (`.git`, `target`, `node_modules`, etc.)
/// 3. Skip ignored files (binary, images, lock files, etc.)
/// 4. Identify source files by extension
/// 5. Normalize file paths
/// 6. Handle large files with partial reads
pub struct Scanner;

impl Scanner {
    /// Scans a directory and returns a list of source files that can be parsed.
    ///
    /// This is the main entry point for file discovery. It walks the directory
    /// tree, applying ignore rules and language detection. When
    /// `respect_gitignore` is enabled, it uses the `ignore` crate's
    /// `WalkBuilder` to honour `.gitignore` patterns.
    pub fn scan(options: &ScanOptions) -> ScanResult {
        let mut result = ScanResult::new();

        if !options.root.exists() {
            let err =
                AtlasError::io(format!("Root path does not exist: {}", options.root.display()))
                    .with_context(
                        ErrorContext::default()
                            .with_operation("scan")
                            .with_path(&options.root),
                    );
            log::error!("{}", err);
            result.errors.push(err);
            return result;
        }

        log::debug!("Scanning directory: {}", options.root.display());

        if !options.root.is_dir() {
            let normalized = Self::normalize_path(&options.root);
            if Self::is_source_file(&normalized) {
                result.source_files.push(normalized);
            } else {
                result.skipped_files.push(normalized);
            }
            return result;
        }

        if options.respect_gitignore {
            Self::walk_with_gitignore(&options.root, options.max_depth, &mut result);
        } else {
            Self::walk_directory(&options.root, 0, options.max_depth, &mut result);
        }
        result
    }

    /// Walks using `ignore::WalkBuilder`, which automatically reads
    /// `.gitignore` files and respects global ignore rules.
    fn walk_with_gitignore(root: &Path, max_depth: Option<usize>, result: &mut ScanResult) {
        let mut builder = ::ignore::WalkBuilder::new(root);
        builder
            .hidden(false)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true);

        if let Some(depth) = max_depth {
            builder.max_depth(Some(depth));
        }

        let walker = builder
            .filter_entry(|entry: &::ignore::DirEntry| {
                if let Some(ft) = entry.file_type()
                    && ft.is_dir()
                {
                    return !ignore::should_ignore_dir(entry.path());
                }
                true
            })
            .build();

        for entry in walker {
            match entry {
                Ok(entry) => {
                    let path = entry.path();
                    let file_type = entry.file_type();

                    match file_type {
                        Some(ft) if ft.is_dir() => {}
                        Some(ft) if ft.is_file() => {
                            let normalized = Self::normalize_path(path);
                            if ignore::should_ignore_file(&normalized) {
                                result.skipped_files.push(normalized);
                                continue;
                            }
                            if Self::is_source_file(&normalized) {
                                result.source_files.push(normalized);
                            } else {
                                result.skipped_files.push(normalized);
                            }
                        }
                        _ => {}
                    }
                }
                Err(e) => {
                    let err = AtlasError::io(format!("Walk error: {}", e)).with_context(
                        ErrorContext::default()
                            .with_operation("scan")
                            .with_path(root),
                    );
                    log::warn!("{}", err);
                    result.errors.push(err);
                }
            }
        }
    }

    /// Recursively walks a directory, collecting source files.
    fn walk_directory(
        dir: &Path,
        current_depth: usize,
        max_depth: Option<usize>,
        result: &mut ScanResult,
    ) {
        // Check depth limit
        if let Some(max) = max_depth
            && current_depth > max
        {
            return;
        }

        // Read directory entries
        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(e) => {
                let err =
                    AtlasError::io(format!("Failed to read directory {}: {e}", dir.display()))
                        .with_context(
                            ErrorContext::default()
                                .with_operation("scan")
                                .with_path(dir),
                        )
                        .with_source(e.to_string());
                log::warn!("{}", err);
                result.errors.push(err);
                return;
            }
        };

        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    let err =
                        AtlasError::io(format!("Failed to read entry in {}: {e}", dir.display()))
                            .with_context(
                                ErrorContext::default()
                                    .with_operation("scan")
                                    .with_path(dir),
                            )
                            .with_source(e.to_string());
                    log::warn!("{}", err);
                    result.errors.push(err);
                    continue;
                }
            };

            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(ft) => ft,
                Err(e) => {
                    let err = AtlasError::io(format!(
                        "Failed to get file type for {}: {e}",
                        path.display()
                    ))
                    .with_context(
                        ErrorContext::default()
                            .with_operation("scan")
                            .with_path(&path),
                    )
                    .with_source(e.to_string());
                    log::warn!("{}", err);
                    result.errors.push(err);
                    continue;
                }
            };

            if file_type.is_dir() {
                // Check if directory should be ignored
                if ignore::should_ignore_dir(&path) {
                    continue;
                }
                // Recurse into subdirectory
                Self::walk_directory(&path, current_depth + 1, max_depth, result);
            } else if file_type.is_file() {
                // Normalize the path
                let normalized = Self::normalize_path(&path);

                // Check if file should be ignored
                if ignore::should_ignore_file(&normalized) {
                    result.skipped_files.push(normalized);
                    continue;
                }

                // Check if file has a supported source extension
                if Self::is_source_file(&normalized) {
                    result.source_files.push(normalized);
                } else {
                    result.skipped_files.push(normalized);
                }
            }
            // Skip symlinks and other special files
        }
    }

    /// Checks if a file is a source file based on its extension.
    ///
    /// Returns true if the file has a recognized source code extension.
    pub fn is_source_file(path: &Path) -> bool {
        let language = FileLanguage::from_path(path);
        language.is_supported()
    }

    /// Normalizes a file path for consistent representation.
    ///
    /// This handles:
    /// - Converting to canonical path when possible
    /// - Resolving `.` and `..` components as a fallback
    /// - Standardizing path separators
    pub fn normalize_path(path: &Path) -> PathBuf {
        // Try to canonicalize the path, but fall back to manual cleanup if it fails
        // (e.g., during testing or for paths that don't exist yet)
        path.canonicalize()
            .unwrap_or_else(|_| Self::clean_path(path))
    }

    /// Cleans a path by resolving `.` and `..` components without requiring
    /// the path to exist on disk.
    fn clean_path(path: &Path) -> PathBuf {
        let mut components = Vec::new();
        for component in path.components() {
            match component {
                Component::CurDir => {
                    // Skip `.` components
                }
                Component::ParentDir => {
                    // Pop the last non-root component if possible
                    let should_pop = match components.last() {
                        Some(c) => !matches!(c, Component::RootDir | Component::Prefix(_)),
                        None => false,
                    };
                    if should_pop {
                        components.pop();
                    } else {
                        components.push(component);
                    }
                }
                other => components.push(other),
            }
        }

        let mut result = PathBuf::new();
        for component in components {
            result.push(component);
        }
        result
    }

    /// Scans a directory and reads all source files into SourceFile objects.
    ///
    /// This is a convenience method that combines scanning with file reading.
    /// Large files (exceeding the threshold) are partially read based on
    /// `large_file_max_lines`.
    pub fn scan_and_read(options: &ScanOptions) -> Vec<AtlasResult<SourceFile>> {
        let scan_result = Self::scan(options);

        let large_threshold = options.large_file_threshold.unwrap_or(0);
        let large_max_lines = options.large_file_max_lines.unwrap_or(0);

        scan_result
            .source_files
            .into_par_iter()
            .map(|path| {
                let metadata = fs::metadata(&path);
                let file_size = metadata.as_ref().map(|m| m.len()).unwrap_or(0);

                let is_large = large_threshold > 0 && file_size as usize > large_threshold;

                let source_text = if is_large && large_max_lines > 0 {
                    Self::read_first_n_lines(&path, large_max_lines)?
                } else {
                    fs::read_to_string(&path).map_err(|e| {
                        let err =
                            AtlasError::io(format!("Failed to read file {}: {e}", path.display()))
                                .with_context(
                                    ErrorContext::default()
                                        .with_operation("scan_and_read")
                                        .with_path(&path),
                                )
                                .with_source(e.to_string());
                        log::error!("{}", err);
                        err
                    })?
                };

                log::debug!("Read file: {} ({} bytes)", path.display(), source_text.len());
                let mut sf = SourceFile::new(path, source_text);
                sf.is_large = is_large;
                Ok(sf)
            })
            .collect()
    }

    /// Reads the first `n` lines from a file.
    fn read_first_n_lines(path: &Path, n: usize) -> AtlasResult<String> {
        let file = fs::File::open(path).map_err(|e| {
            let err = AtlasError::io(format!("Failed to open file {}: {e}", path.display()))
                .with_context(
                    ErrorContext::default()
                        .with_operation("read_first_n_lines")
                        .with_path(path),
                )
                .with_source(e.to_string());
            log::error!("{}", err);
            err
        })?;

        let reader = BufReader::new(file);
        let mut lines = Vec::new();
        for line in reader.lines().take(n) {
            let line = line.map_err(|e| {
                let err =
                    AtlasError::io(format!("Failed to read line from {}: {e}", path.display()))
                        .with_context(
                            ErrorContext::default()
                                .with_operation("read_first_n_lines")
                                .with_path(path),
                        )
                        .with_source(e.to_string());
                log::error!("{}", err);
                err
            })?;
            lines.push(line);
        }

        Ok(lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_structure() -> TempDir {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Create some source files
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn hello() {}").unwrap();
        fs::write(root.join("src/index.js"), "console.log('hello')").unwrap();
        fs::write(root.join("src/app.tsx"), "export default function App() {}").unwrap();

        // Create some ignored directories
        fs::create_dir_all(root.join(".git/objects")).unwrap();
        fs::write(root.join(".git/config"), "").unwrap();
        fs::create_dir_all(root.join("target/debug")).unwrap();
        fs::write(root.join("target/debug/program"), "").unwrap();
        fs::create_dir_all(root.join("node_modules/package")).unwrap();
        fs::write(root.join("node_modules/package/index.js"), "").unwrap();

        // Create some ignored files
        fs::write(root.join("image.png"), "").unwrap();
        fs::write(root.join(".gitignore"), "").unwrap();
        fs::write(root.join("Cargo.lock"), "").unwrap();

        tmp
    }

    #[test]
    fn test_scan_finds_source_files() {
        let tmp = create_test_structure();
        let options = ScanOptions::new(tmp.path());
        let result = Scanner::scan(&options);

        assert_eq!(result.source_files.len(), 4);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_scan_skips_ignored_dirs() {
        let tmp = create_test_structure();
        let options = ScanOptions::new(tmp.path());
        let result = Scanner::scan(&options);

        // Verify no files from ignored directories are included
        for file in &result.source_files {
            let path_str = file.to_string_lossy();
            assert!(!path_str.contains(".git"));
            assert!(!path_str.contains("target"));
            assert!(!path_str.contains("node_modules"));
        }
    }

    #[test]
    fn test_scan_skips_ignored_files() {
        let tmp = create_test_structure();
        let options = ScanOptions::new(tmp.path());
        let result = Scanner::scan(&options);

        // Verify ignored files are in skipped_files, not source_files
        for file in &result.skipped_files {
            let path_str = file.to_string_lossy();
            assert!(
                path_str.contains("image.png")
                    || path_str.contains(".gitignore")
                    || path_str.contains("Cargo.lock")
            );
        }
    }

    #[test]
    fn test_is_source_file() {
        assert!(Scanner::is_source_file(Path::new("main.rs")));
        assert!(Scanner::is_source_file(Path::new("index.js")));
        assert!(Scanner::is_source_file(Path::new("app.tsx")));
        assert!(!Scanner::is_source_file(Path::new("image.png")));
        assert!(!Scanner::is_source_file(Path::new("README.md")));
    }

    #[test]
    fn test_normalize_path_clean() {
        // Test the clean_path fallback (for non-existent paths)
        let path = Path::new("./src/../src/main.rs");
        let cleaned = Scanner::clean_path(path);
        assert_eq!(cleaned, PathBuf::from("src/main.rs"));
    }

    #[test]
    fn test_normalize_path() {
        let path = Path::new("./src/../src/main.rs");
        let normalized = Scanner::normalize_path(path);
        let normalized_str = normalized.to_string_lossy();
        // The normalized path should be cleaned up (no `..` components)
        assert!(!normalized_str.contains("/../"), "path still contains /../: {normalized_str}");
        assert!(!normalized_str.ends_with(".."), "path ends with '..': {normalized_str}");
    }

    #[test]
    fn test_max_depth() {
        let tmp = create_test_structure();
        let options = ScanOptions::new(tmp.path()).with_max_depth(0);
        let result = Scanner::scan(&options);

        // With max_depth=0, we should only get files in the root directory
        // (none in this test case, since all source files are in subdirectories)
        assert_eq!(result.source_files.len(), 0);
    }

    #[test]
    fn test_scan_single_file() {
        let tmp = create_test_structure();
        let file_path = tmp.path().join("src/main.rs");
        let options = ScanOptions::new(&file_path);
        let result = Scanner::scan(&options);

        assert_eq!(result.source_files.len(), 1);
        // Both sides are normalized via the same function to handle
        // platform-specific canonicalization (e.g., Windows UNC paths)
        let expected = Scanner::normalize_path(&file_path);
        assert_eq!(result.source_files[0], expected);
    }
}
