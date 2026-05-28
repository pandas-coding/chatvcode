use std::collections::HashSet;
use std::path::PathBuf;

use atlas_core::{ChunkKind, FileLanguage, IndexResult, index_path};
use atlas_parser::parse_source;

// Keep the real-parser fixture integration tests in `atlas-parser`.
// This exercises the full atlas-core + atlas-parser path without introducing
// an atlas-core -> atlas-parser dev-dependency cycle.
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../atlas-core/tests/fixtures")
}

fn index_fixture(name: &str) -> IndexResult {
    index_path(fixtures_dir().join(name), &parse_source).unwrap()
}

#[test]
fn indexes_rust_fixture_with_real_parser() {
    let result = index_fixture("sample.rs");

    assert_eq!(result.stats.total_files, 1);
    assert_eq!(result.stats.parsed_files, 1);
    assert!(result.errors.is_empty());
    assert!(result.files[0].errors.is_empty());
    assert_eq!(result.files[0].file.language, FileLanguage::Rust);

    let kinds: HashSet<_> = result.files[0].chunks.iter().map(|c| c.kind).collect();
    assert!(kinds.contains(&ChunkKind::Struct));
    assert!(kinds.contains(&ChunkKind::Function));
    assert!(kinds.contains(&ChunkKind::Enum));
    assert!(kinds.contains(&ChunkKind::Trait));
    assert!(kinds.contains(&ChunkKind::Impl));
    assert!(kinds.contains(&ChunkKind::Constant));
    assert!(kinds.contains(&ChunkKind::TypeAlias));
    assert!(kinds.contains(&ChunkKind::Module));

    let names: HashSet<_> = result.files[0]
        .chunks
        .iter()
        .filter_map(|c| c.symbol_name.as_deref())
        .collect();
    assert!(names.contains("Point"));
    assert!(names.contains("new_point"));
    assert!(names.contains("Shape"));
    assert!(names.contains("Drawable"));
    assert!(names.contains("MAX_SIZE"));
    assert!(names.contains("Result"));
    assert!(names.contains("utils"));
}

#[test]
fn indexes_javascript_fixture_with_real_parser() {
    let result = index_fixture("sample.js");

    assert_eq!(result.stats.total_files, 1);
    assert_eq!(result.stats.parsed_files, 1);
    assert!(result.errors.is_empty());
    assert!(result.files[0].errors.is_empty());
    assert_eq!(result.files[0].file.language, FileLanguage::JavaScript);

    let kinds: HashSet<_> = result.files[0].chunks.iter().map(|c| c.kind).collect();
    assert!(kinds.contains(&ChunkKind::Function));
    assert!(kinds.contains(&ChunkKind::Class));
    assert!(kinds.contains(&ChunkKind::Constant));

    let names: HashSet<_> = result.files[0]
        .chunks
        .iter()
        .filter_map(|c| c.symbol_name.as_deref())
        .collect();
    assert!(names.contains("greet"));
    assert!(names.contains("Animal"));
    assert!(names.contains("API_URL"));
}

#[test]
fn indexes_jsx_fixture_with_real_parser() {
    let result = index_fixture("sample.jsx");

    assert_eq!(result.stats.total_files, 1);
    assert_eq!(result.stats.parsed_files, 1);
    assert!(result.errors.is_empty());
    assert!(result.files[0].errors.is_empty());
    assert_eq!(result.files[0].file.language, FileLanguage::Jsx);

    let kinds: HashSet<_> = result.files[0].chunks.iter().map(|c| c.kind).collect();
    assert!(kinds.contains(&ChunkKind::Function));
    assert!(kinds.contains(&ChunkKind::Class));
    assert!(kinds.contains(&ChunkKind::Constant));

    let names: HashSet<_> = result.files[0]
        .chunks
        .iter()
        .filter_map(|c| c.symbol_name.as_deref())
        .collect();
    assert!(names.contains("Greeting"));
    assert!(names.contains("Button"));
    assert!(names.contains("DEFAULT_NAME"));
}

#[test]
fn indexes_typescript_fixture_with_real_parser() {
    let result = index_fixture("sample.ts");

    assert_eq!(result.stats.total_files, 1);
    assert_eq!(result.stats.parsed_files, 1);
    assert!(result.errors.is_empty());
    assert!(result.files[0].errors.is_empty());
    assert_eq!(result.files[0].file.language, FileLanguage::TypeScript);

    let kinds: HashSet<_> = result.files[0].chunks.iter().map(|c| c.kind).collect();
    assert!(kinds.contains(&ChunkKind::Interface));
    assert!(kinds.contains(&ChunkKind::TypeAlias));
    assert!(kinds.contains(&ChunkKind::Function));
    assert!(kinds.contains(&ChunkKind::Class));

    let names: HashSet<_> = result.files[0]
        .chunks
        .iter()
        .filter_map(|c| c.symbol_name.as_deref())
        .collect();
    assert!(names.contains("User"));
    assert!(names.contains("ID"));
    assert!(names.contains("getUser"));
    assert!(names.contains("Service"));
}

#[test]
fn indexes_python_fixture_with_real_parser() {
    let result = index_fixture("sample.py");

    assert_eq!(result.stats.total_files, 1);
    assert_eq!(result.stats.parsed_files, 1);
    assert!(result.errors.is_empty());
    assert!(result.files[0].errors.is_empty());
    assert_eq!(result.files[0].file.language, FileLanguage::Python);

    let kinds: HashSet<_> = result.files[0].chunks.iter().map(|c| c.kind).collect();
    assert!(kinds.contains(&ChunkKind::Function));
    assert!(kinds.contains(&ChunkKind::Class));

    let names: HashSet<_> = result.files[0]
        .chunks
        .iter()
        .filter_map(|c| c.symbol_name.as_deref())
        .collect();
    assert!(names.contains("greet"));
    assert!(names.contains("Animal"));
}

#[test]
fn indexes_php_fixture_with_real_parser() {
    let result = index_fixture("sample.php");

    assert_eq!(result.stats.total_files, 1);
    assert_eq!(result.stats.parsed_files, 1);
    assert!(result.errors.is_empty());
    assert!(result.files[0].errors.is_empty());
    assert_eq!(result.files[0].file.language, FileLanguage::Php);

    let kinds: HashSet<_> = result.files[0].chunks.iter().map(|c| c.kind).collect();
    assert!(kinds.contains(&ChunkKind::Function));
    assert!(kinds.contains(&ChunkKind::Class));
    assert!(kinds.contains(&ChunkKind::Interface));

    let names: HashSet<_> = result.files[0]
        .chunks
        .iter()
        .filter_map(|c| c.symbol_name.as_deref())
        .collect();
    assert!(names.contains("greet"));
    assert!(names.contains("Animal"));
    assert!(names.contains("Loggable"));
}

#[test]
fn indexes_fixture_directory_with_all_supported_languages() {
    let result = index_path(fixtures_dir(), &parse_source).unwrap();

    assert_eq!(result.stats.total_files, 6);
    assert_eq!(result.stats.parsed_files, 6);
    assert!(result.errors.is_empty());
    assert!(result.files.iter().all(|file| file.errors.is_empty()));
    assert!(result.stats.total_chunks >= 20);

    let languages: HashSet<_> = result.files.iter().map(|f| f.file.language).collect();
    assert!(languages.contains(&FileLanguage::Rust));
    assert!(languages.contains(&FileLanguage::JavaScript));
    assert!(languages.contains(&FileLanguage::TypeScript));
    assert!(languages.contains(&FileLanguage::Jsx));
    assert!(languages.contains(&FileLanguage::Python));
    assert!(languages.contains(&FileLanguage::Php));
}
