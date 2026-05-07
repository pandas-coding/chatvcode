use crate::*;
use atlas_core::{ErrorKind, FileLanguage, SourceFile};
use std::path::PathBuf;

#[test]
fn unsupported_language_returns_structured_error() {
    let file = SourceFile {
        path: PathBuf::from("README.md"),
        language: FileLanguage::Unknown,
        source_text: "# title".to_string(),
    };

    let error = parse_source(file).expect_err("unknown language should fail");
    assert_eq!(error.kind, ErrorKind::UnsupportedLanguage);
    assert_eq!(error.context.operation, Some("parse_source"));
}

#[test]
fn rust_file_parses_successfully() {
    let file = SourceFile::new("src/lib.rs", "fn main() { println!(\"hello\"); }");
    let result = parse_source(file).expect("Rust should parse");

    assert_eq!(result.file.language, FileLanguage::Rust);
    assert!(result.errors.is_empty(), "no parse errors expected");
}

#[test]
fn javascript_file_parses_successfully() {
    let file = SourceFile::new("index.js", "function hello() { return 42; }");
    let result = parse_source(file).expect("JavaScript should parse");

    assert_eq!(result.file.language, FileLanguage::JavaScript);
    assert!(result.errors.is_empty(), "no parse errors expected");
}

#[test]
fn typescript_file_parses_successfully() {
    let file = SourceFile::new("app.ts", "const x: number = 42;");
    let result = parse_source(file).expect("TypeScript should parse");

    assert_eq!(result.file.language, FileLanguage::TypeScript);
    assert!(result.errors.is_empty(), "no parse errors expected");
}

#[test]
fn tsx_file_parses_successfully() {
    let file = SourceFile::new("component.tsx", "export function App() { return <div />; }");
    let result = parse_source(file).expect("TSX should parse");

    assert_eq!(result.file.language, FileLanguage::Tsx);
    assert!(result.errors.is_empty(), "no parse errors expected");
}

#[test]
fn jsx_file_parses_successfully() {
    let file = SourceFile::new("component.jsx", "export function App() { return <div />; }");
    let result = parse_source(file).expect("JSX should parse");

    assert_eq!(result.file.language, FileLanguage::Jsx);
    assert!(result.errors.is_empty(), "no parse errors expected");
}

#[test]
fn invalid_rust_syntax_reports_parse_errors() {
    let file = SourceFile::new("bad.rs", "fn fn fn");
    let result = parse_source(file).expect("parsing should still return a result");

    assert_eq!(result.file.language, FileLanguage::Rust);
    assert!(!result.errors.is_empty(), "invalid syntax should produce errors");
}

#[test]
fn language_for_returns_correct_language() {
    assert!(language_for(FileLanguage::Rust).is_some());
    assert!(language_for(FileLanguage::JavaScript).is_some());
    assert!(language_for(FileLanguage::TypeScript).is_some());
    assert!(language_for(FileLanguage::Tsx).is_some());
    assert!(language_for(FileLanguage::Jsx).is_some());
    assert!(language_for(FileLanguage::Unknown).is_none());
}

#[test]
fn parser_service_parse_returns_tree() {
    let mut service = ParserService::new();
    let file = SourceFile::new("test.rs", "struct Foo { x: i32 }");
    let tree = service.parse(&file).expect("should parse");

    assert_eq!(tree.root_node().kind(), "source_file");
}
