use crate::*;
use chatvcode_core::{ChunkKind, ErrorKind, ErrorSeverity, FileLanguage, SourceFile};
use std::path::PathBuf;

#[test]
fn unsupported_language_returns_structured_error() {
    let file = SourceFile {
        path: PathBuf::from("README.md"),
        language: FileLanguage::Unknown,
        source_text: "# title".to_string(),
        is_large: false,
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
    assert!(language_for(FileLanguage::Python).is_some());
    assert!(language_for(FileLanguage::Php).is_some());
    assert!(language_for(FileLanguage::Unknown).is_none());
}

#[test]
fn parser_service_parse_returns_tree() {
    let mut service = ParserService::new();
    let file = SourceFile::new("test.rs", "struct Foo { x: i32 }");
    let tree = service.parse(&file).expect("should parse");

    assert_eq!(tree.root_node().kind(), "source_file");
}

#[test]
fn rust_function_chunk_extraction() {
    let code = "fn hello() { println!(\"world\"); }";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks.len(), 1);
    let chunk = &result.chunks[0];
    assert_eq!(chunk.kind, ChunkKind::Function);
    assert_eq!(chunk.symbol_name.as_deref(), Some("hello"));
    assert_eq!(chunk.source_text, code);
    assert_eq!(chunk.span.start_line, 0);
    assert_eq!(chunk.span.end_line, 0);
}

#[test]
fn rust_struct_chunk_extraction() {
    let code = "struct Point { x: f64, y: f64 }";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].kind, ChunkKind::Struct);
    assert_eq!(result.chunks[0].symbol_name.as_deref(), Some("Point"));
}

#[test]
fn rust_enum_chunk_extraction() {
    let code = "enum Color { Red, Green, Blue }";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].kind, ChunkKind::Enum);
    assert_eq!(result.chunks[0].symbol_name.as_deref(), Some("Color"));
}

#[test]
fn rust_trait_chunk_extraction() {
    let code = "trait Drawable { fn draw(&self); }";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].kind, ChunkKind::Trait);
    assert_eq!(result.chunks[0].symbol_name.as_deref(), Some("Drawable"));
}

#[test]
fn rust_impl_chunk_extraction() {
    let code = "impl Point { fn new() -> Self { Self { x: 0.0, y: 0.0 } } }";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].kind, ChunkKind::Impl);
}

#[test]
fn rust_type_alias_chunk_extraction() {
    let code = "type Result<T> = std::result::Result<T, Error>;";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].kind, ChunkKind::TypeAlias);
    assert_eq!(result.chunks[0].symbol_name.as_deref(), Some("Result"));
}

#[test]
fn rust_const_chunk_extraction() {
    let code = "const MAX_SIZE: usize = 1024;";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].kind, ChunkKind::Constant);
    assert_eq!(result.chunks[0].symbol_name.as_deref(), Some("MAX_SIZE"));
}

#[test]
fn rust_mod_chunk_extraction() {
    let code = "mod utils;";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].kind, ChunkKind::Module);
    assert_eq!(result.chunks[0].symbol_name.as_deref(), Some("utils"));
}

#[test]
fn rust_multiple_chunks() {
    let code = r"
struct Point {
    x: f64,
    y: f64,
}

fn new_point(x: f64, y: f64) -> Point {
    Point { x, y }
}

enum Shape {
    Circle(f64),
    Rectangle(f64, f64),
}
";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks.len(), 3);

    let kinds: Vec<_> = result.chunks.iter().map(|c| c.kind).collect();
    assert!(kinds.contains(&ChunkKind::Struct));
    assert!(kinds.contains(&ChunkKind::Function));
    assert!(kinds.contains(&ChunkKind::Enum));

    let struct_chunk = result
        .chunks
        .iter()
        .find(|c| c.kind == ChunkKind::Struct)
        .unwrap();
    assert_eq!(struct_chunk.symbol_name.as_deref(), Some("Point"));

    let fn_chunk = result
        .chunks
        .iter()
        .find(|c| c.kind == ChunkKind::Function)
        .unwrap();
    assert_eq!(fn_chunk.symbol_name.as_deref(), Some("new_point"));

    let enum_chunk = result
        .chunks
        .iter()
        .find(|c| c.kind == ChunkKind::Enum)
        .unwrap();
    assert_eq!(enum_chunk.symbol_name.as_deref(), Some("Shape"));
}

#[test]
fn rust_nested_function_not_extracted_separately() {
    let code = r"
fn outer() {
    fn inner() {}
}
";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].kind, ChunkKind::Function);
    assert_eq!(result.chunks[0].symbol_name.as_deref(), Some("outer"));
}

#[test]
fn rust_chunk_span_positions() {
    let code = "fn foo() {}\n\nstruct Bar { x: i32 }\n";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    let fn_chunk = result
        .chunks
        .iter()
        .find(|c| c.kind == ChunkKind::Function)
        .unwrap();
    assert_eq!(fn_chunk.span.start_line, 0);
    assert_eq!(fn_chunk.span.end_line, 0);

    let struct_chunk = result
        .chunks
        .iter()
        .find(|c| c.kind == ChunkKind::Struct)
        .unwrap();
    assert_eq!(struct_chunk.span.start_line, 2);
    assert_eq!(struct_chunk.span.end_line, 2);
}

#[test]
fn js_function_chunk_extraction() {
    let code = "function greet(name) { return 'hello ' + name; }";
    let file = SourceFile::new("test.js", code);
    let result = parse_source(file).expect("should parse");

    assert!(!result.chunks.is_empty());
    let fn_chunk = result.chunks.iter().find(|c| c.kind == ChunkKind::Function);
    assert!(fn_chunk.is_some());
    assert_eq!(fn_chunk.unwrap().symbol_name.as_deref(), Some("greet"));
}

#[test]
fn js_class_chunk_extraction() {
    let code = "class Animal { constructor(name) { this.name = name; } }";
    let file = SourceFile::new("test.js", code);
    let result = parse_source(file).expect("should parse");

    let class_chunk = result.chunks.iter().find(|c| c.kind == ChunkKind::Class);
    assert!(class_chunk.is_some());
    assert_eq!(class_chunk.unwrap().symbol_name.as_deref(), Some("Animal"));
}

#[test]
fn ts_interface_chunk_extraction() {
    let code = "interface User { name: string; age: number; }";
    let file = SourceFile::new("test.ts", code);
    let result = parse_source(file).expect("should parse");

    let iface_chunk = result
        .chunks
        .iter()
        .find(|c| c.kind == ChunkKind::Interface);
    assert!(iface_chunk.is_some());
    assert_eq!(iface_chunk.unwrap().symbol_name.as_deref(), Some("User"));
}

#[test]
fn ts_type_alias_chunk_extraction() {
    let code = "type ID = string | number;";
    let file = SourceFile::new("test.ts", code);
    let result = parse_source(file).expect("should parse");

    let type_chunk = result
        .chunks
        .iter()
        .find(|c| c.kind == ChunkKind::TypeAlias);
    assert!(type_chunk.is_some());
    assert_eq!(type_chunk.unwrap().symbol_name.as_deref(), Some("ID"));
}

#[test]
fn ts_const_chunk_extraction() {
    let code = "const API_URL = 'https://example.com';";
    let file = SourceFile::new("test.ts", code);
    let result = parse_source(file).expect("should parse");

    let const_chunk = result.chunks.iter().find(|c| c.kind == ChunkKind::Constant);
    assert!(const_chunk.is_some());
    assert_eq!(const_chunk.unwrap().symbol_name.as_deref(), Some("API_URL"));
}

#[test]
fn empty_file_no_chunks() {
    let file = SourceFile::new("empty.rs", "");
    let result = parse_source(file).expect("should parse");

    assert!(result.chunks.is_empty());
}

#[test]
fn comments_only_no_chunks() {
    let code = "// This is a comment\n/* another comment */";
    let file = SourceFile::new("comments.rs", code);
    let result = parse_source(file).expect("should parse");

    assert!(result.chunks.is_empty());
}

#[test]
fn chunk_source_text_matches_original() {
    let code = "fn main() {\n    println!(\"hello\");\n}";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].source_text, code);
}

#[test]
fn tsx_function_chunk_extraction() {
    let code = "function App() { return <div>Hello</div>; }";
    let file = SourceFile::new("app.tsx", code);
    let result = parse_source(file).expect("should parse");

    let fn_chunk = result.chunks.iter().find(|c| c.kind == ChunkKind::Function);
    assert!(fn_chunk.is_some());
    assert_eq!(fn_chunk.unwrap().symbol_name.as_deref(), Some("App"));
}

#[test]
fn rust_parser_produces_valid_ast() {
    let mut service = ParserService::new();
    let file = SourceFile::new("test.rs", "fn main() {}");
    let tree = service.parse(&file).expect("should parse");

    let root = tree.root_node();
    assert!(!root.has_error());
    assert_eq!(root.child_count(), 1);

    let func = root.child(0).unwrap();
    assert_eq!(func.kind(), "function_item");
}

#[test]
fn js_parser_produces_valid_ast() {
    let mut service = ParserService::new();
    let file = SourceFile::new("test.js", "function hello() {}");
    let tree = service.parse(&file).expect("should parse");

    let root = tree.root_node();
    assert!(!root.has_error());
}

#[test]
fn ts_parser_produces_valid_ast() {
    let mut service = ParserService::new();
    let file = SourceFile::new("test.ts", "function hello(): void {}");
    let tree = service.parse(&file).expect("should parse");

    let root = tree.root_node();
    assert!(!root.has_error());
}

#[test]
fn unsupported_language_error_includes_context() {
    let file = SourceFile {
        path: PathBuf::from("data.csv"),
        language: FileLanguage::Unknown,
        source_text: "a,b,c".to_string(),
        is_large: false,
    };

    let error = parse_source(file).expect_err("should fail");
    assert_eq!(error.kind, ErrorKind::UnsupportedLanguage);
    assert!(error.context.path.is_some());
    assert!(error.context.language.is_some());
}

#[test]
fn parser_for_language_supported() {
    assert!(parser_for_language(FileLanguage::Rust).is_ok());
    assert!(parser_for_language(FileLanguage::JavaScript).is_ok());
    assert!(parser_for_language(FileLanguage::TypeScript).is_ok());
    assert!(parser_for_language(FileLanguage::Tsx).is_ok());
    assert!(parser_for_language(FileLanguage::Jsx).is_ok());
}

#[test]
fn parser_for_language_unsupported() {
    let result = parser_for_language(FileLanguage::Unknown);
    assert!(result.is_err());
    assert_eq!(result.err().unwrap().kind, ErrorKind::UnsupportedLanguage);
}

#[test]
fn rust_chunk_file_path_preserved() {
    let file = SourceFile::new("src/main.rs", "fn main() {}");
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks[0].file_path, PathBuf::from("src/main.rs"));
}

#[test]
fn rust_chunk_language_preserved() {
    let file = SourceFile::new("lib.rs", "fn main() {}");
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks[0].language, FileLanguage::Rust);
}

#[test]
fn js_chunk_language_preserved() {
    let file = SourceFile::new("app.js", "function hello() {}");
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks[0].language, FileLanguage::JavaScript);
}

#[test]
fn tsx_chunk_language_preserved() {
    let file = SourceFile::new("app.tsx", "function App() { return <div />; }");
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks[0].language, FileLanguage::Tsx);
}

#[test]
fn js_generator_function_chunk() {
    let code = "function* gen() { yield 1; }";
    let file = SourceFile::new("test.js", code);
    let result = parse_source(file).expect("should parse");

    let fn_chunk = result.chunks.iter().find(|c| c.kind == ChunkKind::Function);
    assert!(fn_chunk.is_some());
    assert_eq!(fn_chunk.unwrap().symbol_name.as_deref(), Some("gen"));
}

#[test]
fn js_export_function_chunk() {
    let code = "export function hello() { return 42; }";
    let file = SourceFile::new("test.js", code);
    let result = parse_source(file).expect("should parse");

    let export_chunk = result.chunks.iter().find(|c| c.kind == ChunkKind::Module);
    assert!(export_chunk.is_some());
}

#[test]
fn rust_multiline_struct_span() {
    let code = "struct Point {\n    x: f64,\n    y: f64,\n}";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    let chunk = result
        .chunks
        .iter()
        .find(|c| c.kind == ChunkKind::Struct)
        .unwrap();
    assert_eq!(chunk.span.start_line, 0);
    assert_eq!(chunk.span.end_line, 3);
}

#[test]
fn rust_multiple_functions() {
    let code = "fn foo() {}\nfn bar() {}\nfn baz() {}";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    assert_eq!(result.chunks.len(), 3);
    let names: Vec<_> = result
        .chunks
        .iter()
        .filter_map(|c| c.symbol_name.as_deref())
        .collect();
    assert!(names.contains(&"foo"));
    assert!(names.contains(&"bar"));
    assert!(names.contains(&"baz"));
}

#[test]
fn js_const_with_arrow_no_chunk() {
    let code = "const add = (a, b) => a + b;";
    let file = SourceFile::new("test.js", code);
    let result = parse_source(file).expect("should parse");

    let const_chunks: Vec<_> = result
        .chunks
        .iter()
        .filter(|c| c.kind == ChunkKind::Constant)
        .collect();
    assert!(!const_chunks.is_empty());
}

#[test]
fn ts_class_with_methods() {
    let code = "class Service {\n    constructor() {}\n    async fetch() {}\n}";
    let file = SourceFile::new("test.ts", code);
    let result = parse_source(file).expect("should parse");

    let class_chunk = result.chunks.iter().find(|c| c.kind == ChunkKind::Class);
    assert!(class_chunk.is_some());
    assert_eq!(class_chunk.unwrap().symbol_name.as_deref(), Some("Service"));
}

#[test]
fn jsx_class_chunk() {
    let code = "class Component extends React.Component { render() { return <div />; } }";
    let file = SourceFile::new("test.jsx", code);
    let result = parse_source(file).expect("should parse");

    let class_chunk = result.chunks.iter().find(|c| c.kind == ChunkKind::Class);
    assert!(class_chunk.is_some());
    assert_eq!(class_chunk.unwrap().symbol_name.as_deref(), Some("Component"));
}

#[test]
fn rust_impl_with_trait() {
    let code = "impl Display for Point {\n    fn fmt(&self, f: &mut Formatter) -> Result {}\n}";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    let impl_chunk = result.chunks.iter().find(|c| c.kind == ChunkKind::Impl);
    assert!(impl_chunk.is_some());
}

#[test]
fn invalid_js_syntax_reports_errors() {
    let file = SourceFile::new("bad.js", "function function function");
    let result = parse_source(file).expect("should still return result");

    assert!(!result.errors.is_empty(), "invalid JS syntax should report errors");
}

#[test]
fn chunk_byte_span_matches_source_text_length() {
    let code = "fn hello() {}";
    let file = SourceFile::new("test.rs", code);
    let result = parse_source(file).expect("should parse");

    for chunk in &result.chunks {
        assert_eq!(chunk.source_text.len(), chunk.span.end_byte - chunk.span.start_byte);
    }
}

#[test]
fn unsupported_language_error_is_recoverable() {
    let file = SourceFile {
        path: PathBuf::from("data.csv"),
        language: FileLanguage::Unknown,
        source_text: "a,b,c".to_string(),
        is_large: false,
    };

    let error = parse_source(file).expect_err("should fail");
    assert!(error.is_recoverable());
    assert_eq!(error.severity, ErrorSeverity::Recoverable);
}

#[test]
fn parse_errors_include_path_and_language() {
    let file = SourceFile::new("bad.rs", "fn fn fn");
    let result = parse_source(file).expect("should still return result");

    for err in &result.errors {
        assert!(err.context.path.is_some());
        assert!(err.context.language.is_some());
        assert_eq!(err.context.language, Some(FileLanguage::Rust));
    }
}

#[test]
fn parser_for_language_unsupported_is_recoverable() {
    let result = parser_for_language(FileLanguage::Unknown);
    let err = result.unwrap_err();
    assert!(err.is_recoverable());
    assert_eq!(err.severity, ErrorSeverity::Recoverable);
}

#[test]
fn python_file_parses_successfully() {
    let file = SourceFile::new("test.py", "def hello():\n    print('world')\n");
    let result = parse_source(file).expect("Python should parse");

    assert_eq!(result.file.language, FileLanguage::Python);
    assert!(result.errors.is_empty(), "no parse errors expected");
}

#[test]
fn python_function_chunk_extraction() {
    let code = "def greet(name):\n    return f'Hello {name}'\n";
    let file = SourceFile::new("test.py", code);
    let result = parse_source(file).expect("should parse");

    assert!(!result.chunks.is_empty());
    let fn_chunk = result.chunks.iter().find(|c| c.kind == ChunkKind::Function);
    assert!(fn_chunk.is_some());
    assert_eq!(fn_chunk.unwrap().symbol_name.as_deref(), Some("greet"));
}

#[test]
fn python_class_chunk_extraction() {
    let code = "class Animal:\n    def __init__(self, name):\n        self.name = name\n";
    let file = SourceFile::new("test.py", code);
    let result = parse_source(file).expect("should parse");

    let class_chunk = result.chunks.iter().find(|c| c.kind == ChunkKind::Class);
    assert!(class_chunk.is_some());
    assert_eq!(class_chunk.unwrap().symbol_name.as_deref(), Some("Animal"));
}

#[test]
fn php_file_parses_successfully() {
    let file = SourceFile::new("test.php", "<?php\necho 'Hello';\n?>");
    let result = parse_source(file).expect("PHP should parse");

    assert_eq!(result.file.language, FileLanguage::Php);
}

#[test]
fn php_function_chunk_extraction() {
    let code = "<?php\nfunction greet($name) {\n    return 'Hello ' . $name;\n}\n?>";
    let file = SourceFile::new("test.php", code);
    let result = parse_source(file).expect("should parse");

    assert!(!result.chunks.is_empty());
    let fn_chunk = result.chunks.iter().find(|c| c.kind == ChunkKind::Function);
    assert!(fn_chunk.is_some());
    assert_eq!(fn_chunk.unwrap().symbol_name.as_deref(), Some("greet"));
}

#[test]
fn php_class_chunk_extraction() {
    let code = "<?php\nclass Animal {\n    public $name;\n    public function __construct($name) {\n        $this->name = $name;\n    }\n}\n?>";
    let file = SourceFile::new("test.php", code);
    let result = parse_source(file).expect("should parse");

    let class_chunk = result.chunks.iter().find(|c| c.kind == ChunkKind::Class);
    assert!(class_chunk.is_some());
    assert_eq!(class_chunk.unwrap().symbol_name.as_deref(), Some("Animal"));
}
