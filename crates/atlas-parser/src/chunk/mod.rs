use atlas_core::{ChunkSpan, CodeChunk, FileLanguage, SourceFile};
use tree_sitter::Node;

mod chunk_js_ts;
mod chunk_php;
mod chunk_python;
mod chunk_rust;

/// Internal mapping from a [`ChunkKind`] to a tree-sitter node kind and name field.
pub(crate) struct ChunkDef {
    pub kind: atlas_core::ChunkKind,
    pub node_kind: &'static str,
    pub name_field: &'static str,
}

fn chunk_defs_for_language(language: FileLanguage) -> Vec<ChunkDef> {
    match language {
        FileLanguage::Rust => chunk_rust::rust_chunk_defs(),
        FileLanguage::JavaScript | FileLanguage::Jsx => chunk_js_ts::js_ts_chunk_defs(),
        FileLanguage::TypeScript | FileLanguage::Tsx => chunk_js_ts::js_ts_chunk_defs(),
        FileLanguage::Python => chunk_python::python_chunk_defs(),
        FileLanguage::Php => chunk_php::php_chunk_defs(),
        FileLanguage::Unknown => Vec::new(),
    }
}

/// Extracts all code chunks from a parsed AST.
///
/// Walks the tree recursively, matching nodes against language-specific
/// chunk definitions. Returns early on matched nodes (does not descend
/// into children of matched chunks).
pub fn extract_chunks(node: &Node, source_file: &SourceFile) -> Vec<CodeChunk> {
    let defs = chunk_defs_for_language(source_file.language);
    let mut chunks = Vec::new();
    extract_chunks_recursive(node, &defs, source_file, &mut chunks);
    chunks
}

fn extract_chunks_recursive(
    node: &Node,
    defs: &[ChunkDef],
    source_file: &SourceFile,
    chunks: &mut Vec<CodeChunk>,
) {
    if node.is_error() || node.is_missing() {
        return;
    }

    for def in defs {
        if node.kind() == def.node_kind
            && let Some(chunk) = build_chunk_from_node(node, def, source_file)
        {
            chunks.push(chunk);
            return;
        }
    }

    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            extract_chunks_recursive(&cursor.node(), defs, source_file, chunks);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }
}

fn build_chunk_from_node(
    node: &Node,
    def: &ChunkDef,
    source_file: &SourceFile,
) -> Option<CodeChunk> {
    let span = ChunkSpan::new(
        node.start_byte(),
        node.end_byte(),
        node.start_position().row,
        node.end_position().row,
    );

    let symbol_name = extract_symbol_name(node, def, source_file);

    let source_text = source_file
        .source_text
        .get(span.start_byte..span.end_byte)
        .unwrap_or_default()
        .to_string();

    if source_text.is_empty() {
        return None;
    }

    let id = CodeChunk::generate_id(
        &source_file.path,
        def.kind,
        symbol_name.as_deref(),
        span.start_line,
    );

    Some(CodeChunk {
        id,
        file_path: source_file.path.clone(),
        language: source_file.language,
        kind: def.kind,
        symbol_name,
        span,
        source_text,
    })
}

fn extract_symbol_name(node: &Node, def: &ChunkDef, source_file: &SourceFile) -> Option<String> {
    if def.name_field.is_empty() {
        return extract_name_from_node(node, source_file);
    }

    let name_node = node.child_by_field_name(def.name_field)?;
    Some(
        name_node
            .utf8_text(source_file.source_text.as_bytes())
            .ok()?
            .to_string(),
    )
}

fn extract_name_from_node(node: &Node, source_file: &SourceFile) -> Option<String> {
    let language = source_file.language;

    if matches!(
        language,
        FileLanguage::JavaScript | FileLanguage::Jsx | FileLanguage::TypeScript | FileLanguage::Tsx
    ) {
        return chunk_js_ts::extract_name_from_node(node, source_file);
    }

    if language == FileLanguage::Php {
        return chunk_php::extract_name_from_node(node, source_file);
    }

    None
}
