use atlas_core::{ChunkKind, ChunkSpan, CodeChunk, FileLanguage, SourceFile};
use tree_sitter::Node;

struct ChunkDef {
    kind: ChunkKind,
    node_kind: &'static str,
    name_field: &'static str,
}

fn rust_chunk_defs() -> Vec<ChunkDef> {
    vec![
        ChunkDef { kind: ChunkKind::Function, node_kind: "function_item", name_field: "name" },
        ChunkDef { kind: ChunkKind::Struct, node_kind: "struct_item", name_field: "name" },
        ChunkDef { kind: ChunkKind::Enum, node_kind: "enum_item", name_field: "name" },
        ChunkDef { kind: ChunkKind::Trait, node_kind: "trait_item", name_field: "name" },
        ChunkDef { kind: ChunkKind::Impl, node_kind: "impl_item", name_field: "trait" },
        ChunkDef { kind: ChunkKind::TypeAlias, node_kind: "type_item", name_field: "name" },
        ChunkDef { kind: ChunkKind::Constant, node_kind: "const_item", name_field: "name" },
        ChunkDef { kind: ChunkKind::Module, node_kind: "mod_item", name_field: "name" },
    ]
}

fn js_ts_chunk_defs() -> Vec<ChunkDef> {
    vec![
        ChunkDef {
            kind: ChunkKind::Function,
            node_kind: "function_declaration",
            name_field: "name",
        },
        ChunkDef {
            kind: ChunkKind::Function,
            node_kind: "generator_function_declaration",
            name_field: "name",
        },
        ChunkDef { kind: ChunkKind::Class, node_kind: "class_declaration", name_field: "name" },
        ChunkDef {
            kind: ChunkKind::Interface,
            node_kind: "interface_declaration",
            name_field: "name",
        },
        ChunkDef {
            kind: ChunkKind::TypeAlias,
            node_kind: "type_alias_declaration",
            name_field: "name",
        },
        ChunkDef { kind: ChunkKind::Constant, node_kind: "lexical_declaration", name_field: "" },
        ChunkDef { kind: ChunkKind::Module, node_kind: "export_statement", name_field: "" },
    ]
}

fn chunk_defs_for_language(language: FileLanguage) -> Vec<ChunkDef> {
    match language {
        FileLanguage::Rust => rust_chunk_defs(),
        FileLanguage::JavaScript | FileLanguage::Jsx => js_ts_chunk_defs(),
        FileLanguage::TypeScript | FileLanguage::Tsx => js_ts_chunk_defs(),
        FileLanguage::Unknown => Vec::new(),
    }
}

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

    Some(CodeChunk {
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
        if node.kind() == "lexical_declaration" {
            return extract_name_from_lexical_decl(node, source_file);
        }
        if node.kind() == "export_statement" {
            return extract_name_from_export(node, source_file);
        }
    }
    
    None
}

fn extract_name_from_lexical_decl(node: &Node, source_file: &SourceFile) -> Option<String> {
    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            let child = cursor.node();
            if child.kind() == "variable_declarator"
                && let Some(name_node) = child.child_by_field_name("name")
            {
                return name_node
                    .utf8_text(source_file.source_text.as_bytes())
                    .ok()
                    .map(|s| s.to_string());
            }
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }
    None
}

fn extract_name_from_export(node: &Node, source_file: &SourceFile) -> Option<String> {
    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            let child = cursor.node();
            let child_name = extract_name_from_declaration(&child, source_file);
            if child_name.is_some() {
                return child_name;
            }
            if !cursor.goto_next_sibling() {
                break;
            }
        }
    }
    None
}

fn extract_name_from_declaration(node: &Node, source_file: &SourceFile) -> Option<String> {
    let field_name = match node.kind() {
        "function_declaration" | "generator_function_declaration" => "name",
        "class_declaration" => "name",
        "interface_declaration" => "name",
        "type_alias_declaration" => "name",
        "lexical_declaration" => return extract_name_from_lexical_decl(node, source_file),
        "export_statement" => return extract_name_from_export(node, source_file),
        _ => return None,
    };

    let name_node = node.child_by_field_name(field_name)?;
    name_node
        .utf8_text(source_file.source_text.as_bytes())
        .ok()
        .map(|s| s.to_string())
}
