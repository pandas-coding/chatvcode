use chatvcode_core::ChunkKind;
use tree_sitter::Node;

use chatvcode_core::SourceFile;

use crate::chunk::ChunkDef;

/// Returns chunk definitions for JavaScript/JSX/TypeScript/TSX source files.
///
/// Supported node types: `function_declaration`, `generator_function_declaration`,
/// `class_declaration`, `interface_declaration`, `type_alias_declaration`,
/// `lexical_declaration`, `export_statement`.
pub fn js_ts_chunk_defs() -> Vec<ChunkDef> {
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

/// Extracts a symbol name from JS/TS nodes that don't have a direct `name` field.
///
/// Handles `lexical_declaration` (const/let) and `export_statement` nodes.
pub fn extract_name_from_node(node: &Node, source_file: &SourceFile) -> Option<String> {
    if node.kind() == "lexical_declaration" {
        return extract_name_from_lexical_decl(node, source_file);
    }
    if node.kind() == "export_statement" {
        return extract_name_from_export(node, source_file);
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
                    .map(std::string::ToString::to_string);
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
        .map(std::string::ToString::to_string)
}
