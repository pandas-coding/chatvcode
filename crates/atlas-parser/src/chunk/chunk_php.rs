use atlas_core::ChunkKind;
use tree_sitter::Node;

use atlas_core::SourceFile;

use crate::chunk::ChunkDef;

/// Returns chunk definitions for PHP source files.
///
/// Supported node types: `function_definition`, `class_declaration`,
/// `interface_declaration`, `trait_declaration`, `enum_declaration`,
/// `method_declaration`, `namespace_definition`.
pub fn php_chunk_defs() -> Vec<ChunkDef> {
    vec![
        ChunkDef {
            kind: ChunkKind::Function,
            node_kind: "function_definition",
            name_field: "name",
        },
        ChunkDef { kind: ChunkKind::Class, node_kind: "class_declaration", name_field: "name" },
        ChunkDef {
            kind: ChunkKind::Interface,
            node_kind: "interface_declaration",
            name_field: "name",
        },
        ChunkDef { kind: ChunkKind::Trait, node_kind: "trait_declaration", name_field: "name" },
        ChunkDef { kind: ChunkKind::Enum, node_kind: "enum_declaration", name_field: "name" },
        ChunkDef { kind: ChunkKind::Method, node_kind: "method_declaration", name_field: "name" },
        ChunkDef { kind: ChunkKind::Module, node_kind: "namespace_definition", name_field: "" },
    ]
}

/// Extracts a symbol name from PHP nodes that don't have a direct `name` field.
///
/// Handles `decorated_definition` and `namespace_definition` nodes.
pub fn extract_name_from_node(node: &Node, source_file: &SourceFile) -> Option<String> {
    if node.kind() == "namespace_definition" {
        return extract_name_from_namespace(node, source_file);
    }
    None
}

fn extract_name_from_namespace(node: &Node, source_file: &SourceFile) -> Option<String> {
    let mut cursor = node.walk();
    if cursor.goto_first_child() {
        loop {
            let child = cursor.node();
            if child.kind() == "namespace_name" || child.kind() == "identifier" {
                return child
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
