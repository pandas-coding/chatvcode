use chatvcode_core::ChunkKind;

use crate::chunk::ChunkDef;

/// Returns chunk definitions for Python source files.
///
/// Supported node types: `function_definition`, `class_definition`,
/// `decorated_definition`.
pub fn python_chunk_defs() -> Vec<ChunkDef> {
    vec![
        ChunkDef {
            kind: ChunkKind::Function,
            node_kind: "function_definition",
            name_field: "name",
        },
        ChunkDef { kind: ChunkKind::Class, node_kind: "class_definition", name_field: "name" },
        ChunkDef { kind: ChunkKind::Module, node_kind: "decorated_definition", name_field: "" },
    ]
}
