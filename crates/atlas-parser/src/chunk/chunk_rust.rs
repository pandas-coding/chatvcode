use atlas_core::ChunkKind;

use crate::chunk::ChunkDef;

/// Returns chunk definitions for Rust source files.
///
/// Supported node types: `function_item`, `struct_item`, `enum_item`,
/// `trait_item`, `impl_item`, `type_item`, `const_item`, `mod_item`.
pub fn rust_chunk_defs() -> Vec<ChunkDef> {
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
