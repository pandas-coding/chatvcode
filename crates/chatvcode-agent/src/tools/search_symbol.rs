use chatvcode_llm::{ToolCall, ToolDefinition, ToolParameter, ToolResult};
use serde_json::Value;

use crate::context::ToolContext;
use crate::error::AgentError;

use super::BuiltinTool;

pub struct SearchSymbolTool;

impl BuiltinTool for SearchSymbolTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("search_symbol")
            .description("Search for a symbol (function, struct, class, etc.) by name in the indexed codebase. Returns matching symbols with their file locations and kinds.")
            .parameter(
                ToolParameter::string("symbol")
                    .description("Symbol name to search for (partial match, case-insensitive)")
                    .required(true),
            )
            .parameter(
                ToolParameter::string("kind")
                    .description("Filter by symbol kind: function, struct, enum, trait, class, interface, method, etc."),
            )
    }

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        self.validate_arguments(call)?;

        let symbol = call.get_string("symbol").unwrap();
        let kind = call.get_string("kind");

        let chunks = ctx.services.chunk_store.get_chunks_by_symbol(symbol, kind);

        let results: Vec<Value> = chunks
            .iter()
            .map(|meta| {
                serde_json::json!({
                    "chunk_id": meta.chunk_id,
                    "symbol": meta.symbol_name,
                    "kind": format!("{}", meta.kind),
                    "file": meta.file_path.to_string_lossy().replace('\\', "/"),
                    "start_line": meta.start_line,
                    "end_line": meta.end_line,
                    "language": meta.language,
                })
            })
            .collect();

        let result = serde_json::json!({
            "symbol": symbol,
            "kind_filter": kind,
            "match_count": results.len(),
            "matches": results,
        });

        Ok(ToolResult::success(result))
    }

    fn is_cacheable(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AgentServices, ChunkMetadataStoreAdapter, CodeSearchService};
    use chatvcode_core::model::{ChunkKind, ChunkMetadata, ChunkMetadataStore, SearchResult};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;

    struct MockSearch;
    impl CodeSearchService for MockSearch {
        fn search(&self, _: &str, _: usize) -> Result<Vec<SearchResult>, AgentError> {
            Ok(vec![])
        }
    }

    fn make_metadata(chunk_id: &str, symbol: &str, kind: ChunkKind, file: &str) -> ChunkMetadata {
        ChunkMetadata {
            chunk_id: chunk_id.into(),
            file_path: PathBuf::from(file),
            language: "rust".into(),
            kind,
            symbol_name: Some(symbol.into()),
            start_line: 1,
            end_line: 10,
            start_byte: 0,
            end_byte: 100,
            source_text: String::new(),
        }
    }

    fn make_ctx_with_store(store: ChunkMetadataStore) -> ToolContext {
        ToolContext {
            project_path: PathBuf::from("/test"),
            timeout: Duration::from_secs(30),
            token_budget: 4096,
            services: Arc::new(
                AgentServices {
                    search: Box::new(MockSearch),
                    parser:
                        Box::new(
                            |_: chatvcode_core::model::SourceFile| -> chatvcode_core::ChatVCodeResult<
                                chatvcode_core::model::ParseResult,
                            > { unimplemented!() },
                        ),
                    chunk_store: Box::new(ChunkMetadataStoreAdapter::new(store)),
                },
            ),
        }
    }

    #[test]
    fn test_search_symbol_basic() {
        let mut store = ChunkMetadataStore::new();
        store.insert(make_metadata("c1", "foo_bar", ChunkKind::Function, "src/a.rs"));
        store.insert(make_metadata("c2", "baz", ChunkKind::Function, "src/b.rs"));
        store.insert(make_metadata("c3", "FooStruct", ChunkKind::Struct, "src/c.rs"));

        let ctx = make_ctx_with_store(store);
        let tool = SearchSymbolTool;
        let mut args = std::collections::HashMap::new();
        args.insert("symbol".to_string(), Value::String("foo".into()));
        let call = ToolCall { name: "search_symbol".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["match_count"].as_u64().unwrap(), 2);
    }

    #[test]
    fn test_search_symbol_with_kind_filter() {
        let mut store = ChunkMetadataStore::new();
        store.insert(make_metadata("c1", "foo", ChunkKind::Function, "a.rs"));
        store.insert(make_metadata("c2", "FooStruct", ChunkKind::Struct, "b.rs"));

        let ctx = make_ctx_with_store(store);
        let tool = SearchSymbolTool;
        let mut args = std::collections::HashMap::new();
        args.insert("symbol".to_string(), Value::String("foo".into()));
        args.insert("kind".to_string(), Value::String("struct".into()));
        let call = ToolCall { name: "search_symbol".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["match_count"].as_u64().unwrap(), 1);
    }

    #[test]
    fn test_search_symbol_no_results() {
        let store = ChunkMetadataStore::new();
        let ctx = make_ctx_with_store(store);
        let tool = SearchSymbolTool;
        let mut args = std::collections::HashMap::new();
        args.insert("symbol".to_string(), Value::String("nonexistent".into()));
        let call = ToolCall { name: "search_symbol".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["match_count"].as_u64().unwrap(), 0);
    }

    #[test]
    fn test_search_symbol_missing_param() {
        let store = ChunkMetadataStore::new();
        let ctx = make_ctx_with_store(store);
        let tool = SearchSymbolTool;
        let call = ToolCall {
            name: "search_symbol".into(),
            arguments: std::collections::HashMap::new(),
            id: None,
        };
        assert!(tool.execute(&call, &ctx).is_err());
    }

    #[test]
    fn test_search_symbol_definition() {
        let tool = SearchSymbolTool;
        let def = tool.definition();
        assert_eq!(def.name, "search_symbol");
        assert_eq!(def.required_params(), vec!["symbol"]);
    }
}
