use std::fs;

use chatvcode_core::model::SourceFile;
use chatvcode_llm::{ToolCall, ToolDefinition, ToolParameter, ToolResult};
use serde_json::Value;

use crate::context::ToolContext;
use crate::error::AgentError;

use super::{BuiltinTool, resolve_safe_path};

pub struct GetFileStructureTool;

impl BuiltinTool for GetFileStructureTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("get_file_structure")
            .description("Parse a source file and return its structure (functions, structs, classes, etc.) with line numbers.")
            .parameter(
                ToolParameter::string("path")
                    .description("Path to the source file to analyze (relative to project root)")
                    .required(true),
            )
    }

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        self.validate_arguments(call)?;

        let path = call.get_string("path").unwrap();
        let resolved = resolve_safe_path(&ctx.project_path, path)?;

        if !resolved.is_file() {
            return Ok(ToolResult::error(format!("Not a file: {}", path)));
        }

        let source_text = fs::read_to_string(&resolved).map_err(|e| AgentError::ToolError {
            tool_name: "get_file_structure".into(),
            message: format!("Failed to read file '{}': {}", path, e),
        })?;

        let source_file = SourceFile::new(&resolved, source_text);
        let parse_result =
            ctx.services
                .parser
                .parse(source_file)
                .map_err(|e| AgentError::ToolError {
                    tool_name: "get_file_structure".into(),
                    message: format!("Failed to parse file '{}': {}", path, e),
                })?;

        let chunks: Vec<Value> = parse_result
            .chunks
            .iter()
            .map(|chunk| {
                serde_json::json!({
                    "kind": format!("{}", chunk.kind),
                    "symbol": chunk.symbol_name,
                    "start_line": chunk.span.start_line + 1,
                    "end_line": chunk.span.end_line + 1,
                })
            })
            .collect();

        let errors: Vec<String> = parse_result
            .errors
            .iter()
            .map(|e| format!("{}", e))
            .collect();

        let result = serde_json::json!({
            "path": path,
            "language": format!("{}", parse_result.file.language),
            "chunks": chunks,
            "chunk_count": chunks.len(),
            "errors": errors,
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
    use crate::context::{AgentServices, ChunkMetadataStoreTrait, CodeSearchService};
    use chatvcode_core::ChatVCodeResult;
    use chatvcode_core::model::{
        ChunkKind, ChunkMetadata, ChunkSpan, CodeChunk, ParseResult, SearchResult, SourceFile,
    };
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::TempDir;

    struct MockSearch;
    impl CodeSearchService for MockSearch {
        fn search(&self, _: &str, _: usize) -> Result<Vec<SearchResult>, AgentError> {
            Ok(vec![])
        }
    }

    struct MockChunkStore;
    impl ChunkMetadataStoreTrait for MockChunkStore {
        fn get_chunks_by_symbol(&self, _: &str, _: Option<&str>) -> Vec<ChunkMetadata> {
            vec![]
        }
        fn get_chunk_by_id(&self, _: &str) -> Option<ChunkMetadata> {
            None
        }
    }

    struct MockParser;
    impl chatvcode_core::ParseSource for MockParser {
        fn parse(&self, source_file: SourceFile) -> ChatVCodeResult<ParseResult> {
            let chunk = CodeChunk {
                id: "test:fn:hello:0".into(),
                file_path: source_file.path.clone(),
                language: source_file.language,
                kind: ChunkKind::Function,
                symbol_name: Some("hello".into()),
                span: ChunkSpan::new(0, 20, 0, 2),
                source_text: "fn hello() {}".into(),
            };
            Ok(ParseResult::success(source_file, vec![chunk]))
        }
    }

    fn make_ctx(project_path: PathBuf) -> ToolContext {
        ToolContext {
            project_path,
            timeout: Duration::from_secs(30),
            token_budget: 4096,
            services: Arc::new(AgentServices {
                search: Box::new(MockSearch),
                parser: Box::new(MockParser),
                chunk_store: Box::new(MockChunkStore),
            }),
        }
    }

    #[test]
    fn test_get_file_structure() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("test.rs"), "fn hello() {}").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GetFileStructureTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path".to_string(), Value::String("test.rs".into()));
        let call = ToolCall { name: "get_file_structure".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["chunk_count"].as_u64().unwrap(), 1);
        let chunks = result.value["chunks"].as_array().unwrap();
        assert_eq!(chunks[0]["kind"].as_str().unwrap(), "function");
        assert_eq!(chunks[0]["symbol"].as_str().unwrap(), "hello");
    }

    #[test]
    fn test_get_file_structure_not_found() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GetFileStructureTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path".to_string(), Value::String("nonexistent.rs".into()));
        let call = ToolCall { name: "get_file_structure".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_get_file_structure_missing_path() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GetFileStructureTool;
        let call = ToolCall {
            name: "get_file_structure".into(),
            arguments: std::collections::HashMap::new(),
            id: None,
        };
        assert!(tool.execute(&call, &ctx).is_err());
    }

    #[test]
    fn test_get_file_structure_definition() {
        let tool = GetFileStructureTool;
        let def = tool.definition();
        assert_eq!(def.name, "get_file_structure");
        assert_eq!(def.required_params(), vec!["path"]);
    }
}
