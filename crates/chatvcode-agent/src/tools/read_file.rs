use std::fs;
use std::io::{BufRead, BufReader};

use chatvcode_llm::{ToolCall, ToolDefinition, ToolParameter, ToolResult};

use crate::context::ToolContext;
use crate::error::AgentError;

use super::{BuiltinTool, resolve_safe_path};

pub struct ReadFileTool;

impl BuiltinTool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("read_file")
            .description("Read the contents of a file. Supports reading the full file or a specific line range.")
            .parameter(
                ToolParameter::string("path")
                    .description("Path to the file to read (relative to project root)")
                    .required(true),
            )
            .parameter(
                ToolParameter::integer("offset")
                    .description("Starting line number (1-indexed, optional)"),
            )
            .parameter(
                ToolParameter::integer("limit")
                    .description("Maximum number of lines to read (optional)"),
            )
    }

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        self.validate_arguments(call)?;

        let path = call.get_string("path").unwrap();
        let offset = call.get_i64("offset").unwrap_or(1).max(1) as usize;
        let limit = call.get_i64("limit").map(|v| v.max(1) as usize);

        let resolved = resolve_safe_path(&ctx.project_path, path)?;

        if !resolved.is_file() {
            return Ok(ToolResult::error(format!("Not a file: {}", path)));
        }

        let file = fs::File::open(&resolved).map_err(|e| AgentError::ToolError {
            tool_name: "read_file".into(),
            message: format!("Failed to open file '{}': {}", path, e),
        })?;

        let reader = BufReader::new(file);
        let lines: Vec<String> = reader
            .lines()
            .skip(offset - 1)
            .take(limit.unwrap_or(usize::MAX))
            .collect::<Result<_, _>>()
            .map_err(|e| AgentError::ToolError {
                tool_name: "read_file".into(),
                message: format!("Failed to read file '{}': {}", path, e),
            })?;

        let total_lines = fs::read_to_string(&resolved)
            .map(|c| c.lines().count())
            .unwrap_or(0);

        let numbered: Vec<String> = lines
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{}: {}", offset + i, line))
            .collect();

        let content = numbered.join("\n");
        let result = serde_json::json!({
            "path": path,
            "offset": offset,
            "lines_returned": lines.len(),
            "total_lines": total_lines,
            "content": content,
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
    use chatvcode_core::model::{ChunkMetadata, SearchResult};
    use serde_json::Value;
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
        fn get_chunks_by_symbol(&self, _: &str, _: Option<&str>) -> Vec<ChunkMetadata> { vec![] }
        fn get_chunk_by_id(&self, _: &str) -> Option<ChunkMetadata> { None }
    }

    fn make_ctx(project_path: PathBuf) -> ToolContext {
        ToolContext {
            project_path,
            timeout: Duration::from_secs(30),
            token_budget: 4096,
            services: Arc::new(AgentServices {
                search: Box::new(MockSearch),
                parser: Box::new(|_: chatvcode_core::model::SourceFile| -> chatvcode_core::ChatVCodeResult<chatvcode_core::model::ParseResult> {
                    unimplemented!()
                }),
                chunk_store: Box::new(MockChunkStore),
            }),
        }
    }

    #[test]
    fn test_read_file_full() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "line1\nline2\nline3\n").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = ReadFileTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path".to_string(), Value::String("test.txt".into()));
        let call = ToolCall { name: "read_file".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        let content = result.value["content"].as_str().unwrap();
        assert!(content.contains("1: line1"));
        assert!(content.contains("2: line2"));
        assert!(content.contains("3: line3"));
    }

    #[test]
    fn test_read_file_with_offset_and_limit() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "a\nb\nc\nd\ne\n").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = ReadFileTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path".to_string(), Value::String("test.txt".into()));
        args.insert("offset".to_string(), Value::Number(2.into()));
        args.insert("limit".to_string(), Value::Number(2.into()));
        let call = ToolCall { name: "read_file".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        let content = result.value["content"].as_str().unwrap();
        assert!(content.contains("2: b"));
        assert!(content.contains("3: c"));
        assert!(!content.contains("4: d"));
    }

    #[test]
    fn test_read_file_not_found() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = ReadFileTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path".to_string(), Value::String("nonexistent.txt".into()));
        let call = ToolCall { name: "read_file".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_read_file_missing_path_param() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = ReadFileTool;
        let call = ToolCall {
            name: "read_file".into(),
            arguments: std::collections::HashMap::new(),
            id: None,
        };

        let result = tool.execute(&call, &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_file_definition() {
        let tool = ReadFileTool;
        let def = tool.definition();
        assert_eq!(def.name, "read_file");
        assert_eq!(def.required_params(), vec!["path"]);
    }
}
