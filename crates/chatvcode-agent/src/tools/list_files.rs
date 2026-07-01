use chatvcode_llm::{ToolCall, ToolDefinition, ToolParameter, ToolResult};
use serde_json::Value;
use walkdir::WalkDir;

use crate::context::ToolContext;
use crate::error::AgentError;

use super::BuiltinTool;

pub struct ListFilesTool;

impl BuiltinTool for ListFilesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("list_files")
            .description("List files in a directory recursively with optional glob pattern filtering and depth limit.")
            .parameter(
                ToolParameter::string("path")
                    .description("Directory path to list (relative to project root, default: \".\")"),
            )
            .parameter(
                ToolParameter::string("pattern")
                    .description("Glob pattern to filter files (e.g., \"*.rs\", \"src/**/*.ts\")"),
            )
            .parameter(
                ToolParameter::integer("max_depth")
                    .description("Maximum directory depth to traverse (default: unlimited)"),
            )
    }

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        self.validate_arguments(call)?;

        let sub_path = call.get_string("path").unwrap_or(".");
        let pattern = call.get_string("pattern");
        let max_depth = call.get_i64("max_depth").map(|v| v.max(1) as usize);

        let target_dir = if sub_path == "." {
            ctx.project_path.clone()
        } else {
            ctx.project_path.join(sub_path)
        };

        if !target_dir.is_dir() {
            return Ok(ToolResult::error(format!("Not a directory: {}", sub_path)));
        }

        let canonical_project = ctx
            .project_path
            .canonicalize()
            .unwrap_or_else(|_| ctx.project_path.clone());
        let canonical_target = target_dir
            .canonicalize()
            .unwrap_or_else(|_| target_dir.clone());
        if !canonical_target.starts_with(&canonical_project) {
            return Ok(ToolResult::error(format!(
                "Path '{}' is outside the project directory",
                sub_path
            )));
        }

        let mut walker = WalkDir::new(&target_dir).follow_links(true);
        if let Some(depth) = max_depth {
            walker = walker.max_depth(depth);
        }

        let mut files: Vec<Value> = Vec::new();
        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            if !entry.file_type().is_file() {
                continue;
            }

            let rel_path = entry
                .path()
                .strip_prefix(&canonical_project)
                .unwrap_or(entry.path())
                .to_string_lossy()
                .replace('\\', "/");

            if let Some(pat) = pattern
                && !glob_match(&rel_path, pat)
            {
                continue;
            }

            let metadata = entry.metadata().ok();
            let size = metadata.as_ref().map(|m| m.len()).unwrap_or(0);

            files.push(serde_json::json!({
                "path": rel_path,
                "size": size,
            }));
        }

        files.sort_by(|a, b| {
            a["path"]
                .as_str()
                .unwrap_or("")
                .cmp(b["path"].as_str().unwrap_or(""))
        });

        let result = serde_json::json!({
            "directory": sub_path,
            "pattern": pattern,
            "count": files.len(),
            "files": files,
        });

        Ok(ToolResult::success(result))
    }

    fn is_cacheable(&self) -> bool {
        true
    }
}

fn glob_match(path: &str, pattern: &str) -> bool {
    if pattern.contains("**") {
        let parts: Vec<&str> = pattern.split("**").collect();
        if parts.len() == 2 {
            let prefix = parts[0].trim_end_matches('/');
            let suffix = parts[1].trim_start_matches('/');
            let prefix_ok = prefix.is_empty() || path.starts_with(prefix);
            let suffix_ok = suffix.is_empty() || simple_glob_match(path, suffix);
            return prefix_ok && suffix_ok;
        }
    }
    simple_glob_match(path, pattern)
}

fn simple_glob_match(path: &str, pattern: &str) -> bool {
    if pattern.starts_with('*') && !pattern.starts_with("**") {
        let suffix = &pattern[1..];
        return path.ends_with(suffix);
    }
    if pattern.ends_with('*') && !pattern.ends_with("**") {
        let prefix = &pattern[..pattern.len() - 1];
        return path.starts_with(prefix);
    }
    if !pattern.contains('/') {
        let file_name = path.rsplit('/').next().unwrap_or(path);
        return simple_glob_match_flat(file_name, pattern);
    }
    path == pattern
}

fn simple_glob_match_flat(name: &str, pattern: &str) -> bool {
    if let Some(suffix) = pattern.strip_prefix('*') {
        return name.ends_with(suffix);
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }
    name == pattern
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AgentServices, ChunkMetadataStoreTrait, CodeSearchService};
    use chatvcode_core::model::{ChunkMetadata, SearchResult};
    use std::fs;
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

    fn make_ctx(project_path: PathBuf) -> ToolContext {
        ToolContext {
            project_path,
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
                    chunk_store: Box::new(MockChunkStore),
                },
            ),
        }
    }

    #[test]
    fn test_list_files_basic() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.rs"), "fn a() {}").unwrap();
        fs::write(dir.path().join("b.txt"), "hello").unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("sub/c.rs"), "fn c() {}").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = ListFilesTool;
        let call = ToolCall {
            name: "list_files".into(),
            arguments: std::collections::HashMap::new(),
            id: None,
        };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["count"].as_u64().unwrap(), 3);
    }

    #[test]
    fn test_list_files_with_pattern() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.rs"), "").unwrap();
        fs::write(dir.path().join("b.txt"), "").unwrap();
        fs::write(dir.path().join("c.rs"), "").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = ListFilesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("pattern".to_string(), Value::String("*.rs".into()));
        let call = ToolCall { name: "list_files".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["count"].as_u64().unwrap(), 2);
    }

    #[test]
    fn test_list_files_with_depth() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.rs"), "").unwrap();
        fs::create_dir_all(dir.path().join("sub/deep")).unwrap();
        fs::write(dir.path().join("sub/b.rs"), "").unwrap();
        fs::write(dir.path().join("sub/deep/c.rs"), "").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = ListFilesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("max_depth".to_string(), Value::Number(2.into()));
        let call = ToolCall { name: "list_files".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["count"].as_u64().unwrap(), 2);
    }

    #[test]
    fn test_list_files_not_a_directory() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("file.txt"), "").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = ListFilesTool;
        let mut args = std::collections::HashMap::new();
        args.insert("path".to_string(), Value::String("file.txt".into()));
        let call = ToolCall { name: "list_files".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_list_files_definition() {
        let tool = ListFilesTool;
        let def = tool.definition();
        assert_eq!(def.name, "list_files");
        assert!(def.required_params().is_empty());
    }

    #[test]
    fn test_glob_match() {
        assert!(glob_match("src/main.rs", "*.rs"));
        assert!(!glob_match("src/main.rs", "*.ts"));
        assert!(glob_match("src/lib/mod.rs", "**/*.rs"));
        assert!(glob_match("src/deep/nested/file.rs", "**/*.rs"));
    }
}
