use std::fs;
use std::io::{BufRead, BufReader};

use chatvcode_llm::{ToolCall, ToolDefinition, ToolParameter, ToolResult};
use regex::Regex;
use serde_json::Value;
use walkdir::WalkDir;

use crate::context::ToolContext;
use crate::error::AgentError;

use super::BuiltinTool;

pub struct GrepCodeTool;

impl BuiltinTool for GrepCodeTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("grep_code")
            .description("Search for a pattern in source files using regular expressions. Returns matching lines with file paths and line numbers.")
            .parameter(
                ToolParameter::string("pattern")
                    .description("Regular expression pattern to search for")
                    .required(true),
            )
            .parameter(
                ToolParameter::string("path")
                    .description("Directory or file to search in (relative to project root, default: \".\")"),
            )
            .parameter(
                ToolParameter::string("file_pattern")
                    .description("Glob pattern to filter which files to search (e.g., \"*.rs\")"),
            )
            .parameter(
                ToolParameter::boolean("case_sensitive")
                    .description("Whether the search is case-sensitive (default: true)"),
            )
            .parameter(
                ToolParameter::integer("max_results")
                    .description("Maximum number of results to return (default: 50)"),
            )
    }

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        self.validate_arguments(call)?;

        let pattern_str = call.get_string("pattern").unwrap();
        let search_path = call.get_string("path").unwrap_or(".");
        let file_pattern = call.get_string("file_pattern");
        let case_sensitive = call.get_bool("case_sensitive").unwrap_or(true);
        let max_results = call.get_i64("max_results").unwrap_or(50).max(1) as usize;

        let regex_pattern = if case_sensitive {
            pattern_str.to_string()
        } else {
            format!("(?i){}", pattern_str)
        };

        let regex = Regex::new(&regex_pattern).map_err(|e| AgentError::ToolError {
            tool_name: "grep_code".into(),
            message: format!("Invalid regex pattern '{}': {}", pattern_str, e),
        })?;

        let target = if search_path == "." {
            ctx.project_path.clone()
        } else {
            ctx.project_path.join(search_path)
        };

        if !target.exists() {
            return Ok(ToolResult::error(format!("Path does not exist: {}", search_path)));
        }

        let canonical_project = ctx.project_path.canonicalize().unwrap_or_else(|_| ctx.project_path.clone());

        let mut results: Vec<Value> = Vec::new();

        if target.is_file() {
            search_file(&target, &regex, &canonical_project, &mut results, max_results)?;
        } else {
            for entry in WalkDir::new(&target).follow_links(true) {
                if results.len() >= max_results {
                    break;
                }
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

                if let Some(pat) = file_pattern {
                    let file_name = entry.file_name().to_string_lossy();
                    if !simple_glob_match(&file_name, pat) && !simple_glob_match(&rel_path, pat) {
                        continue;
                    }
                }

                search_file(entry.path(), &regex, &canonical_project, &mut results, max_results)?;
            }
        }

        let result = serde_json::json!({
            "pattern": pattern_str,
            "path": search_path,
            "match_count": results.len(),
            "truncated": results.len() >= max_results,
            "matches": results,
        });

        Ok(ToolResult::success(result))
    }

    fn is_cacheable(&self) -> bool {
        true
    }
}

fn search_file(
    path: &std::path::Path,
    regex: &Regex,
    project_root: &std::path::Path,
    results: &mut Vec<Value>,
    max_results: usize,
) -> Result<(), AgentError> {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(()),
    };

    let reader = BufReader::new(file);
    let rel_path = path
        .strip_prefix(project_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");

    for (line_num, line_result) in reader.lines().enumerate() {
        if results.len() >= max_results {
            break;
        }
        let line = match line_result {
            Ok(l) => l,
            Err(_) => continue,
        };
        if regex.is_match(&line) {
            results.push(serde_json::json!({
                "file": rel_path,
                "line": line_num + 1,
                "text": line.trim(),
            }));
        }
    }

    Ok(())
}

fn simple_glob_match(name: &str, pattern: &str) -> bool {
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
    fn test_grep_basic() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("test.rs"), "fn hello() {}\nfn world() {}\nfn hello_world() {}\n").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GrepCodeTool;
        let mut args = std::collections::HashMap::new();
        args.insert("pattern".to_string(), Value::String("hello".into()));
        let call = ToolCall { name: "grep_code".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["match_count"].as_u64().unwrap(), 2);
    }

    #[test]
    fn test_grep_case_insensitive() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("test.rs"), "Hello\nhello\nHELLO\n").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GrepCodeTool;
        let mut args = std::collections::HashMap::new();
        args.insert("pattern".to_string(), Value::String("hello".into()));
        args.insert("case_sensitive".to_string(), Value::Bool(false));
        let call = ToolCall { name: "grep_code".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["match_count"].as_u64().unwrap(), 3);
    }

    #[test]
    fn test_grep_with_file_filter() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.rs"), "target\n").unwrap();
        fs::write(dir.path().join("b.txt"), "target\n").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GrepCodeTool;
        let mut args = std::collections::HashMap::new();
        args.insert("pattern".to_string(), Value::String("target".into()));
        args.insert("file_pattern".to_string(), Value::String("*.rs".into()));
        let call = ToolCall { name: "grep_code".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["match_count"].as_u64().unwrap(), 1);
    }

    #[test]
    fn test_grep_max_results() {
        let dir = TempDir::new().unwrap();
        let content: String = (0..100).map(|i| format!("match line {}\n", i)).collect();
        fs::write(dir.path().join("big.rs"), content).unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GrepCodeTool;
        let mut args = std::collections::HashMap::new();
        args.insert("pattern".to_string(), Value::String("match".into()));
        args.insert("max_results".to_string(), Value::Number(5.into()));
        let call = ToolCall { name: "grep_code".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["match_count"].as_u64().unwrap(), 5);
        assert_eq!(result.value["truncated"].as_bool().unwrap(), true);
    }

    #[test]
    fn test_grep_invalid_regex() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("test.rs"), "hello").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GrepCodeTool;
        let mut args = std::collections::HashMap::new();
        args.insert("pattern".to_string(), Value::String("[invalid".into()));
        let call = ToolCall { name: "grep_code".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_grep_regex_pattern() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("test.rs"), "fn foo() {}\nfn bar() {}\nlet x = 42;\n").unwrap();

        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GrepCodeTool;
        let mut args = std::collections::HashMap::new();
        args.insert("pattern".to_string(), Value::String(r"fn \w+\(\)".into()));
        let call = ToolCall { name: "grep_code".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["match_count"].as_u64().unwrap(), 2);
    }

    #[test]
    fn test_grep_missing_pattern() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx(dir.path().to_path_buf());
        let tool = GrepCodeTool;
        let call = ToolCall {
            name: "grep_code".into(),
            arguments: std::collections::HashMap::new(),
            id: None,
        };
        assert!(tool.execute(&call, &ctx).is_err());
    }

    #[test]
    fn test_grep_definition() {
        let tool = GrepCodeTool;
        let def = tool.definition();
        assert_eq!(def.name, "grep_code");
        assert_eq!(def.required_params(), vec!["pattern"]);
    }
}
