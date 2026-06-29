use chatvcode_llm::{ToolCall, ToolDefinition, ToolParameter, ToolResult};
use serde_json::Value;

use crate::context::ToolContext;
use crate::error::AgentError;

use super::BuiltinTool;

pub struct SearchCodeTool;

impl BuiltinTool for SearchCodeTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("search_code")
            .description("Semantic code search using embeddings. Finds relevant code chunks by natural language query.")
            .parameter(
                ToolParameter::string("query")
                    .description("Natural language query describing the code you're looking for")
                    .required(true),
            )
            .parameter(
                ToolParameter::integer("top_k")
                    .description("Maximum number of results to return (default: 10)"),
            )
    }

    fn execute(&self, call: &ToolCall, ctx: &ToolContext) -> Result<ToolResult, AgentError> {
        self.validate_arguments(call)?;

        let query = call.get_string("query").unwrap();
        let top_k = call.get_i64("top_k").unwrap_or(10).max(1) as usize;

        let results = ctx.services.search.search(query, top_k)?;

        let items: Vec<Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "file": r.chunk.file_path.to_string_lossy().replace('\\', "/"),
                    "start_line": r.chunk.span.start_line + 1,
                    "end_line": r.chunk.span.end_line + 1,
                    "symbol": r.chunk.symbol_name,
                    "kind": format!("{}", r.chunk.kind),
                    "score": r.score,
                    "snippet": truncate_snippet(&r.chunk.source_text, 300),
                })
            })
            .collect();

        let result = serde_json::json!({
            "query": query,
            "result_count": items.len(),
            "results": items,
        });

        Ok(ToolResult::success(result))
    }

    fn summarize_result(&self, result: &ToolResult) -> String {
        if !result.success {
            return format!("Error: {}", result.value);
        }

        let results = match result.value.get("results").and_then(|v| v.as_array()) {
            Some(arr) => arr,
            None => return "No results".to_string(),
        };

        let summaries: Vec<String> = results
            .iter()
            .take(5)
            .map(|r| {
                let file = r["file"].as_str().unwrap_or("?");
                let line = r["start_line"].as_u64().unwrap_or(0);
                let symbol = r["symbol"].as_str().unwrap_or("?");
                let score = r["score"].as_f64().unwrap_or(0.0);
                format!("{}:{} {} (score: {:.2})", file, line, symbol, score)
            })
            .collect();

        let total = result.value["result_count"].as_u64().unwrap_or(0);
        format!("Found {} results:\n{}", total, summaries.join("\n"))
    }

    fn is_cacheable(&self) -> bool {
        true
    }
}

fn truncate_snippet(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AgentServices, ChunkMetadataStoreAdapter, CodeSearchService};
    use chatvcode_core::model::{
        ChunkKind, ChunkMetadataStore, ChunkSpan, CodeChunk, SearchResult,
    };
    use chatvcode_core::model::FileLanguage;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;

    struct MockSearch {
        results: Vec<SearchResult>,
    }

    impl CodeSearchService for MockSearch {
        fn search(&self, _query: &str, _top_k: usize) -> Result<Vec<SearchResult>, AgentError> {
            Ok(self.results.clone())
        }
    }

    fn make_search_result(chunk_id: &str, symbol: &str, score: f32) -> SearchResult {
        SearchResult {
            chunk_id: chunk_id.into(),
            score,
            chunk: CodeChunk {
                id: chunk_id.into(),
                file_path: PathBuf::from("src/test.rs"),
                language: FileLanguage::Rust,
                kind: ChunkKind::Function,
                symbol_name: Some(symbol.into()),
                span: ChunkSpan::new(0, 100, 0, 10),
                source_text: format!("fn {}() {{ /* implementation */ }}", symbol),
            },
        }
    }

    fn make_ctx(results: Vec<SearchResult>) -> ToolContext {
        let store = ChunkMetadataStore::new();
        ToolContext {
            project_path: PathBuf::from("/test"),
            timeout: Duration::from_secs(30),
            token_budget: 4096,
            services: Arc::new(AgentServices {
                search: Box::new(MockSearch { results }),
                parser: Box::new(|_: chatvcode_core::model::SourceFile| -> chatvcode_core::ChatVCodeResult<chatvcode_core::model::ParseResult> {
                    unimplemented!()
                }),
                chunk_store: Box::new(ChunkMetadataStoreAdapter::new(store)),
            }),
        }
    }

    #[test]
    fn test_search_code_basic() {
        let results = vec![
            make_search_result("c1", "process_data", 0.95),
            make_search_result("c2", "parse_input", 0.87),
        ];
        let ctx = make_ctx(results);
        let tool = SearchCodeTool;
        let mut args = std::collections::HashMap::new();
        args.insert("query".to_string(), Value::String("data processing".into()));
        let call = ToolCall { name: "search_code".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["result_count"].as_u64().unwrap(), 2);

        let items = result.value["results"].as_array().unwrap();
        assert_eq!(items[0]["symbol"].as_str().unwrap(), "process_data");
        assert!((items[0]["score"].as_f64().unwrap() - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_search_code_empty_results() {
        let ctx = make_ctx(vec![]);
        let tool = SearchCodeTool;
        let mut args = std::collections::HashMap::new();
        args.insert("query".to_string(), Value::String("nothing".into()));
        let call = ToolCall { name: "search_code".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        assert!(result.success);
        assert_eq!(result.value["result_count"].as_u64().unwrap(), 0);
    }

    #[test]
    fn test_search_code_summarize() {
        let results = vec![
            make_search_result("c1", "foo", 0.9),
            make_search_result("c2", "bar", 0.8),
        ];
        let ctx = make_ctx(results);
        let tool = SearchCodeTool;
        let mut args = std::collections::HashMap::new();
        args.insert("query".to_string(), Value::String("test".into()));
        let call = ToolCall { name: "search_code".into(), arguments: args, id: None };

        let result = tool.execute(&call, &ctx).unwrap();
        let summary = tool.summarize_result(&result);
        assert!(summary.contains("Found 2 results"));
        assert!(summary.contains("foo"));
        assert!(summary.contains("bar"));
    }

    #[test]
    fn test_search_code_missing_query() {
        let ctx = make_ctx(vec![]);
        let tool = SearchCodeTool;
        let call = ToolCall {
            name: "search_code".into(),
            arguments: std::collections::HashMap::new(),
            id: None,
        };
        assert!(tool.execute(&call, &ctx).is_err());
    }

    #[test]
    fn test_search_code_definition() {
        let tool = SearchCodeTool;
        let def = tool.definition();
        assert_eq!(def.name, "search_code");
        assert_eq!(def.required_params(), vec!["query"]);
    }

    #[test]
    fn test_truncate_snippet() {
        assert_eq!(truncate_snippet("short", 10), "short");
        let long = "x".repeat(500);
        let truncated = truncate_snippet(&long, 100);
        assert_eq!(truncated.len(), 103);
        assert!(truncated.ends_with("..."));
    }
}
