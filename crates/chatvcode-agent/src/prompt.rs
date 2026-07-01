use std::sync::Arc;

use chatvcode_llm::{ToolCall, ToolResult};

use crate::executor::ToolExecutor;
use crate::types::AgentStep;

/// System prompt template for the ChatVCode Agent.
///
/// Contains a `{tool_descriptions}` placeholder that is replaced with the
/// formatted tool definitions produced by [`ToolExecutor::format_tool_prompt`].
pub const AGENT_SYSTEM_PROMPT: &str = r#"You are ChatVCode Agent, an AI assistant specialized in understanding and reasoning about codebases.

You explore codebases step by step using the available tools. Think carefully before acting.

## Available Tools

{tool_descriptions}

## How to Use Tools

To use a tool, output a JSON object in this format:
```json
{"tool": "<tool_name>", "arguments": {<parameters>}}
```

You can call multiple tools in a single step:
```json
[
  {"tool": "search_code", "arguments": {"query": "authentication"}},
  {"tool": "list_files", "arguments": {"path": "src/auth"}}
]
```

## Reasoning Process

1. **Analyze**: Understand what information you need to answer the question.
2. **Search**: Use search_code or grep_code to find relevant code.
3. **Inspect**: Use read_file to examine specific files in detail.
4. **Synthesize**: Combine your findings into a comprehensive answer.

## Guidelines

- Always cite file paths and line numbers when referencing code (e.g., `src/auth.rs:42`).
- Start with semantic search (search_code) for broad exploration, then narrow down.
- Use get_file_structure to quickly understand a file's organization.
- Use grep_code when you need to find specific patterns or strings.
- Limit your exploration to what's necessary; don't over-search.
- If you cannot find relevant information after reasonable exploration, say so honestly.
- When you have enough information, provide your final answer directly (without tool calls).

## Output Format

Your final answer should:
- Be clear and well-structured
- Include specific code references (file:line)
- Explain the key concepts and relationships
- Highlight any caveats or edge cases you noticed
"#;

/// Planning phase prompt template.
///
/// Contains a `{query}` placeholder replaced with the user's question.
pub const PLANNING_PROMPT: &str = r#"
The user has asked: "{query}"

Please analyze this question and decide how to explore the codebase. Consider:
1. What specific information do you need?
2. Which tools would be most helpful?
3. What's the most efficient exploration path?

You can either:
- Call tools to gather information
- Answer directly if you already have enough context from the codebase"#;

/// Observing phase prompt template.
///
/// Contains a `{tool_results}` placeholder replaced with the formatted
/// output of the most recent tool calls.
pub const OBSERVING_PROMPT: &str = r#"
Here are the results from your tool calls:

{tool_results}

Based on these results:
- Do you have enough information to answer the question?
- If not, what additional information do you need?
- Which tools should you use next?

Proceed with more tool calls if needed, or provide your final answer."#;

/// Concluding phase prompt template.
///
/// Contains a `{exploration_summary}` placeholder replaced with a summary of
/// all exploration steps taken so far.
pub const CONCLUDING_PROMPT: &str = r#"
You have gathered information through your exploration. Here's a summary:

{exploration_summary}

Please synthesize your findings and provide a comprehensive final answer.

Remember to:
- Include specific file paths and line numbers
- Explain the key concepts clearly
- Note any limitations or caveats"#;

/// Error recovery prompt template.
///
/// Contains an `{error}` placeholder replaced with the error message produced
/// by a failed tool call.
pub const ERROR_RECOVERY_PROMPT: &str = r#"
The previous tool call failed with error: {error}

Please try a different approach. Consider:
- Using alternative tools
- Adjusting your search parameters
- Working with the information you already have"#;

/// Loop detection prompt template.
///
/// Contains a `{summary}` placeholder replaced with a summary of the
/// exploration steps that triggered the loop detection.
pub const LOOP_DETECTED_PROMPT: &str = r#"
I notice you've been using similar tool calls repeatedly. Please either:
- Try a completely different approach
- Provide a final answer with the information gathered so far

Current exploration summary:
{summary}"#;

/// Builds Agent prompts by composing static templates with dynamic content
/// (tool descriptions, step data, exploration summaries).
///
/// The builder is stateless aside from its reference to a [`ToolExecutor`],
/// which supplies the tool descriptions injected into the system prompt.
pub struct AgentPromptBuilder {
    system_prompt_template: String,
    tool_registry: Arc<dyn ToolExecutor>,
}

impl AgentPromptBuilder {
    /// Create a new prompt builder backed by the given tool registry.
    pub fn new(tool_registry: Arc<dyn ToolExecutor>) -> Self {
        Self { system_prompt_template: AGENT_SYSTEM_PROMPT.to_string(), tool_registry }
    }

    /// Override the default system prompt template.
    ///
    /// The template must contain a `{tool_descriptions}` placeholder that
    /// will be replaced when [`build_system_prompt`](Self::build_system_prompt)
    /// is called.
    pub fn with_system_prompt_template(mut self, template: impl Into<String>) -> Self {
        self.system_prompt_template = template.into();
        self
    }

    /// Build the complete system prompt with tool descriptions injected.
    pub fn build_system_prompt(&self) -> String {
        let tool_descriptions = self.tool_registry.format_tool_prompt();
        self.system_prompt_template
            .replace("{tool_descriptions}", &tool_descriptions)
    }

    /// Build the planning phase prompt for a user query.
    pub fn build_planning_prompt(&self, query: &str) -> String {
        PLANNING_PROMPT.replace("{query}", query)
    }

    /// Build the observing phase prompt for a completed step.
    ///
    /// The step's tool calls are paired with their results so that tool names
    /// are included in the formatted output.
    pub fn build_observing_prompt(&self, step: &AgentStep) -> String {
        let tool_results =
            self.format_tool_results_with_calls(&step.tool_calls, &step.tool_results);
        OBSERVING_PROMPT.replace("{tool_results}", &tool_results)
    }

    /// Build the concluding phase prompt from all exploration steps.
    pub fn build_concluding_prompt(&self, steps: &[AgentStep]) -> String {
        let summary = self.build_exploration_summary(steps);
        CONCLUDING_PROMPT.replace("{exploration_summary}", &summary)
    }

    /// Build an error recovery prompt for a failed tool call.
    pub fn build_error_recovery_prompt(&self, error: &str) -> String {
        ERROR_RECOVERY_PROMPT.replace("{error}", error)
    }

    /// Build a loop detection prompt from the exploration steps that
    /// triggered the detection.
    pub fn build_loop_detected_prompt(&self, steps: &[AgentStep]) -> String {
        let summary = self.build_exploration_summary(steps);
        LOOP_DETECTED_PROMPT.replace("{summary}", &summary)
    }

    /// Format a slice of tool results for inclusion in a prompt.
    ///
    /// Because [`ToolResult`](chatvcode_llm::ToolResult) does not carry the
    /// originating tool name, results are labelled by their 1-based index.
    /// Use [`format_tool_results_with_calls`](Self::format_tool_results_with_calls)
    /// when the corresponding tool calls are available and tool names are
    /// desired.
    pub fn format_tool_results(&self, results: &[ToolResult]) -> String {
        results
            .iter()
            .enumerate()
            .map(|(i, r)| format_tool_result_entry(i + 1, None, r))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Format tool results alongside their originating tool calls, producing
    /// entries that include the tool name for each result.
    pub fn format_tool_results_with_calls(
        &self,
        calls: &[ToolCall],
        results: &[ToolResult],
    ) -> String {
        results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let name = calls.get(i).map(|c| c.name.as_str());
                format_tool_result_entry(i + 1, name, r)
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Build a textual summary of the exploration from the recorded steps.
    ///
    /// Each step that performed tool calls is summarised as a single line
    /// listing the tool invocations, optionally followed by a truncated
    /// preview of the agent's observation.
    pub fn build_exploration_summary(&self, steps: &[AgentStep]) -> String {
        let mut summary = Vec::new();

        for step in steps {
            if !step.tool_calls.is_empty() {
                let tools: Vec<String> = step
                    .tool_calls
                    .iter()
                    .map(|c| format!("{}({})", c.name, summarize_args(&c.arguments)))
                    .collect();
                summary.push(format!("Step {}: {}", step.step_number, tools.join(", ")));
            }
            if let Some(thought) = &step.thought
                && !thought.is_empty()
            {
                let preview = truncate_preview(thought, 100);
                summary.push(format!("  Observation: {}", preview));
            }
        }

        summary.join("\n")
    }
}

fn format_tool_result_entry(index: usize, name: Option<&str>, result: &ToolResult) -> String {
    let label = match name {
        Some(n) => format!("[Tool {}] {}", index, n),
        None => format!("[Tool {}]", index),
    };

    if result.success {
        let body = serde_json::to_string_pretty(&result.value).unwrap_or_default();
        format!("{}\nResult:\n{}\n---", label, body)
    } else {
        let error = match &result.value {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        format!("{} (FAILED)\nError: {}\n---", label, error)
    }
}

fn summarize_args(args: &std::collections::HashMap<String, serde_json::Value>) -> String {
    if args.is_empty() {
        return String::new();
    }
    let mut keys: Vec<&String> = args.keys().collect();
    keys.sort();
    let parts: Vec<String> = keys
        .iter()
        .map(|k| format!("{}={}", k, compact_value(&args[*k])))
        .collect();
    parts.join(", ")
}

fn compact_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => {
            if s.chars().count() > 30 {
                let prefix: String = s.chars().take(27).collect();
                format!("\"{}...\"", prefix)
            } else {
                format!("\"{}\"", s)
            }
        }
        other => other.to_string(),
    }
}

fn truncate_preview(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let prefix: String = text.chars().take(max).collect();
    format!("{}...", prefix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{BuiltinToolRegistry, ToolExecutor};
    use crate::types::{AgentState, ThinkingPhase, TokenUsage};
    use chatvcode_llm::ToolCall;
    use insta::assert_snapshot;
    use serde_json::Value;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn make_registry() -> Arc<dyn ToolExecutor> {
        let mut registry = BuiltinToolRegistry::new(crate::types::ToolRetryConfig::default());
        registry.register_defaults();
        Arc::new(registry)
    }

    fn make_call(name: &str, args: HashMap<String, Value>) -> ToolCall {
        ToolCall { name: name.to_string(), arguments: args, id: None }
    }

    fn make_step(
        number: usize,
        calls: Vec<ToolCall>,
        results: Vec<ToolResult>,
        thought: Option<&str>,
    ) -> AgentStep {
        AgentStep {
            step_number: number,
            state: AgentState::Acting,
            thinking_phase: Some(ThinkingPhase::Observing),
            thought: thought.map(|s| s.to_string()),
            tool_calls: calls,
            tool_results: results,
            duration_ms: 0,
            token_usage: TokenUsage::default(),
        }
    }

    #[test]
    fn snapshot_system_prompt() {
        let builder = AgentPromptBuilder::new(make_registry());
        let prompt = builder.build_system_prompt();
        assert_snapshot!(prompt);
    }

    #[test]
    fn snapshot_system_prompt_includes_tool_descriptions() {
        let builder = AgentPromptBuilder::new(make_registry());
        let prompt = builder.build_system_prompt();
        assert!(prompt.contains("read_file"));
        assert!(prompt.contains("search_code"));
        assert!(!prompt.contains("{tool_descriptions}"));
    }

    #[test]
    fn snapshot_planning_prompt() {
        let builder = AgentPromptBuilder::new(make_registry());
        let prompt = builder.build_planning_prompt("How is authentication implemented?");
        assert_snapshot!(prompt);
    }

    #[test]
    fn snapshot_observing_prompt() {
        let builder = AgentPromptBuilder::new(make_registry());
        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("src/auth.rs".into()));
        let call = make_call("read_file", args);
        let result = ToolResult::success(Value::String("fn validate_token() {}".into()));
        let step = make_step(1, vec![call], vec![result], Some("Found the token validator."));
        let prompt = builder.build_observing_prompt(&step);
        assert_snapshot!(prompt);
    }

    #[test]
    fn snapshot_concluding_prompt() {
        let builder = AgentPromptBuilder::new(make_registry());
        let mut args = HashMap::new();
        args.insert("query".to_string(), Value::String("auth".into()));
        let call = make_call("search_code", args);
        let result = ToolResult::success(Value::String("3 matches".into()));
        let steps = vec![make_step(
            1,
            vec![call],
            vec![result],
            Some("Located the authentication module."),
        )];
        let prompt = builder.build_concluding_prompt(&steps);
        assert_snapshot!(prompt);
    }

    #[test]
    fn snapshot_error_recovery_prompt() {
        let builder = AgentPromptBuilder::new(make_registry());
        let prompt = builder.build_error_recovery_prompt("Tool 'read_file' timed out");
        assert_snapshot!(prompt);
    }

    #[test]
    fn snapshot_loop_detected_prompt() {
        let builder = AgentPromptBuilder::new(make_registry());
        let mut args = HashMap::new();
        args.insert("query".to_string(), Value::String("auth".into()));
        let call = make_call("search_code", args);
        let result = ToolResult::success(Value::String("3 matches".into()));
        let steps = vec![
            make_step(1, vec![call.clone()], vec![result.clone()], None),
            make_step(2, vec![call], vec![result], None),
        ];
        let prompt = builder.build_loop_detected_prompt(&steps);
        assert_snapshot!(prompt);
    }

    #[test]
    fn format_tool_results_success_and_failure() {
        let builder = AgentPromptBuilder::new(make_registry());
        let ok = ToolResult::success(Value::String("hello".into()));
        let err = ToolResult::error("boom");
        let formatted = builder.format_tool_results(&[ok, err]);
        assert!(formatted.contains("[Tool 1]"));
        assert!(formatted.contains("hello"));
        assert!(formatted.contains("[Tool 2]"));
        assert!(formatted.contains("FAILED"));
        assert!(formatted.contains("boom"));
    }

    #[test]
    fn format_tool_results_with_calls_includes_names() {
        let builder = AgentPromptBuilder::new(make_registry());
        let call = make_call("grep_code", HashMap::new());
        let result = ToolResult::success(Value::String("matched".into()));
        let formatted = builder.format_tool_results_with_calls(&[call], &[result]);
        assert!(formatted.contains("grep_code"));
    }

    #[test]
    fn build_exploration_summarises_steps() {
        let builder = AgentPromptBuilder::new(make_registry());
        let mut args = HashMap::new();
        args.insert("query".to_string(), Value::String("auth".into()));
        let call = make_call("search_code", args);
        let result = ToolResult::success(Value::String("hits".into()));
        let steps = vec![make_step(
            1,
            vec![call],
            vec![result],
            Some(
                "A long observation that should be truncated because it exceeds the one hundred character preview limit easily.",
            ),
        )];
        let summary = builder.build_exploration_summary(&steps);
        assert!(summary.contains("Step 1: search_code(query=\"auth\")"));
        assert!(summary.contains("Observation:"));
        assert!(summary.contains("..."));
    }

    #[test]
    fn templates_keep_placeholders_when_substituted() {
        let builder = AgentPromptBuilder::new(make_registry());

        assert!(
            !builder
                .build_system_prompt()
                .contains("{tool_descriptions}")
        );
        assert!(!builder.build_planning_prompt("q").contains("{query}"));
        assert!(!builder.build_error_recovery_prompt("e").contains("{error}"));
    }
}
