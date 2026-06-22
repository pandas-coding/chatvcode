//! Function calling and tool use support for LLM inference.
//!
//! Provides types and utilities for defining tools that an LLM can call,
//! parsing tool call responses, and executing tool results.
//!
//! # Example
//!
//! ```ignore
//! use chatvcode_llm::tools::{ToolDefinition, ToolParameter, ToolRegistry, parse_tool_calls};
//!
//! // Define a tool
//! let tool = ToolDefinition::new("get_weather")
//!     .description("Get the current weather for a location")
//!     .parameter(ToolParameter::string("location").required(true))
//!     .parameter(ToolParameter::string("unit").enum_values(vec!["celsius", "fahrenheit"]));
//!
//! // Register tools
//! let registry = ToolRegistry::new().register(tool);
//!
//! // Generate prompt with tool definitions
//! let prompt = registry.format_tool_prompt();
//!
//! // Parse tool calls from model response
//! let response = r#"{"name": "get_weather", "arguments": {"location": "San Francisco", "unit": "celsius"}}"#;
//! let calls = parse_tool_calls(response).unwrap();
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{LlmError, LlmResult};

/// A tool (function) definition that can be provided to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name (must be unique).
    pub name: String,
    /// Human-readable description of what the tool does.
    pub description: String,
    /// Parameters the tool accepts.
    pub parameters: Vec<ToolParameter>,
}

impl ToolDefinition {
    /// Create a new tool definition.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            parameters: Vec::new(),
        }
    }

    /// Set the tool description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add a parameter to the tool.
    pub fn parameter(mut self, param: ToolParameter) -> Self {
        self.parameters.push(param);
        self
    }

    /// Get required parameter names.
    pub fn required_params(&self) -> Vec<&str> {
        self.parameters.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect()
    }

    /// Format this tool as a JSON schema-like object.
    pub fn to_schema(&self) -> Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &self.parameters {
            properties.insert(param.name.clone(), param.to_schema());
            if param.required {
                required.push(Value::String(param.name.clone()));
            }
        }

        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        })
    }
}

/// A parameter for a tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    /// Parameter name.
    pub name: String,
    /// Parameter type (string, number, integer, boolean, array, object).
    pub param_type: String,
    /// Parameter description.
    pub description: String,
    /// Whether this parameter is required.
    pub required: bool,
    /// Allowed enum values (if any).
    pub enum_values: Vec<String>,
    /// Default value (if any).
    pub default: Option<Value>,
}

impl ToolParameter {
    /// Create a string parameter.
    pub fn string(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            param_type: "string".to_string(),
            description: String::new(),
            required: false,
            enum_values: Vec::new(),
            default: None,
        }
    }

    /// Create a number parameter.
    pub fn number(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            param_type: "number".to_string(),
            description: String::new(),
            required: false,
            enum_values: Vec::new(),
            default: None,
        }
    }

    /// Create an integer parameter.
    pub fn integer(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            param_type: "integer".to_string(),
            description: String::new(),
            required: false,
            enum_values: Vec::new(),
            default: None,
        }
    }

    /// Create a boolean parameter.
    pub fn boolean(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            param_type: "boolean".to_string(),
            description: String::new(),
            required: false,
            enum_values: Vec::new(),
            default: None,
        }
    }

    /// Create an array parameter.
    pub fn array(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            param_type: "array".to_string(),
            description: String::new(),
            required: false,
            enum_values: Vec::new(),
            default: None,
        }
    }

    /// Create an object parameter.
    pub fn object(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            param_type: "object".to_string(),
            description: String::new(),
            required: false,
            enum_values: Vec::new(),
            default: None,
        }
    }

    /// Set the parameter description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Mark this parameter as required.
    pub fn required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// Set allowed enum values.
    pub fn enum_values(mut self, values: Vec<impl Into<String>>) -> Self {
        self.enum_values = values.into_iter().map(|v| v.into()).collect();
        self
    }

    /// Set a default value.
    pub fn default_value(mut self, value: Value) -> Self {
        self.default = Some(value);
        self
    }

    /// Format this parameter as a JSON schema.
    pub fn to_schema(&self) -> Value {
        let mut schema = serde_json::json!({
            "type": self.param_type,
            "description": self.description
        });

        if !self.enum_values.is_empty() {
            schema["enum"] = Value::Array(self.enum_values.iter().map(|v| Value::String(v.clone())).collect());
        }

        if let Some(ref default) = self.default {
            schema["default"] = default.clone();
        }

        schema
    }
}

/// A tool call parsed from the model's response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// The name of the tool to call.
    pub name: String,
    /// The arguments to pass to the tool.
    pub arguments: HashMap<String, Value>,
    /// Optional call ID for tracking.
    pub id: Option<String>,
}

impl ToolCall {
    /// Get a string argument.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.arguments.get(key).and_then(|v| v.as_str())
    }

    /// Get an integer argument.
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.arguments.get(key).and_then(|v| v.as_i64())
    }

    /// Get a float argument.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.arguments.get(key).and_then(|v| v.as_f64())
    }

    /// Get a boolean argument.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.arguments.get(key).and_then(|v| v.as_bool())
    }
}

/// Result of executing a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The call ID this result corresponds to.
    pub call_id: Option<String>,
    /// Whether the tool execution succeeded.
    pub success: bool,
    /// The result value (or error message).
    pub value: Value,
}

impl ToolResult {
    /// Create a successful result.
    pub fn success(value: Value) -> Self {
        Self { call_id: None, success: true, value }
    }

    /// Create an error result.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            call_id: None,
            success: false,
            value: Value::String(message.into()),
        }
    }

    /// Set the call ID.
    pub fn with_call_id(mut self, id: impl Into<String>) -> Self {
        self.call_id = Some(id.into());
        self
    }
}

/// Registry for managing tool definitions.
#[derive(Debug, Clone, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, ToolDefinition>,
}

impl ToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    /// Register a tool definition.
    pub fn register(mut self, tool: ToolDefinition) -> Self {
        self.tools.insert(tool.name.clone(), tool);
        self
    }

    /// Register a tool definition (mutable reference version).
    pub fn register_tool(&mut self, tool: ToolDefinition) {
        self.tools.insert(tool.name.clone(), tool);
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name)
    }

    /// List all registered tools.
    pub fn list(&self) -> Vec<&ToolDefinition> {
        self.tools.values().collect()
    }

    /// Get the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Format tool definitions as a prompt section.
    ///
    /// This generates a system prompt section that describes available tools
    /// in a format that LLMs can understand.
    pub fn format_tool_prompt(&self) -> String {
        if self.tools.is_empty() {
            return String::new();
        }

        let mut prompt = String::from(
            "You have access to the following tools. To use a tool, respond with a JSON object in this format:\n\
             {\"name\": \"tool_name\", \"arguments\": {\"param1\": \"value1\"}}\n\n\
             Available tools:\n\n",
        );

        for tool in self.tools.values() {
            prompt.push_str(&format!("## {}\n", tool.name));
            prompt.push_str(&format!("{}\n\n", tool.description));

            if !tool.parameters.is_empty() {
                prompt.push_str("Parameters:\n");
                for param in &tool.parameters {
                    let required = if param.required { " (required)" } else { "" };
                    prompt.push_str(&format!("- {}: {}{}\n", param.name, param.description, required));
                    if !param.enum_values.is_empty() {
                        prompt.push_str(&format!("  Allowed values: {}\n", param.enum_values.join(", ")));
                    }
                }
                prompt.push('\n');
            }
        }

        prompt
    }

    /// Format tool definitions as JSON schema array (for OpenAI-compatible APIs).
    pub fn to_json_schema(&self) -> Value {
        let schemas: Vec<Value> = self.tools.values().map(|t| t.to_schema()).collect();
        Value::Array(schemas)
    }
}

/// Parse tool calls from a model response.
///
/// Supports multiple formats:
/// - Single JSON object: `{"name": "tool", "arguments": {...}}`
/// - JSON array: `[{"name": "tool1", ...}, {"name": "tool2", ...}]`
/// - Markdown code block with JSON
pub fn parse_tool_calls(response: &str) -> LlmResult<Vec<ToolCall>> {
    let trimmed = response.trim();

    // Try to extract JSON from markdown code blocks
    let json_str = if trimmed.starts_with("```") {
        // Extract content between ``` markers
        let start = trimmed.find('\n').unwrap_or(0) + 1;
        let end = trimmed.rfind("```").unwrap_or(trimmed.len());
        trimmed[start..end].trim()
    } else {
        trimmed
    };

    // Try parsing as array first
    if let Ok(arr) = serde_json::from_str::<Vec<Value>>(json_str) {
        let mut calls = Vec::new();
        for item in arr {
            if let Some(call) = parse_single_call(&item) {
                calls.push(call);
            }
        }
        if !calls.is_empty() {
            return Ok(calls);
        }
    }

    // Try parsing as single object
    if let Ok(obj) = serde_json::from_str::<Value>(json_str) {
        if let Some(call) = parse_single_call(&obj) {
            return Ok(vec![call]);
        }
    }

    // Try to find JSON objects in the text
    let mut calls = Vec::new();
    let mut depth = 0;
    let mut start = None;

    for (i, ch) in json_str.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        let candidate = &json_str[s..=i];
                        if let Ok(obj) = serde_json::from_str::<Value>(candidate) {
                            if let Some(call) = parse_single_call(&obj) {
                                calls.push(call);
                            }
                        }
                    }
                    start = None;
                }
            }
            _ => {}
        }
    }

    if calls.is_empty() {
        return Err(LlmError::Internal("No tool calls found in response".to_string()));
    }

    Ok(calls)
}

fn parse_single_call(value: &Value) -> Option<ToolCall> {
    let name = value.get("name")?.as_str()?.to_string();

    let arguments = if let Some(args) = value.get("arguments") {
        if let Some(obj) = args.as_object() {
            obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        } else {
            HashMap::new()
        }
    } else {
        HashMap::new()
    };

    let id = value.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());

    Some(ToolCall { name, arguments, id })
}

/// Check if a response contains tool calls.
pub fn has_tool_calls(response: &str) -> bool {
    parse_tool_calls(response).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_definition_new() {
        let tool = ToolDefinition::new("get_weather")
            .description("Get weather info")
            .parameter(ToolParameter::string("location").required(true));

        assert_eq!(tool.name, "get_weather");
        assert_eq!(tool.description, "Get weather info");
        assert_eq!(tool.parameters.len(), 1);
        assert_eq!(tool.required_params(), vec!["location"]);
    }

    #[test]
    fn test_tool_parameter_types() {
        let s = ToolParameter::string("name");
        assert_eq!(s.param_type, "string");

        let n = ToolParameter::number("value");
        assert_eq!(n.param_type, "number");

        let i = ToolParameter::integer("count");
        assert_eq!(i.param_type, "integer");

        let b = ToolParameter::boolean("flag");
        assert_eq!(b.param_type, "boolean");

        let a = ToolParameter::array("items");
        assert_eq!(a.param_type, "array");

        let o = ToolParameter::object("data");
        assert_eq!(o.param_type, "object");
    }

    #[test]
    fn test_tool_parameter_builder() {
        let param = ToolParameter::string("unit")
            .description("Temperature unit")
            .required(true)
            .enum_values(vec!["celsius", "fahrenheit"])
            .default_value(Value::String("celsius".to_string()));

        assert_eq!(param.description, "Temperature unit");
        assert!(param.required);
        assert_eq!(param.enum_values, vec!["celsius", "fahrenheit"]);
        assert!(param.default.is_some());
    }

    #[test]
    fn test_tool_definition_to_schema() {
        let tool = ToolDefinition::new("search")
            .description("Search the web")
            .parameter(ToolParameter::string("query").required(true))
            .parameter(ToolParameter::integer("limit"));

        let schema = tool.to_schema();
        assert_eq!(schema["type"], "function");
        assert_eq!(schema["function"]["name"], "search");
        assert!(schema["function"]["parameters"]["required"].as_array().unwrap().len() == 1);
    }

    #[test]
    fn test_tool_call_getters() {
        let mut args = HashMap::new();
        args.insert("name".to_string(), Value::String("Alice".to_string()));
        args.insert("age".to_string(), Value::Number(serde_json::Number::from(30)));
        args.insert("score".to_string(), Value::Number(serde_json::Number::from_f64(95.5).unwrap()));
        args.insert("active".to_string(), Value::Bool(true));

        let call = ToolCall { name: "test".to_string(), arguments: args, id: None };

        assert_eq!(call.get_string("name"), Some("Alice"));
        assert_eq!(call.get_i64("age"), Some(30));
        assert_eq!(call.get_f64("score"), Some(95.5));
        assert_eq!(call.get_bool("active"), Some(true));
        assert_eq!(call.get_string("missing"), None);
    }

    #[test]
    fn test_tool_registry() {
        let registry = ToolRegistry::new()
            .register(ToolDefinition::new("tool1").description("First tool"))
            .register(ToolDefinition::new("tool2").description("Second tool"));

        assert_eq!(registry.len(), 2);
        assert!(!registry.is_empty());
        assert!(registry.get("tool1").is_some());
        assert!(registry.get("tool3").is_none());
    }

    #[test]
    fn test_tool_registry_format_prompt() {
        let registry = ToolRegistry::new()
            .register(
                ToolDefinition::new("get_weather")
                    .description("Get weather information")
                    .parameter(ToolParameter::string("location").required(true).description("City name")),
            );

        let prompt = registry.format_tool_prompt();
        assert!(prompt.contains("get_weather"));
        assert!(prompt.contains("Get weather information"));
        assert!(prompt.contains("location"));
        assert!(prompt.contains("required"));
    }

    #[test]
    fn test_parse_tool_calls_single() {
        let response = r#"{"name": "get_weather", "arguments": {"location": "SF"}}"#;
        let calls = parse_tool_calls(response).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].get_string("location"), Some("SF"));
    }

    #[test]
    fn test_parse_tool_calls_array() {
        let response = r#"[{"name": "tool1", "arguments": {}}, {"name": "tool2", "arguments": {"x": 1}}]"#;
        let calls = parse_tool_calls(response).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "tool1");
        assert_eq!(calls[1].name, "tool2");
    }

    #[test]
    fn test_parse_tool_calls_markdown() {
        let response = "Here's the tool call:\n```json\n{\"name\": \"search\", \"arguments\": {\"query\": \"rust\"}}\n```";
        let calls = parse_tool_calls(response).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn test_parse_tool_calls_embedded() {
        let response = "I'll call the tool: {\"name\": \"calc\", \"arguments\": {\"expr\": \"2+2\"}} done.";
        let calls = parse_tool_calls(response).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "calc");
    }

    #[test]
    fn test_parse_tool_calls_none() {
        let response = "This is just a regular response with no tool calls.";
        let result = parse_tool_calls(response);
        assert!(result.is_err());
    }

    #[test]
    fn test_has_tool_calls() {
        assert!(has_tool_calls(r#"{"name": "tool", "arguments": {}}"#));
        assert!(!has_tool_calls("Just a regular response"));
    }

    #[test]
    fn test_tool_result() {
        let result = ToolResult::success(Value::String("done".to_string()));
        assert!(result.success);
        assert!(result.call_id.is_none());

        let result = ToolResult::error("failed").with_call_id("call-123");
        assert!(!result.success);
        assert_eq!(result.call_id, Some("call-123".to_string()));
    }

    #[test]
    fn test_empty_registry_prompt() {
        let registry = ToolRegistry::new();
        assert!(registry.format_tool_prompt().is_empty());
    }

    #[test]
    fn test_registry_to_json_schema() {
        let registry = ToolRegistry::new()
            .register(ToolDefinition::new("tool1"))
            .register(ToolDefinition::new("tool2"));

        let schema = registry.to_json_schema();
        assert!(schema.as_array().is_some());
        assert_eq!(schema.as_array().unwrap().len(), 2);
    }
}
