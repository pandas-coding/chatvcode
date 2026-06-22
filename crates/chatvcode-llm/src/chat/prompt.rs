//! Chat prompt builder for RAG-aware single-turn prompts.
//!
//! [`ChatPromptBuilder`] constructs chat prompts with optional system prompt,
//! retrieved context (RAG), and token budget management.

use crate::chat::message::ChatMessage;
use crate::chat::template::ChatTemplate;

// Chat prompt builder (single-turn, RAG-aware)
// ---------------------------------------------------------------------------

/// Builder for constructing chat prompts with optional system prompt and
/// retrieved context (RAG).
///
/// This is the primary interface for building prompts for single-turn
/// question-answering. It supports:
/// - System prompt injection
/// - Context injection from code retrieval results
/// - Token budget management for context snippets
/// - Multiple chat template formats
///
/// # Example
///
/// ```ignore
/// use chatvcode_llm::{ChatPromptBuilder, ChatTemplate};
///
/// let prompt = ChatPromptBuilder::new(ChatTemplate::ChatML)
///     .system_prompt("You are a helpful coding assistant.")
///     .user_question("What does the `main` function do?")
///     .context("fn main() { println!(\"hello\"); }")
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ChatPromptBuilder {
    /// Chat template to use for formatting.
    template: ChatTemplate,
    /// System prompt (prepended as a system message).
    system_prompt: Option<String>,
    /// User question.
    user_question: Option<String>,
    /// Retrieved context snippets to inject before the question.
    context: Vec<String>,
    /// Maximum token budget for context. 0 = unlimited.
    context_token_budget: usize,
    /// Whether to add the assistant generation prompt at the end.
    add_generation_prompt: bool,
    /// Additional messages to include (for future multi-turn support).
    /// In M3, this is reserved but not actively used.
    history: Vec<ChatMessage>,
}

impl ChatPromptBuilder {
    /// Create a new prompt builder with the given template.
    ///
    /// Defaults:
    /// - No system prompt
    /// - No context
    /// - No token budget (unlimited)
    /// - `add_generation_prompt` = `true`
    #[must_use]
    pub fn new(template: ChatTemplate) -> Self {
        Self {
            template,
            system_prompt: None,
            user_question: None,
            context: Vec::new(),
            context_token_budget: 0,
            add_generation_prompt: true,
            history: Vec::new(),
        }
    }

    /// Set the system prompt.
    ///
    /// The system prompt is prepended as a `system` role message,
    /// establishing the model's behavior and persona.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the user question.
    ///
    /// This is the main question the user is asking.
    pub fn user_question(mut self, question: impl Into<String>) -> Self {
        self.user_question = Some(question.into());
        self
    }

    /// Add a context snippet.
    ///
    /// Context snippets are injected into the prompt to provide
    /// retrieved code or other reference material (RAG).
    pub fn context(mut self, ctx: impl Into<String>) -> Self {
        self.context.push(ctx.into());
        self
    }

    /// Set multiple context snippets at once.
    pub fn context_snippets(mut self, snippets: Vec<String>) -> Self {
        self.context = snippets;
        self
    }

    /// Set the token budget for context snippets.
    ///
    /// When set to a non-zero value, context snippets are truncated
    /// to fit within approximately this many tokens. A rough heuristic
    /// of 4 characters per token is used for estimation.
    ///
    /// Set to 0 for unlimited context (default).
    #[must_use]
    pub const fn context_token_budget(mut self, budget: usize) -> Self {
        self.context_token_budget = budget;
        self
    }

    /// Set whether to add the assistant generation prompt.
    ///
    /// When `true` (default), the formatted prompt ends with the
    /// assistant prefix, ready for the model to continue.
    #[must_use]
    pub const fn add_generation_prompt(mut self, add: bool) -> Self {
        self.add_generation_prompt = add;
        self
    }

    /// Add a message to the conversation history.
    ///
    /// **Reserved for future multi-turn support.** In M3, this is
    /// not actively used but provides an extension point.
    pub fn message(mut self, msg: ChatMessage) -> Self {
        self.history.push(msg);
        self
    }

    /// Build the formatted prompt string.
    ///
    /// This constructs the full message sequence:
    /// 1. System prompt (if set)
    /// 2. Conversation history (reserved for multi-turn, currently empty)
    /// 3. Context-injected user question
    ///
    /// The user question is formatted with context as:
    /// ```text
    /// [Retrieved Context]
    /// --- snippet 1 ---
    /// {context_1}
    /// --- snippet 2 ---
    /// {context_2}
    /// ---
    ///
    /// {user_question}
    /// ```
    ///
    /// If no context is provided, the user question is used directly.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::InvalidParameter`] if no user question is set,
    /// or if the template cannot format the messages (e.g., Custom jinja).
    pub fn build(self) -> Result<String, crate::error::LlmError> {
        let question = self.user_question.ok_or_else(|| {
            crate::error::LlmError::InvalidParameter(
                "user question is required for ChatPromptBuilder".into(),
            )
        })?;

        // Construct the user message content, with optional context injection
        let user_content = if self.context.is_empty() {
            question
        } else {
            let mut content = String::new();
            content.push_str("[Retrieved Context]\n");

            // Apply token budget if set
            let mut remaining_budget = if self.context_token_budget > 0 {
                // Rough heuristic: ~4 chars per token
                self.context_token_budget.saturating_mul(4)
            } else {
                usize::MAX
            };

            for (i, snippet) in self.context.iter().enumerate() {
                let header = format!("--- snippet {} ---\n", i + 1);
                let budget_for_snippet = remaining_budget.saturating_sub(header.len());
                let truncated = if snippet.len() > budget_for_snippet {
                    let end = snippet.floor_char_boundary(budget_for_snippet);
                    &snippet[..end]
                } else {
                    snippet.as_str()
                };

                content.push_str(&header);
                content.push_str(truncated);
                content.push('\n');

                remaining_budget = remaining_budget.saturating_sub(header.len() + truncated.len());
                if remaining_budget == 0 {
                    content.push_str("... (context truncated due to token budget)\n");
                    break;
                }
            }

            content.push_str("---\n\n");
            content.push_str(&question);
            content
        };

        // Build message list
        let mut messages = Vec::new();

        if let Some(sys) = &self.system_prompt {
            messages.push(ChatMessage::system(sys.as_str()));
        }

        messages.extend(self.history);
        messages.push(ChatMessage::user(user_content));

        self.template.format(&messages, self.add_generation_prompt)
    }
}
