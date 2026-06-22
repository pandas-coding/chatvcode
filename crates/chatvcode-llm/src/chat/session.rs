//! Chat session for multi-turn conversation management.
//!
//! [`ChatSession`] maintains conversation history across multiple turns,
//! supports KV cache reuse, history trimming, and JSON serialization.

use crate::chat::message::ChatMessage;
use crate::chat::template::ChatTemplate;
use crate::types::{GenerationParams, InferenceResponse, StreamEvent};
use super::{token_estimate, token_estimate_messages};

// Chat session (multi-turn conversation support)
// ---------------------------------------------------------------------------

/// A chat session that maintains conversation history across multiple turns.
///
/// Supports:
/// - Multi-turn conversation with context accumulation
/// - KV cache reuse for efficient multi-turn inference
/// - History trimming for context window management
/// - System prompt management
/// - Both synchronous and streaming inference
///
/// # Example
///
/// ```ignore
/// use chatvcode_llm::{ChatSession, ChatTemplate, GenerationParams, LlmService};
///
/// let mut session = ChatSession::new(ChatTemplate::ChatML)
///     .system_prompt("You are a helpful coding assistant.")
///     .max_context_tokens(4096);
///
/// // First turn
/// let response = session.chat("What is Rust?", &llm, &params)?;
/// println!("{}", response.text);
///
/// // Second turn (context from first turn is preserved)
/// let response = session.chat("How do lifetimes work?", &llm, &params)?;
/// println!("{}", response.text);
/// ```
#[derive(Debug, Clone)]
pub struct ChatSession {
    session_id: String,
    system_prompt: Option<String>,
    messages: Vec<ChatMessage>,
    max_history_turns: usize,
    template: ChatTemplate,
    estimated_tokens: usize,
    max_context_tokens: usize,
    reserve_for_response: usize,
    /// Opaque KV cache state from the last inference call.
    /// Passed to `LlmService::infer_cached` for multi-turn efficiency.
    pub(crate) kv_cache_state: crate::service::KvCacheState,
}

impl ChatSession {
    /// Create a new chat session with the given template.
    #[must_use]
    pub fn new(template: ChatTemplate) -> Self {
        Self {
            session_id: uuid_like_id(),
            system_prompt: None,
            messages: Vec::new(),
            max_history_turns: 0,
            template,
            estimated_tokens: 0,
            max_context_tokens: 0,
            reserve_for_response: 512,
            kv_cache_state: 0,
        }
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the maximum number of history turns to keep.
    ///
    /// A "turn" is one user + one assistant message.
    /// Setting to 0 means unlimited history.
    #[must_use]
    pub const fn max_history_turns(mut self, turns: usize) -> Self {
        self.max_history_turns = turns;
        self
    }

    /// Set the maximum context token count.
    ///
    /// When the estimated token count exceeds this, older messages
    /// are trimmed from the history.
    #[must_use]
    pub const fn max_context_tokens(mut self, tokens: usize) -> Self {
        self.max_context_tokens = tokens;
        self
    }

    /// Set the number of tokens to reserve for the model's response.
    ///
    /// When trimming history, this amount is subtracted from the
    /// `max_context_tokens` budget to leave room for the generated response.
    /// Default: 512.
    #[must_use]
    pub const fn reserve_for_response(mut self, tokens: usize) -> Self {
        self.reserve_for_response = tokens;
        self
    }

    /// Set the system prompt at runtime.
    ///
    /// Replaces any existing system prompt. Pass `None` to remove it.
    /// Invalidates the KV cache since the context changed.
    pub fn set_system_prompt(&mut self, prompt: Option<String>) {
        self.system_prompt = prompt;
        self.recalculate_tokens();
        self.kv_cache_state = 0;
    }

    /// Get the current system prompt.
    #[must_use]
    pub fn get_system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    /// Clear the system prompt.
    ///
    /// Invalidates the KV cache since the context changed.
    pub fn clear_system_prompt(&mut self) {
        self.system_prompt = None;
        self.recalculate_tokens();
        self.kv_cache_state = 0;
    }

    /// Add a message to the session.
    pub fn add_message(&mut self, msg: ChatMessage) {
        self.messages.push(msg);
        self.recalculate_tokens();
    }

    /// Add a user message to the session.
    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.messages.push(ChatMessage::user(content));
        self.recalculate_tokens();
    }

    /// Add an assistant message to the session.
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.messages.push(ChatMessage::assistant(content));
        self.recalculate_tokens();
    }

    /// Clear all conversation history.
    ///
    /// The system prompt is preserved. Call `clear_system_prompt()` separately
    /// to also remove the system prompt.
    ///
    /// The KV cache state is invalidated since the context changed.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.estimated_tokens = 0;
        self.kv_cache_state = 0;
    }

    /// Reset the session entirely, including the system prompt.
    ///
    /// The KV cache state is invalidated.
    pub fn reset(&mut self) {
        self.messages.clear();
        self.system_prompt = None;
        self.estimated_tokens = 0;
        self.kv_cache_state = 0;
    }

    /// Run synchronous inference for the next user message.
    ///
    /// This method:
    /// 1. Adds the user message to history
    /// 2. Trims history if needed to fit within `max_context_tokens`
    /// 3. Builds the prompt from the full conversation
    /// 4. Runs inference via the provided `LlmService`
    /// 5. Adds the assistant response to history
    /// 6. Returns the inference response
    ///
    /// # Errors
    ///
    /// Returns an error if prompt building or inference fails.
    pub fn chat(
        &mut self,
        user_message: &str,
        llm: &dyn crate::service::LlmService,
        params: &GenerationParams,
    ) -> Result<InferenceResponse, crate::error::LlmError> {
        self.add_user_message(user_message);
        self.trim_history();

        let prompt = self.build_prompt()?;

        let cancel_flag = std::sync::atomic::AtomicBool::new(false);
        let (response, new_cache_state) =
            llm.infer_cached(&prompt, params, self.kv_cache_state, Some(&cancel_flag))?;

        // Update KV cache state. If the backend returned 0 (not supported),
        // we keep 0 — no caching benefit, but also no harm.
        self.kv_cache_state = new_cache_state;

        self.add_assistant_message(&response.text);

        Ok(response)
    }

    /// Run streaming inference for the next user message.
    ///
    /// Similar to [`chat`](Self::chat), but returns a receiver for streaming
    /// token events. The caller must collect the tokens and add the complete
    /// assistant response to the session via [`add_assistant_message`](Self::add_assistant_message).
    ///
    /// # Note
    ///
    /// Unlike `chat()`, this method does NOT automatically add the assistant
    /// response to history. The caller should collect all tokens, concatenate
    /// them, and call `add_assistant_message()` with the full response text.
    ///
    /// # Errors
    ///
    /// Returns an error if prompt building or inference initiation fails.
    pub fn chat_stream(
        &mut self,
        user_message: &str,
        llm: &dyn crate::service::LlmService,
        params: &GenerationParams,
    ) -> Result<std::sync::mpsc::Receiver<StreamEvent>, crate::error::LlmError> {
        self.add_user_message(user_message);
        self.trim_history();

        let prompt = self.build_prompt()?;

        let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (rx, new_cache_state) =
            llm.infer_stream_cached(&prompt, params, self.kv_cache_state, Some(cancel_flag))?;

        // For streaming backends, the cache state comes from a new context
        // each time, so it's typically 0. Still track it for consistency.
        self.kv_cache_state = new_cache_state;

        Ok(rx)
    }

    /// Build the prompt for the current session state.
    ///
    /// Constructs the full message list including system prompt,
    /// history, and the next assistant generation prompt.
    ///
    /// # Errors
    ///
    /// Returns an error if the template cannot format the messages.
    pub fn build_prompt(&self) -> Result<String, crate::error::LlmError> {
        let mut messages = Vec::new();

        if let Some(sys) = &self.system_prompt {
            messages.push(ChatMessage::system(sys.as_str()));
        }

        let messages_to_include = self.trimmed_messages();
        messages.extend_from_slice(messages_to_include);

        self.template.format(&messages, true)
    }

    /// Build the prompt including an explicit user message at the end.
    ///
    /// Unlike `build_prompt()`, this does NOT require the user message to
    /// already be in the session history. Useful for previewing what the
    /// prompt would look like without modifying the session.
    ///
    /// # Errors
    ///
    /// Returns an error if the template cannot format the messages.
    pub fn build_prompt_with(
        &self,
        user_message: &str,
    ) -> Result<String, crate::error::LlmError> {
        let mut messages = Vec::new();

        if let Some(sys) = &self.system_prompt {
            messages.push(ChatMessage::system(sys.as_str()));
        }

        let messages_to_include = self.trimmed_messages();
        messages.extend_from_slice(messages_to_include);
        messages.push(ChatMessage::user(user_message));

        self.template.format(&messages, true)
    }

    /// Get the messages that would be included in the prompt after trimming.
    fn trimmed_messages(&self) -> &[ChatMessage] {
        if self.max_history_turns > 0 {
            let max_msgs = self.max_history_turns * 2;
            if self.messages.len() > max_msgs {
                return &self.messages[self.messages.len() - max_msgs..];
            }
        }
        &self.messages[..]
    }

    /// Recalculate the estimated token count.
    fn recalculate_tokens(&mut self) {
        self.estimated_tokens = self.estimate_tokens();
    }

    /// Estimate the token count for current messages.
    ///
    /// Uses [`token_estimate`] per message plus a per-message overhead
    /// for role markers and template formatting.
    fn estimate_tokens(&self) -> usize {
        let mut total = token_estimate_messages(&self.messages);
        if let Some(sys) = &self.system_prompt {
            total += token_estimate(sys) + 4;
        }
        total
    }

    /// Trim history to fit within the token budget.
    ///
    /// Removes oldest user+assistant pairs until the estimated
    /// token count fits within `max_context_tokens - reserve_for_response`.
    ///
    /// The system prompt is always preserved (never trimmed).
    /// If `max_context_tokens` is 0, no trimming is performed.
    pub fn trim_history(&mut self) {
        if self.max_context_tokens == 0 {
            return;
        }

        let budget = self.max_context_tokens.saturating_sub(self.reserve_for_response);
        let system_tokens = self
            .system_prompt
            .as_ref()
            .map_or(0, |s| token_estimate(s) + 4);
        let history_budget = budget.saturating_sub(system_tokens);

        if history_budget == 0 {
            self.messages.clear();
            self.recalculate_tokens();
            return;
        }

        while self.messages.len() > 1 {
            let current = self.estimate_message_tokens(&self.messages);
            if current <= history_budget {
                break;
            }
            self.messages.remove(0);
        }

        self.recalculate_tokens();
    }

    /// Estimate tokens for a specific slice of messages.
    fn estimate_message_tokens(&self, messages: &[ChatMessage]) -> usize {
        token_estimate_messages(messages)
    }

    /// Returns the number of messages in the session (excluding system prompt).
    #[must_use]
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns whether the session has no messages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Returns the estimated token count.
    #[must_use]
    pub fn estimated_tokens(&self) -> usize {
        self.estimated_tokens
    }

    /// Returns a reference to the session messages.
    #[must_use]
    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Returns the session ID.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Returns the chat template.
    #[must_use]
    pub fn template(&self) -> &ChatTemplate {
        &self.template
    }

    /// Returns the max context token limit.
    #[must_use]
    pub const fn context_token_limit(&self) -> usize {
        self.max_context_tokens
    }

    /// Returns the number of tokens reserved for the model response.
    #[must_use]
    pub const fn response_token_reserve(&self) -> usize {
        self.reserve_for_response
    }

    /// Returns the number of complete turns (user+assistant pairs).
    #[must_use]
    pub fn turn_count(&self) -> usize {
        let user_count = self.messages.iter().filter(|m| m.role == "user").count();
        let assistant_count = self.messages.iter().filter(|m| m.role == "assistant").count();
        user_count.min(assistant_count)
    }

    /// Returns the current KV cache state for multi-turn inference.
    ///
    /// Zero means no cache is active. Non-zero values are opaque
    /// counters managed by the LLM backend.
    #[must_use]
    pub fn kv_cache_state(&self) -> crate::service::KvCacheState {
        self.kv_cache_state
    }

    /// Invalidate the KV cache state.
    ///
    /// Call this when the conversation context has been modified outside
    /// of `chat()` / `chat_stream()` (e.g., after adding or removing
    /// messages manually). The next inference call will start fresh.
    pub fn invalidate_kv_cache(&mut self) {
        self.kv_cache_state = 0;
    }

    /// Serialize the session state to JSON for persistence.
    ///
    /// Returns a JSON string containing the session ID, system prompt,
    /// and all messages. The chat template is NOT serialized (it should
    /// be re-specified when restoring).
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String, crate::error::LlmError> {
        let state = SessionState {
            session_id: self.session_id.clone(),
            system_prompt: self.system_prompt.clone(),
            messages: self.messages.clone(),
        };
        serde_json::to_string_pretty(&state).map_err(|e| {
            crate::error::LlmError::Internal(format!("Failed to serialize session: {e}"))
        })
    }

    /// Restore a session from a JSON string.
    ///
    /// The chat template must be provided separately since it is not
    /// serialized.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    pub fn from_json(
        json: &str,
        template: ChatTemplate,
    ) -> Result<Self, crate::error::LlmError> {
        let state: SessionState = serde_json::from_str(json).map_err(|e| {
            crate::error::LlmError::Internal(format!("Failed to deserialize session: {e}"))
        })?;

        let mut session = Self {
            session_id: state.session_id,
            system_prompt: state.system_prompt,
            messages: state.messages,
            max_history_turns: 0,
            template,
            estimated_tokens: 0,
            max_context_tokens: 0,
            reserve_for_response: 512,
            kv_cache_state: 0,
        };
        session.recalculate_tokens();
        Ok(session)
    }
}

/// Serializable session state for persistence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SessionState {
    session_id: String,
    system_prompt: Option<String>,
    messages: Vec<ChatMessage>,
}

/// Generate a simple unique ID for sessions.
///
/// NOT a full UUID implementation — just enough for uniqueness.
fn uuid_like_id() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("session-{:x}-{:x}", now.as_secs(), now.subsec_nanos())
}
