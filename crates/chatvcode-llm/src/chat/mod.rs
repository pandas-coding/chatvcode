//! Chat template, message, prompt builder, and session management.
//!
//! This module contains all types related to chat-style interaction with
//! LLM models:
//!
//! - [`ChatMessage`] — a single role-tagged message in a conversation
//! - [`ChatTemplate`] — supported chat template formats (ChatML, Llama3, ...)
//! - [`ChatPromptBuilder`] — RAG-aware single-turn prompt construction
//! - [`ChatSession`] — multi-turn conversation with KV cache tracking
//!
//! ## Template Formatting
//!
//! All built-in templates (`ChatML`, `Llama3`, `DeepSeek`) are implemented
//! in pure Rust and do not require `llama.cpp` FFI. Only `Custom` (jinja)
//! templates require the C template engine.
//!
//! ## Session Persistence
//!
//! [`ChatSession`] supports JSON serialization via [`ChatSession::to_json`]
//! and [`ChatSession::from_json`] for conversation persistence across
//! program restarts.

pub mod message;
pub mod prompt;
pub mod session;
pub mod template;

pub use message::ChatMessage;
pub use prompt::ChatPromptBuilder;
pub use session::ChatSession;
pub use template::ChatTemplate;

// Token estimation
// ---------------------------------------------------------------------------

/// Estimate the number of tokens in a text string.
///
/// Uses a heuristic of ~4 characters per token, which is a reasonable
/// approximation for most Latin-script text. CJK characters and special
/// tokens may use fewer characters per token.
///
/// This is useful for:
/// - Pre-checking whether a prompt fits within a context window
/// - Deciding when to trim conversation history
/// - Budget management for RAG context injection
#[must_use]
pub fn token_estimate(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let total_chars = text.len();
    (total_chars + 3) / 4
}

/// Estimate the number of tokens for a list of chat messages.
///
/// Includes a small overhead (~4 tokens per message) for role markers
/// and template formatting overhead.
#[must_use]
pub fn token_estimate_messages(messages: &[ChatMessage]) -> usize {
    const OVERHEAD_PER_MSG: usize = 4;
    messages
        .iter()
        .map(|m| token_estimate(&m.content) + OVERHEAD_PER_MSG)
        .sum()
}
