//! Chat template formats for different model families.
//!
//! [`ChatTemplate`] defines supported chat template variants and provides
//! pure-Rust formatting for built-in templates. Custom jinja templates
//! require the `llama.cpp` template engine and cannot be formatted in pure Rust.

use crate::chat::message::ChatMessage;

/// Supported chat template variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Auto-detect from model metadata (falls back to `ChatML` if not found).
    ///
    /// When no model metadata is available, `Auto` behaves identically to
    /// `ChatML` to maximize portability. Callers that need a different
    /// fallback can explicitly select [`Self::Raw`] or another template.
    Auto,

    /// Raw text, no template formatting applied.
    Raw,

    /// `ChatML` format (used by many models).
    /// Format: `<|im_start|>role\ncontent<|im_end|>\n`
    ChatML,

    /// Llama 3 format.
    /// Format: `<|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
    Llama3,

    /// DeepSeek 2.5 / V3 format.
    /// Format: `<｜begin▁of▁sentence｜>system\n\n<｜User｜>content<｜Assistant｜>`
    DeepSeek,

    /// Phi-3 / Phi-4 format (Microsoft).
    /// Format: `<|system|>\n{content}<|end|>\n<|user|>\n{content}<|end|>\n<|assistant|>\n`
    Phi,

    /// Gemma / Gemma 2 format (Google).
    /// Format: `<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n{content}<end_of_turn>\n`
    Gemma,

    /// Command R / Command R+ format (Cohere).
    /// Format: `<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{content}<|END_OF_TURN_TOKEN|>`
    CommandR,

    /// Custom jinja template string.
    Custom(String),
}

impl ChatTemplate {
    /// Returns the template name used for `llama_chat_apply_template`.
    #[must_use]
    pub const fn template_name(&self) -> Option<&str> {
        match self {
            Self::Auto => None, // use model default
            Self::Raw => Some("raw"),
            Self::ChatML => Some("chatml"),
            Self::Llama3 => Some("llama3"),
            Self::DeepSeek => Some("deepseek"),
            Self::Phi => Some("phi3"),
            Self::Gemma => Some("gemma"),
            Self::CommandR => Some("command-r"),
            Self::Custom(_) => None, // handled separately
        }
    }

    /// Returns the custom jinja template string, if any.
    #[must_use]
    pub const fn custom_template(&self) -> Option<&str> {
        match self {
            Self::Custom(tmpl) => Some(tmpl.as_str()),
            _ => None,
        }
    }

    /// Format a list of chat messages into a prompt string using this template.
    ///
    /// This is a pure-Rust implementation that does not require `llama.cpp` FFI.
    /// It supports `Auto` (falls back to `ChatML`), `Raw`, `ChatML`, `Llama3`,
    /// and `DeepSeek`.
    ///
    /// For `Custom` templates, this returns an error — callers should use
    /// the FFI-based `LlamaService::format_prompt` instead.
    ///
    /// # Arguments
    ///
    /// * `messages` — Ordered list of chat messages (system, user, assistant).
    /// * `add_generation_prompt` — If `true`, appends the assistant prefix
    ///   so the model continues generating from there.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::InvalidParameter`] for `Custom` templates (which
    /// require jinja evaluation via `llama.cpp`).
    pub fn format(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, crate::error::LlmError> {
        match self {
            Self::Raw => {
                // Raw: concatenate message contents with newlines
                let prompt: String = messages
                    .iter()
                    .map(|m| m.content.as_str())
                    .collect::<Vec<&str>>()
                    .join("\n\n");
                Ok(prompt)
            }
            Self::ChatML => Ok(Self::format_chatml(messages, add_generation_prompt)),
            Self::Llama3 => Ok(Self::format_llama3(messages, add_generation_prompt)),
            Self::DeepSeek => Ok(Self::format_deepseek(messages, add_generation_prompt)),
            Self::Phi => Ok(Self::format_phi(messages, add_generation_prompt)),
            Self::Gemma => Ok(Self::format_gemma(messages, add_generation_prompt)),
            Self::CommandR => Ok(Self::format_command_r(messages, add_generation_prompt)),
            Self::Auto => {
                // Auto falls back to ChatML — in production use, the model's
                // own template (discovered via GGUF metadata) is preferred,
                // but this provides a safe default. Users needing a different
                // fallback can select `Raw` or another template explicitly.
                Ok(Self::format_chatml(messages, add_generation_prompt))
            }
            Self::Custom(tmpl) => {
                // Custom jinja templates require llama.cpp's template engine.
                // We can't evaluate jinja in pure Rust, so return an error.
                Err(crate::error::LlmError::InvalidParameter(format!(
                    "Custom jinja templates require the llama.cpp template engine. \
                     Template: {tmpl}"
                )))
            }
        }
    }

    /// Format messages using ChatML template.
    ///
    /// ChatML format:
    /// ```text
    /// <|im_start|>system
    /// {content}<|im_end|>
    /// <|im_start|>user
    /// {content}<|im_end|>
    /// <|im_start|>assistant
    /// {content}<|im_end|>
    /// <|im_start|>assistant
    /// ```
    fn format_chatml(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let mut prompt = String::new();

        for msg in messages {
            prompt.push_str("<|im_start|>");
            prompt.push_str(&msg.role);
            prompt.push('\n');
            prompt.push_str(&msg.content);
            prompt.push_str("<|im_end|>\n");
        }

        if add_generation_prompt {
            prompt.push_str("<|im_start|>assistant\n");
        }

        prompt
    }

    /// Format messages using Llama 3 template.
    ///
    /// Llama 3 format:
    /// ```text
    /// <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    ///
    /// {content}<|eot_id|><|start_header_id|>user<|end_header_id|>
    ///
    /// {content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    ///
    /// {content}<|eot_id|>
    /// ```
    fn format_llama3(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let mut prompt = String::from("<|begin_of_text|>");

        for msg in messages {
            prompt.push_str("<|start_header_id|>");
            prompt.push_str(&msg.role);
            prompt.push_str("<|end_header_id|>\n\n");
            prompt.push_str(&msg.content);
            prompt.push_str("<|eot_id|>");
        }

        if add_generation_prompt {
            prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }

        prompt
    }

    /// Format messages using DeepSeek 2.5 / V3 template.
    ///
    /// This mirrors the `deepseek3` template used by `llama.cpp`:
    /// ```text
    /// <｜begin▁of▁sentence｜>system content
    ///
    /// <｜User｜>user content<｜Assistant｜>assistant content<｜end▁of▁sentence｜>
    /// <｜Assistant｜>
    /// ```
    fn format_deepseek(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let mut prompt = String::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    prompt.push_str("<｜begin▁of▁sentence｜>");
                    prompt.push_str(&msg.content);
                    prompt.push_str("\n\n");
                }
                "user" => {
                    prompt.push_str("<｜User｜>");
                    prompt.push_str(&msg.content);
                }
                _ => {
                    // assistant and any other role
                    prompt.push_str("<｜Assistant｜>");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<｜end▁of▁sentence｜>");
                }
            }
        }

        if add_generation_prompt {
            prompt.push_str("<｜Assistant｜>");
        }

        prompt
    }

    /// Format messages using Phi-3 / Phi-4 template (Microsoft).
    ///
    /// Phi format:
    /// ```text
    /// <|system|>
    /// {content}<|end|>
    /// <|user|>
    /// {content}<|end|>
    /// <|assistant|>
    /// {content}<|end|>
    /// ```
    fn format_phi(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let mut prompt = String::new();

        for msg in messages {
            prompt.push_str("<|");
            prompt.push_str(&msg.role);
            prompt.push_str("|>\n");
            prompt.push_str(&msg.content);
            prompt.push_str("<|end|>\n");
        }

        if add_generation_prompt {
            prompt.push_str("<|assistant|>\n");
        }

        prompt
    }

    /// Format messages using Gemma / Gemma 2 template (Google).
    ///
    /// Gemma format:
    /// ```text
    /// <start_of_turn>user
    /// {content}<end_of_turn>
    /// <start_of_turn>model
    /// {content}<end_of_turn>
    /// <start_of_turn>model
    /// ```
    ///
    /// Note: Gemma does not have a dedicated system role; system messages
    /// are prepended to the first user message.
    fn format_gemma(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let mut prompt = String::new();
        let mut system_content: Option<&str> = None;

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    system_content = Some(&msg.content);
                }
                "user" => {
                    prompt.push_str("<start_of_turn>user\n");
                    if let Some(sys) = system_content.take() {
                        prompt.push_str(sys);
                        prompt.push_str("\n\n");
                    }
                    prompt.push_str(&msg.content);
                    prompt.push_str("<end_of_turn>\n");
                }
                _ => {
                    prompt.push_str("<start_of_turn>model\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<end_of_turn>\n");
                }
            }
        }

        if add_generation_prompt {
            prompt.push_str("<start_of_turn>model\n");
        }

        prompt
    }

    /// Format messages using Command R / Command R+ template (Cohere).
    ///
    /// Command R format:
    /// ```text
    /// <|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{content}<|END_OF_TURN_TOKEN|>
    /// <|START_OF_TURN_TOKEN|><|USER_TOKEN|>{content}<|END_OF_TURN_TOKEN|>
    /// <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{content}<|END_OF_TURN_TOKEN|>
    /// <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
    /// ```
    fn format_command_r(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let mut prompt = String::new();

        for msg in messages {
            prompt.push_str("<|START_OF_TURN_TOKEN|>");
            match msg.role.as_str() {
                "system" => {
                    prompt.push_str("<|SYSTEM_TOKEN|>");
                }
                "user" => {
                    prompt.push_str("<|USER_TOKEN|>");
                }
                _ => {
                    prompt.push_str("<|CHATBOT_TOKEN|>");
                }
            }
            prompt.push_str(&msg.content);
            prompt.push_str("<|END_OF_TURN_TOKEN|>");
        }

        if add_generation_prompt {
            prompt.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
        }

        prompt
    }
}
