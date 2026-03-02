//! Shared chat message types across text-chat model families.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub fn as_prompt_role(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

const QWEN35_THINKING_CONTROL_PREFIX: &str = "__izwi_qwen35_enable_thinking=";

/// Internal control marker used by server/runtime to steer Qwen3.5 chat-template
/// thinking mode without exposing implementation details to users.
pub fn qwen35_thinking_control_content(enable_thinking: bool) -> String {
    format!("{QWEN35_THINKING_CONTROL_PREFIX}{enable_thinking}")
}

pub fn parse_qwen35_thinking_control_content(content: &str) -> Option<bool> {
    let raw = content.trim();
    let suffix = raw.strip_prefix(QWEN35_THINKING_CONTROL_PREFIX)?;
    match suffix.trim() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_qwen35_thinking_control_content, qwen35_thinking_control_content};

    #[test]
    fn qwen35_control_roundtrip_true() {
        let content = qwen35_thinking_control_content(true);
        assert_eq!(parse_qwen35_thinking_control_content(&content), Some(true));
    }

    #[test]
    fn qwen35_control_roundtrip_false() {
        let content = qwen35_thinking_control_content(false);
        assert_eq!(parse_qwen35_thinking_control_content(&content), Some(false));
    }

    #[test]
    fn qwen35_control_ignores_non_control_text() {
        assert_eq!(
            parse_qwen35_thinking_control_content("You are a helpful assistant."),
            None
        );
    }
}
