use crate::errors::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryMessageRole {
    System,
    User,
    Assistant,
}

impl MemoryMessageRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryMessageMeta {
    pub model_id: Option<String>,
    pub tokens_generated: Option<usize>,
    pub generation_time_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMessage {
    pub role: MemoryMessageRole,
    pub content: String,
}

#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn load_messages(&self, thread_id: &str) -> Result<Vec<MemoryMessage>>;

    async fn append_message(
        &self,
        thread_id: &str,
        role: MemoryMessageRole,
        content: String,
        meta: MemoryMessageMeta,
    ) -> Result<()>;
}
