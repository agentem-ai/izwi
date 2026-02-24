use crate::errors::Result;
use crate::memory::MemoryMessage;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequest {
    pub model_id: String,
    pub messages: Vec<MemoryMessage>,
    pub max_output_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutput {
    pub text: String,
    pub tokens_generated: usize,
    pub generation_time_ms: f64,
}

#[async_trait]
pub trait ModelBackend: Send + Sync {
    async fn generate(&self, request: ModelRequest) -> Result<ModelOutput>;
}
