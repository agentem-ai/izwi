//! Chat runtime methods routed through the unified core engine.

use crate::engine::EngineCoreRequest;
use crate::error::Result;
use crate::model::ModelVariant;
use crate::models::chat_types::ChatMessage;
use crate::runtime::service::InferenceEngine;
use crate::runtime::types::ChatGeneration;

impl InferenceEngine {
    pub async fn chat_generate(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        max_new_tokens: usize,
    ) -> Result<ChatGeneration> {
        self.load_model(variant).await?;

        let mut request = EngineCoreRequest::chat(messages);
        request.model_variant = Some(variant);
        request.params.max_tokens = max_new_tokens.max(1);

        let output = self.core_engine.generate(request).await?;
        Ok(ChatGeneration {
            text: output.text.unwrap_or_default(),
            tokens_generated: output.num_tokens,
            generation_time_ms: output.generation_time.as_secs_f64() * 1000.0,
        })
    }

    pub async fn chat_generate_streaming<F>(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        max_new_tokens: usize,
        mut on_delta: F,
    ) -> Result<ChatGeneration>
    where
        F: FnMut(String) + Send + 'static,
    {
        let result = self
            .chat_generate(variant, messages, max_new_tokens)
            .await?;
        if !result.text.is_empty() {
            on_delta(result.text.clone());
        }
        Ok(result)
    }
}
