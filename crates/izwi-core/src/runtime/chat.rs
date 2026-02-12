//! Chat runtime methods.

use std::sync::Arc;

use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::chat_types::ChatMessage;
use crate::models::registry::NativeChatModel;
use crate::runtime::service::InferenceEngine;
use crate::runtime::types::ChatGeneration;

impl InferenceEngine {
    async fn get_or_load_chat_model(&self, variant: ModelVariant) -> Result<Arc<NativeChatModel>> {
        if !variant.is_chat() {
            return Err(Error::InvalidInput(format!(
                "Model {variant} is not a chat model"
            )));
        }

        if let Some(model) = self.model_registry.get_chat(variant).await {
            return Ok(model);
        }

        let path = self
            .model_manager
            .get_model_info(variant)
            .await
            .and_then(|i| i.local_path)
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;

        let model = self.model_registry.load_chat(variant, &path).await?;
        self.model_manager.mark_loaded(variant).await;
        Ok(model)
    }

    pub async fn chat_generate(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        max_new_tokens: usize,
    ) -> Result<ChatGeneration> {
        let model = self.get_or_load_chat_model(variant).await?;
        let started = std::time::Instant::now();

        let output = tokio::task::spawn_blocking(move || model.generate(&messages, max_new_tokens))
            .await
            .map_err(|e| Error::InferenceError(format!("Chat generation task failed: {}", e)))??;

        Ok(ChatGeneration {
            text: output.text,
            tokens_generated: output.tokens_generated,
            generation_time_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub async fn chat_generate_streaming<F>(
        &self,
        variant: ModelVariant,
        messages: Vec<ChatMessage>,
        max_new_tokens: usize,
        on_delta: F,
    ) -> Result<ChatGeneration>
    where
        F: FnMut(String) + Send + 'static,
    {
        let model = self.get_or_load_chat_model(variant).await?;
        let started = std::time::Instant::now();

        let output = tokio::task::spawn_blocking(move || {
            let mut callback = on_delta;
            let mut emit = |delta: &str| callback(delta.to_string());
            model.generate_with_callback(&messages, max_new_tokens, &mut emit)
        })
        .await
        .map_err(|e| Error::InferenceError(format!("Chat generation task failed: {}", e)))??;

        Ok(ChatGeneration {
            text: output.text,
            tokens_generated: output.tokens_generated,
            generation_time_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }
}
