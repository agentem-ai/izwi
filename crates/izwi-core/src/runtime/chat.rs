//! Chat runtime methods routed through the unified core engine.

use crate::engine::EngineCoreRequest;
use crate::error::Result;
use crate::model::ModelVariant;
use crate::models::chat_types::ChatMessage;
use crate::runtime::service::RuntimeService;
use crate::runtime::types::ChatGeneration;

impl RuntimeService {
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

        let output = self.run_request(request).await?;
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
        self.load_model(variant).await?;

        let mut request = EngineCoreRequest::chat(messages);
        request.model_variant = Some(variant);
        request.params.max_tokens = max_new_tokens.max(1);

        let mut streamed_text = String::new();
        let output = self
            .run_streaming_request(request, |chunk| {
                if let Some(delta) = chunk.text {
                    if !delta.is_empty() {
                        streamed_text.push_str(&delta);
                        on_delta(delta);
                    }
                }
                std::future::ready(Ok(()))
            })
            .await?;

        Ok(ChatGeneration {
            text: output.text.unwrap_or(streamed_text),
            tokens_generated: output.num_tokens,
            generation_time_ms: output.generation_time.as_secs_f64() * 1000.0,
        })
    }
}
