//! Kokoro TTS runtime helpers (isolated from generic runtime routing).

use tokio::sync::mpsc;

use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::runtime::service::RuntimeService;
use crate::runtime::types::{AudioChunk, GenerationRequest, GenerationResult};

impl RuntimeService {
    fn default_kokoro_variant() -> ModelVariant {
        ModelVariant::Kokoro82M
    }

    async fn resolve_active_kokoro_variant(&self) -> ModelVariant {
        if let Some(variant) = *self.loaded_tts_variant.read().await {
            if matches!(variant.family(), crate::catalog::ModelFamily::KokoroTts) {
                return variant;
            }
        }
        Self::default_kokoro_variant()
    }

    pub async fn kokoro_tts_generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        let variant = self.resolve_active_kokoro_variant().await;
        self.load_model(variant).await?;
        let model = self
            .model_registry
            .get_kokoro(variant)
            .await
            .ok_or_else(|| Error::InferenceError("Kokoro model not loaded".to_string()))?;

        let opts = &request.config.options;
        let speaker = opts.speaker.as_deref().or(opts.voice.as_deref());
        let result = model.generate(
            &request.text,
            speaker,
            request.language.as_deref(),
            opts.speed,
        )?;

        Ok(GenerationResult {
            request_id: request.id,
            samples: result.samples,
            sample_rate: result.sample_rate,
            total_tokens: result.tokens_generated,
            total_time_ms: 0.0,
        })
    }

    pub async fn kokoro_tts_generate_streaming(
        &self,
        request: GenerationRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let request_id = request.id.clone();
        let result = self.kokoro_tts_generate(request).await?;
        let chunk = AudioChunk::final_chunk(request_id, 0, result.samples);
        chunk_tx
            .send(chunk)
            .await
            .map_err(|_| Error::InferenceError("Streaming output channel closed".to_string()))?;
        Ok(())
    }
}
