//! Text-to-speech runtime methods.

use tokio::sync::mpsc;
use tracing::info;

use crate::engine::{EngineCoreRequest, GenerationParams as CoreGenParams};
use crate::error::{Error, Result};
use crate::runtime::service::InferenceEngine;
use crate::runtime::types::{AudioChunk, GenerationConfig, GenerationRequest, GenerationResult};

impl InferenceEngine {
    /// Generate audio from text using the unified core engine.
    pub async fn generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        let loaded_variant = *self.loaded_tts_variant.read().await;
        if loaded_variant
            .map(|variant| variant.is_lfm2())
            .unwrap_or(false)
        {
            return self.lfm2_tts_generate(request).await;
        }

        let mut core_request = EngineCoreRequest::tts(request.text.clone());
        core_request.id = request.id.clone();
        core_request.model_variant = loaded_variant;
        core_request.language = request.language.clone();
        core_request.reference_audio = request.reference_audio.clone();
        core_request.reference_text = request.reference_text.clone();
        core_request.voice_description = request.voice_description.clone();
        core_request.params = core_params_from_generation(&request.config);

        let output = self.core_engine.generate(core_request).await?;
        let samples = output.audio.samples;
        let sample_rate = output.audio.sample_rate;
        let total_tokens = output.num_tokens;
        let total_time_ms = output.generation_time.as_secs_f32() * 1000.0;

        info!(
            "Generated {} samples in {:.1}ms via core engine",
            samples.len(),
            total_time_ms
        );

        Ok(GenerationResult {
            request_id: output.request_id,
            samples,
            sample_rate,
            total_tokens,
            total_time_ms,
        })
    }

    /// Generate audio with streaming output.
    ///
    /// Streaming is emitted from engine outputs in chunked form so all synthesis
    /// execution still routes through the core engine.
    pub async fn generate_streaming(
        &self,
        request: GenerationRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let loaded_variant = *self.loaded_tts_variant.read().await;
        if loaded_variant
            .map(|variant| variant.is_lfm2())
            .unwrap_or(false)
        {
            return self.lfm2_tts_generate_streaming(request, chunk_tx).await;
        }

        let result = self.generate(request.clone()).await?;
        if result.samples.is_empty() {
            return Ok(());
        }

        let chunk_samples = (self.config.chunk_size.max(1) * 200).clamp(1200, 9600);
        let total_chunks = result.samples.len().div_ceil(chunk_samples);

        for (index, samples) in result.samples.chunks(chunk_samples).enumerate() {
            let mut chunk = AudioChunk::new(request.id.clone(), index, samples.to_vec());
            chunk.is_final = index + 1 >= total_chunks;
            chunk_tx.send(chunk).await.map_err(|_| {
                Error::InferenceError("Streaming output channel closed".to_string())
            })?;
        }

        info!("Streaming generation complete via core engine");
        Ok(())
    }
}

fn core_params_from_generation(config: &GenerationConfig) -> CoreGenParams {
    CoreGenParams {
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
        repetition_penalty: config.repetition_penalty,
        max_tokens: config.max_tokens,
        speaker: config.speaker.clone(),
        voice: config.speaker.clone(),
        audio_temperature: None,
        audio_top_k: None,
        speed: config.speed,
        stop_sequences: Vec::new(),
        stop_token_ids: Vec::new(),
    }
}
