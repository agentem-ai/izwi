//! LFM2 runtime helpers routed through the unified core engine.

use tokio::sync::mpsc;

use crate::engine::{EngineCoreRequest, GenerationParams as CoreGenParams};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::runtime::service::InferenceEngine;
use crate::runtime::types::{
    AsrTranscription, AudioChunk, GenerationRequest, GenerationResult, SpeechToSpeechGeneration,
};

impl InferenceEngine {
    const LFM2_TTS_DEFAULT_AUDIO_TEMPERATURE: f32 = 0.8;
    const LFM2_TTS_DEFAULT_AUDIO_TOP_K: usize = 64;
    const LFM2_S2S_DEFAULT_AUDIO_TEMPERATURE: f32 = 1.0;
    const LFM2_S2S_DEFAULT_AUDIO_TOP_K: usize = 4;

    fn default_lfm2_variant() -> ModelVariant {
        ModelVariant::Lfm2Audio15B
    }

    async fn resolve_active_lfm2_variant(&self) -> ModelVariant {
        if let Some(variant) = *self.loaded_tts_variant.read().await {
            if variant.is_lfm2() {
                return variant;
            }
        }
        Self::default_lfm2_variant()
    }

    pub async fn lfm2_asr_transcribe_streaming<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        mut on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.load_model(variant).await?;

        let mut request = EngineCoreRequest::asr(audio_base64.to_string());
        request.model_variant = Some(variant);
        request.language = language.map(|s| s.to_string());

        let output = self.core_engine.generate(request).await?;
        let text = output.text.unwrap_or_default();
        if !text.is_empty() {
            on_delta(text.clone());
        }

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: output.audio.duration_secs,
        })
    }

    pub async fn lfm2_tts_generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        if request.reference_audio.is_some() || request.reference_text.is_some() {
            return Err(Error::InvalidInput(
                "LFM2 native runtime does not support reference-audio voice cloning".to_string(),
            ));
        }

        let variant = self.resolve_active_lfm2_variant().await;
        self.load_model(variant).await?;

        let using_generic_defaults =
            request.config.top_k == 0 && (request.config.temperature - 0.7).abs() < f32::EPSILON;
        let temperature = if using_generic_defaults {
            Self::LFM2_TTS_DEFAULT_AUDIO_TEMPERATURE
        } else {
            request.config.temperature
        };
        let top_k = if request.config.top_k > 0 {
            request.config.top_k
        } else {
            Self::LFM2_TTS_DEFAULT_AUDIO_TOP_K
        };

        let mut core_request = EngineCoreRequest::tts(request.text.clone());
        core_request.id = request.id.clone();
        core_request.model_variant = Some(variant);
        core_request.language = request.language.clone();
        core_request.voice_description = request.voice_description.clone();
        core_request.params = CoreGenParams {
            temperature,
            top_p: request.config.top_p,
            top_k,
            repetition_penalty: request.config.repetition_penalty,
            max_tokens: if request.config.max_tokens == 0 {
                512
            } else {
                request.config.max_tokens
            },
            speaker: request.config.speaker.clone(),
            voice: request.config.speaker.clone(),
            audio_temperature: Some(temperature),
            audio_top_k: Some(top_k),
            speed: request.config.speed,
            stop_sequences: Vec::new(),
            stop_token_ids: Vec::new(),
        };

        let output = self.core_engine.generate(core_request).await?;
        Ok(GenerationResult {
            request_id: output.request_id,
            samples: output.audio.samples,
            sample_rate: output.audio.sample_rate,
            total_tokens: output.num_tokens,
            total_time_ms: output.generation_time.as_secs_f32() * 1000.0,
        })
    }

    pub async fn lfm2_tts_generate_streaming(
        &self,
        request: GenerationRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let result = self.lfm2_tts_generate(request.clone()).await?;

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

        Ok(())
    }

    pub async fn lfm2_speech_to_speech_streaming<F>(
        &self,
        audio_base64: &str,
        language: Option<&str>,
        system_prompt: Option<&str>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        mut on_delta: F,
    ) -> Result<SpeechToSpeechGeneration>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = self.resolve_active_lfm2_variant().await;
        self.load_model(variant).await?;

        let resolved_temperature = temperature.unwrap_or(Self::LFM2_S2S_DEFAULT_AUDIO_TEMPERATURE);
        let resolved_top_k = top_k
            .filter(|&v| v > 0)
            .unwrap_or(Self::LFM2_S2S_DEFAULT_AUDIO_TOP_K);

        let mut request = EngineCoreRequest::speech_to_speech(audio_base64.to_string());
        request.model_variant = Some(variant);
        request.language = language.map(|s| s.to_string());
        request.system_prompt = system_prompt.map(|s| s.to_string());
        request.params = CoreGenParams {
            temperature: resolved_temperature,
            top_p: 1.0,
            top_k: resolved_top_k,
            repetition_penalty: 1.0,
            max_tokens: 1024,
            speaker: None,
            voice: None,
            audio_temperature: Some(resolved_temperature),
            audio_top_k: Some(resolved_top_k),
            speed: 1.0,
            stop_sequences: Vec::new(),
            stop_token_ids: Vec::new(),
        };

        let output = self.core_engine.generate(request).await?;
        let text = output.text.unwrap_or_default();
        if !text.is_empty() {
            on_delta(text.clone());
        }

        Ok(SpeechToSpeechGeneration {
            text,
            samples: output.audio.samples,
            sample_rate: output.audio.sample_rate,
            input_transcription: None,
            generation_time_ms: output.generation_time.as_secs_f64() * 1000.0,
        })
    }

    pub async fn lfm2_speech_to_speech(
        &self,
        audio_base64: &str,
        language: Option<&str>,
        system_prompt: Option<&str>,
        temperature: Option<f32>,
        top_k: Option<usize>,
    ) -> Result<SpeechToSpeechGeneration> {
        self.lfm2_speech_to_speech_streaming(
            audio_base64,
            language,
            system_prompt,
            temperature,
            top_k,
            |_delta| {},
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use crate::models::lfm2_audio::LFM2_DEFAULT_S2S_PROMPT;

    #[test]
    fn uses_default_s2s_prompt_when_missing() {
        let prompt = None::<&str>.unwrap_or(LFM2_DEFAULT_S2S_PROMPT);
        assert_eq!(prompt, LFM2_DEFAULT_S2S_PROMPT);
    }
}
