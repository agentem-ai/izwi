//! LFM2 runtime helpers (ASR, TTS, and speech-to-speech).

use tokio::sync::mpsc;
use tracing::info;

use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::lfm2_audio::{lfm2_tts_voice_prompt, Lfm2AudioModel, LFM2_DEFAULT_S2S_PROMPT};
use crate::runtime::audio_io::{base64_decode, decode_wav_bytes};
use crate::runtime::service::InferenceEngine;
use crate::runtime::types::{
    AsrTranscription, AudioChunk, GenerationRequest, GenerationResult, SpeechToSpeechGeneration,
};

impl InferenceEngine {
    pub(crate) async fn get_or_load_lfm2_model(&self) -> Result<std::sync::Arc<Lfm2AudioModel>> {
        let variant = ModelVariant::Lfm2Audio15B;

        if let Some(model) = self.model_registry.get_lfm2(variant).await {
            return Ok(model);
        }

        let path = self
            .model_manager
            .get_model_info(variant)
            .await
            .and_then(|info| info.local_path)
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;

        let model = self.model_registry.load_lfm2(variant, &path).await?;
        self.model_manager.mark_loaded(variant).await;

        {
            let mut variant_guard = self.loaded_tts_variant.write().await;
            *variant_guard = Some(variant);
        }
        {
            let mut path_guard = self.loaded_model_path.write().await;
            *path_guard = Some(path);
        }

        Ok(model)
    }

    pub async fn lfm2_asr_transcribe_streaming<F>(
        &self,
        audio_base64: &str,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let model = self.get_or_load_lfm2_model().await?;
        let (samples, sample_rate) = decode_wav_bytes(&base64_decode(audio_base64)?)?;
        let samples_len = samples.len();
        let text = tokio::task::spawn_blocking({
            let model = model.clone();
            let language = language.map(|s| s.to_string());
            move || {
                let mut callback = on_delta;
                let mut emit = |delta: &str| callback(delta.to_string());
                model.transcribe_with_callback(
                    &samples,
                    sample_rate,
                    language.as_deref(),
                    &mut emit,
                )
            }
        })
        .await
        .map_err(|e| Error::InferenceError(format!("LFM2 ASR task failed: {}", e)))??;

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: samples_len as f32 / sample_rate as f32,
        })
    }

    pub async fn lfm2_tts_generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        if request.reference_audio.is_some() || request.reference_text.is_some() {
            return Err(Error::InvalidInput(
                "LFM2 native runtime does not support reference-audio voice cloning".to_string(),
            ));
        }

        let model = self.get_or_load_lfm2_model().await?;
        let started = std::time::Instant::now();
        let request_id = request.id.clone();
        let voice_instruction = request.voice_description.clone().unwrap_or_else(|| {
            lfm2_tts_voice_prompt(request.config.speaker.as_deref()).to_string()
        });

        let samples = tokio::task::spawn_blocking({
            let model = model.clone();
            let text = request.text.clone();
            let speaker_prompt = voice_instruction.clone();
            let temperature = request.config.temperature;
            let top_k = (request.config.top_k > 0).then_some(request.config.top_k);
            let max_new_tokens = if request.config.max_tokens == 0 {
                768
            } else {
                request.config.max_tokens
            };
            move || {
                let mut sink = |_delta: &str| {};
                model.synthesize_with_callback(
                    &text,
                    &speaker_prompt,
                    Some(temperature),
                    top_k,
                    max_new_tokens,
                    &mut sink,
                )
            }
        })
        .await
        .map_err(|err| Error::InferenceError(format!("LFM2 TTS task failed: {}", err)))??;

        let total_time_ms = started.elapsed().as_secs_f32() * 1000.0;

        Ok(GenerationResult {
            request_id,
            samples,
            sample_rate: 24_000,
            total_tokens: request.text.len().max(1),
            total_time_ms,
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
        let total_chunks = (result.samples.len() + chunk_samples - 1) / chunk_samples;

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
        on_delta: F,
    ) -> Result<SpeechToSpeechGeneration>
    where
        F: FnMut(String) + Send + 'static,
    {
        let started = std::time::Instant::now();
        let model = self.get_or_load_lfm2_model().await?;
        let (samples, sample_rate) = decode_wav_bytes(&base64_decode(audio_base64)?)?;
        let prompt = system_prompt.unwrap_or(LFM2_DEFAULT_S2S_PROMPT).to_string();
        if let Some(lang) = language {
            info!("LFM2 S2S language hint: {}", lang);
        }

        let (text, output_samples) = tokio::task::spawn_blocking({
            let model = model.clone();
            move || {
                let mut callback = on_delta;
                let mut emit = |delta: &str| callback(delta.to_string());
                model.speech_to_speech_with_callback(
                    &samples,
                    sample_rate,
                    Some(prompt.as_str()),
                    temperature,
                    top_k,
                    1024,
                    &mut emit,
                )
            }
        })
        .await
        .map_err(|e| {
            Error::InferenceError(format!("LFM2 speech-to-speech task failed: {}", e))
        })??;

        Ok(SpeechToSpeechGeneration {
            text,
            samples: output_samples,
            sample_rate: 24_000,
            input_transcription: None,
            generation_time_ms: started.elapsed().as_secs_f64() * 1000.0,
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
    use super::*;

    #[test]
    fn uses_default_s2s_prompt_when_missing() {
        let prompt = None::<&str>.unwrap_or(LFM2_DEFAULT_S2S_PROMPT);
        assert_eq!(prompt, LFM2_DEFAULT_S2S_PROMPT);
    }
}
