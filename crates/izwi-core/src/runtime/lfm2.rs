//! LFM2 runtime helpers (ASR, TTS, and speech-to-speech).

use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::chat_types::{ChatMessage, ChatRole};
use crate::models::lfm2_audio::{lfm2_tts_voice_prompt, Lfm2AudioModel, LFM2_DEFAULT_S2S_PROMPT};
use crate::models::qwen3_tts::{Qwen3TtsModel, TtsGenerationParams};
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

    async fn ensure_lfm2_tts_fallback_loaded(&self, variant: ModelVariant) -> Result<()> {
        {
            let loaded_variant = *self.lfm2_fallback_tts_variant.read().await;
            let model_guard = self.lfm2_fallback_tts_model.read().await;
            if loaded_variant == Some(variant) && model_guard.is_some() {
                return Ok(());
            }
        }

        let model_path = self
            .model_manager
            .get_model_info(variant)
            .await
            .and_then(|i| i.local_path)
            .ok_or_else(|| {
                Error::ModelNotFound(format!(
                    "LFM2 native runtime fallback TTS model '{}' is not downloaded",
                    variant
                ))
            })?;

        info!(
            "Loading LFM2 native fallback TTS model {} from {:?}",
            variant, model_path
        );

        let device = self.device.clone();
        let model = tokio::task::spawn_blocking(move || Qwen3TtsModel::load(&model_path, device))
            .await
            .map_err(|err| {
                Error::ModelLoadError(format!("LFM2 fallback TTS task failed: {}", err))
            })??;

        {
            let mut model_guard = self.lfm2_fallback_tts_model.write().await;
            *model_guard = Some(model);
        }
        {
            let mut variant_guard = self.lfm2_fallback_tts_variant.write().await;
            *variant_guard = Some(variant);
        }

        self.model_manager.mark_loaded(variant).await;
        Ok(())
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
        self.asr_transcribe_with_variant_streaming(
            model.asr_fallback_variant(),
            audio_base64,
            language,
            on_delta,
        )
        .await
    }

    pub async fn lfm2_tts_generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        if request.reference_audio.is_some() || request.reference_text.is_some() {
            return Err(Error::InvalidInput(
                "LFM2 native runtime does not support reference-audio voice cloning".to_string(),
            ));
        }

        let model = self.get_or_load_lfm2_model().await?;
        let fallback_variant = model.tts_fallback_variant();
        self.ensure_lfm2_tts_fallback_loaded(fallback_variant)
            .await?;

        let started = std::time::Instant::now();
        let request_id = request.id.clone();
        let voice_instruction =
            lfm2_tts_voice_prompt(request.config.speaker.as_deref()).to_string();

        let samples = tokio::task::spawn_blocking({
            let tts_store = self.lfm2_fallback_tts_model.clone();
            let text = request.text.clone();
            let language = request.language.clone();
            let temperature = request.config.temperature;
            let top_k = if request.config.top_k == 0 {
                50
            } else {
                request.config.top_k
            };
            move || {
                let rt = tokio::runtime::Handle::try_current().map_err(|_| {
                    Error::InferenceError(
                        "No async runtime available for LFM2 fallback TTS".to_string(),
                    )
                })?;
                let model_guard = rt.block_on(async { tts_store.read().await });
                let model = model_guard.as_ref().ok_or_else(|| {
                    Error::InferenceError("LFM2 fallback TTS model was not initialized".to_string())
                })?;

                let params = TtsGenerationParams {
                    temperature,
                    top_p: 1.0,
                    top_k,
                    repetition_penalty: 1.05,
                    max_frames: 512,
                };

                model.generate_with_text_params(
                    &text,
                    language.as_deref(),
                    Some(voice_instruction.as_str()),
                    &params,
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
        mut on_delta: F,
    ) -> Result<SpeechToSpeechGeneration>
    where
        F: FnMut(String) + Send + 'static,
    {
        let started = std::time::Instant::now();

        let transcription = self
            .lfm2_asr_transcribe_streaming(audio_base64, language, |_delta| {})
            .await?;

        let model = self.get_or_load_lfm2_model().await?;
        let prompt = system_prompt.unwrap_or(LFM2_DEFAULT_S2S_PROMPT);
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: prompt.to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: transcription.text.clone(),
            },
        ];

        let response_text = match self
            .chat_generate(model.chat_fallback_variant(), messages, 256)
            .await
        {
            Ok(chat) if !chat.text.trim().is_empty() => chat.text,
            Ok(_) => transcription.text.clone(),
            Err(err) => {
                warn!("LFM2 fallback chat failed: {}", err);
                transcription.text.clone()
            }
        };

        if !response_text.trim().is_empty() {
            on_delta(response_text.clone());
        }

        let mut tts_request = GenerationRequest::new(response_text.clone());
        tts_request.language = language.map(|value| value.to_string());
        if let Some(value) = temperature {
            tts_request.config.temperature = value;
        }
        if let Some(value) = top_k {
            tts_request.config.top_k = value;
        }

        let tts_output = self.lfm2_tts_generate(tts_request).await?;

        Ok(SpeechToSpeechGeneration {
            text: response_text,
            samples: tts_output.samples,
            sample_rate: tts_output.sample_rate,
            input_transcription: Some(transcription.text),
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
