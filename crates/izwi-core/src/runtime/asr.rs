//! ASR runtime methods.

use crate::catalog::resolve_asr_model_variant;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::runtime::audio_io::{base64_decode, decode_wav_bytes};
use crate::runtime::service::InferenceEngine;
use crate::runtime::types::AsrTranscription;
use tracing::warn;

impl InferenceEngine {
    /// Transcribe audio with Voxtral Realtime (native).
    pub async fn voxtral_transcribe(
        &self,
        audio_base64: &str,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = ModelVariant::VoxtralMini4BRealtime2602;

        let model = if let Some(model) = self.model_registry.get_voxtral(variant).await {
            model
        } else {
            let path = self
                .model_manager
                .get_model_info(variant)
                .await
                .and_then(|i| i.local_path)
                .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;
            self.model_registry.load_voxtral(variant, &path).await?
        };

        let (samples, sample_rate) = decode_wav_bytes(&base64_decode(audio_base64)?)?;
        let samples_len = samples.len();

        let text = tokio::task::spawn_blocking({
            let model = model.clone();
            let language = language.map(|s| s.to_string());
            move || model.transcribe(&samples, sample_rate, language.as_deref())
        })
        .await
        .map_err(|e| {
            Error::InferenceError(format!("Voxtral transcription task failed: {}", e))
        })??;

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: samples_len as f32 / sample_rate as f32,
        })
    }

    /// Transcribe audio with Qwen3-ASR (native).
    pub async fn asr_transcribe(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);

        if variant.is_voxtral() {
            return self.voxtral_transcribe(audio_base64, language).await;
        }

        let model = if let Some(model) = self.model_registry.get_asr(variant).await {
            model
        } else {
            let path = self
                .model_manager
                .get_model_info(variant)
                .await
                .and_then(|i| i.local_path)
                .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;
            self.model_registry.load_asr(variant, &path).await?
        };

        let (samples, sample_rate) = decode_wav_bytes(&base64_decode(audio_base64)?)?;
        let mut text = model.transcribe(&samples, sample_rate, language)?;

        // MLX-community quantized checkpoints can emit a degenerate empty result
        // on the native backend. Recover by retrying once with the matching
        // full-precision ASR model when available.
        if variant.is_quantized() && text.trim().is_empty() {
            if let Some(fallback_variant) = quantized_asr_fallback_variant(variant) {
                if let Some(fallback_path) = self
                    .model_manager
                    .get_model_info(fallback_variant)
                    .await
                    .and_then(|i| i.local_path)
                {
                    let fallback_model =
                        if let Some(model) = self.model_registry.get_asr(fallback_variant).await {
                            model
                        } else {
                            self.model_registry
                                .load_asr(fallback_variant, &fallback_path)
                                .await?
                        };

                    let fallback_text = fallback_model.transcribe(&samples, sample_rate, language)?;
                    if !fallback_text.trim().is_empty() {
                        warn!(
                            "ASR fallback recovered empty quantized output: {} -> {}",
                            variant,
                            fallback_variant
                        );
                        text = fallback_text;
                    }
                }
            }
        }

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: samples.len() as f32 / sample_rate as f32,
        })
    }

    /// Force alignment: align reference text with audio timestamps.
    pub async fn force_align(
        &self,
        audio_base64: &str,
        reference_text: &str,
    ) -> Result<Vec<(String, u32, u32)>> {
        let variant = ModelVariant::Qwen3ForcedAligner06B;

        let model = if let Some(model) = self.model_registry.get_asr(variant).await {
            model
        } else {
            let path = self
                .model_manager
                .get_model_info(variant)
                .await
                .and_then(|i| i.local_path)
                .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;
            self.model_registry.load_asr(variant, &path).await?
        };

        let (samples, sample_rate) = decode_wav_bytes(&base64_decode(audio_base64)?)?;
        let alignments = model.force_align(&samples, sample_rate, reference_text)?;

        Ok(alignments)
    }
}

fn quantized_asr_fallback_variant(variant: ModelVariant) -> Option<ModelVariant> {
    match variant {
        ModelVariant::Qwen3Asr06B4Bit | ModelVariant::Qwen3Asr06B8Bit => {
            Some(ModelVariant::Qwen3Asr06B)
        }
        ModelVariant::Qwen3Asr17B4Bit | ModelVariant::Qwen3Asr17B8Bit => {
            Some(ModelVariant::Qwen3Asr17B)
        }
        _ => None,
    }
}
