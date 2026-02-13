//! ASR runtime methods.

use crate::catalog::resolve_asr_model_variant;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::runtime::audio_io::{base64_decode, decode_wav_bytes};
use crate::runtime::service::InferenceEngine;
use crate::runtime::types::AsrTranscription;

impl InferenceEngine {
    /// Transcribe audio with Voxtral Realtime (native).
    pub async fn voxtral_transcribe(
        &self,
        audio_base64: &str,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.voxtral_transcribe_streaming(audio_base64, language, |_delta| {})
            .await
    }

    /// Transcribe audio with Voxtral Realtime and emit incremental deltas.
    pub async fn voxtral_transcribe_streaming<F>(
        &self,
        audio_base64: &str,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
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
        self.asr_transcribe_streaming(audio_base64, model_id, language, |_delta| {})
            .await
    }

    /// Transcribe audio with Qwen3-ASR and emit incremental deltas.
    pub async fn asr_transcribe_streaming<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);

        if variant.is_voxtral() {
            return self
                .voxtral_transcribe_streaming(audio_base64, language, on_delta)
                .await;
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
        .map_err(|e| Error::InferenceError(format!("ASR transcription task failed: {}", e)))??;

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: samples_len as f32 / sample_rate as f32,
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::config::EngineConfig;
    use std::sync::{Mutex, OnceLock};
    use uuid::Uuid;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[tokio::test]
    async fn parakeet_load_rejects_invalid_nemo_archive() {
        let _guard = env_lock().lock().expect("env lock poisoned");

        let root = std::env::temp_dir().join(format!("izwi-parakeet-runtime-{}", Uuid::new_v4()));
        let model_dir = root.join("Parakeet-TDT-0.6B-v2");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("parakeet-tdt-0.6b-v2.nemo"), b"mock-nemo").unwrap();

        let mut config = EngineConfig::default();
        config.models_dir = root.clone();
        config.use_metal = false;

        let engine = InferenceEngine::new(config).unwrap();
        let err = engine
            .load_model(ModelVariant::ParakeetTdt06BV2)
            .await
            .expect_err("invalid .nemo archive should fail to load");
        let msg = err.to_string();
        assert!(
            msg.contains("Failed to list .nemo archive")
                || msg.contains("Unrecognized archive format"),
            "unexpected error: {msg}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }
}
