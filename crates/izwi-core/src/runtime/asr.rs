//! ASR runtime methods routed through the unified core engine.

use crate::catalog::resolve_asr_model_variant;
use crate::engine::EngineCoreRequest;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::runtime::audio_io::{base64_decode, decode_wav_bytes};
use crate::runtime::service::RuntimeService;
use crate::runtime::types::AsrTranscription;

impl RuntimeService {
    pub(crate) async fn asr_transcribe_with_variant_streaming<F>(
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
        let text = output.text.unwrap_or(streamed_text);

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: output.audio.duration_secs,
        })
    }

    /// Transcribe audio with Voxtral Realtime.
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
        self.asr_transcribe_with_variant_streaming(
            ModelVariant::VoxtralMini4BRealtime2602,
            audio_base64,
            language,
            on_delta,
        )
        .await
    }

    /// Transcribe audio with native ASR models.
    pub async fn asr_transcribe(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_streaming(audio_base64, model_id, language, |_delta| {})
            .await
    }

    /// Transcribe audio and emit deltas.
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
        self.asr_transcribe_with_variant_streaming(variant, audio_base64, language, on_delta)
            .await
    }

    /// Force alignment remains a specialized operation not expressed by the
    /// generic engine output type.
    pub async fn force_align(
        &self,
        audio_base64: &str,
        reference_text: &str,
    ) -> Result<Vec<(String, u32, u32)>> {
        let variant = ModelVariant::Qwen3ForcedAligner06B;
        self.load_model(variant).await?;

        let model = self
            .model_registry
            .get_asr(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;

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

        let engine = RuntimeService::new(config).unwrap();
        let err = engine
            .load_model(ModelVariant::ParakeetTdt06BV2)
            .await
            .expect_err("invalid .nemo archive should fail to load");
        let msg = err.to_string();
        assert!(
            msg.contains(".nemo")
                || msg.contains("archive")
                || msg.contains("Failed to load")
                || msg.contains("invalid")
        );

        let _ = std::fs::remove_dir_all(root);
    }
}
