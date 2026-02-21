use std::path::Path;

use tracing::info;

use crate::error::Result;
use crate::model::ModelVariant;
use crate::models::qwen3_tts::Qwen3TtsModel;
use crate::runtime::lifecycle::phases::PreparedLoad;
use crate::runtime::service::RuntimeService;
use crate::tokenizer::Tokenizer;

impl RuntimeService {
    /// Load a model for inference.
    pub async fn load_model(&self, variant: ModelVariant) -> Result<()> {
        let prepared = self.prepare_model_load(variant).await?;

        if variant.is_asr() || variant.is_forced_aligner() {
            self.model_registry
                .load_asr(variant, &prepared.model_path)
                .await?;
            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        if variant.is_diarization() {
            self.model_registry
                .load_diarization(variant, &prepared.model_path)
                .await?;
            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        if variant.is_chat() {
            self.model_registry
                .load_chat(variant, &prepared.model_path)
                .await?;
            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        if variant.is_lfm2() {
            self.model_registry
                .load_lfm2(variant, &prepared.model_path)
                .await?;

            // LFM2 owns active TTS routing and does not use the legacy Qwen slot.
            let mut tts_guard = self.tts_model.write().await;
            *tts_guard = None;
            drop(tts_guard);

            self.set_active_tts_variant(variant, prepared.model_path)
                .await;
            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        if variant.is_voxtral() {
            self.model_registry
                .load_voxtral(variant, &prepared.model_path)
                .await?;
            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        if variant.is_tts() {
            if self.is_tts_model_already_loaded(&prepared.model_path).await {
                self.model_manager.mark_loaded(variant).await;
                return Ok(());
            }

            info!("Loading native TTS model from {:?}", prepared.model_path);
            let tts_model = Qwen3TtsModel::load(
                &prepared.model_path,
                self.device.clone(),
                self.config.kv_page_size,
                &self.config.kv_cache_dtype,
            )?;

            let mut model_guard = self.tts_model.write().await;
            *model_guard = Some(tts_model);
            drop(model_guard);

            self.set_active_tts_variant(variant, prepared.model_path)
                .await;
            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        self.load_auxiliary_artifacts(variant, &prepared).await
    }

    async fn is_tts_model_already_loaded(&self, model_path: &Path) -> bool {
        let loaded_path = self.loaded_model_path.read().await;
        let tts_model = self.tts_model.read().await;

        tts_model.is_some()
            && loaded_path
                .as_ref()
                .map(|p| p.as_path() == model_path)
                .unwrap_or(false)
    }

    async fn load_auxiliary_artifacts(
        &self,
        variant: ModelVariant,
        prepared: &PreparedLoad,
    ) -> Result<()> {
        match Tokenizer::from_path(&prepared.model_path) {
            Ok(tokenizer) => {
                let mut guard = self.tokenizer.write().await;
                *guard = Some(tokenizer);
            }
            Err(err) => {
                tracing::warn!("Failed to load tokenizer from model directory: {}", err);
            }
        }

        if variant.is_tokenizer() {
            let mut codec_guard = self.codec.write().await;
            codec_guard.load_weights(&prepared.model_path)?;
        }

        Ok(())
    }
}
