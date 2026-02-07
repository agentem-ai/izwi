//! Model lifecycle routing.

use tracing::info;

use crate::backends::ExecutionBackend;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::qwen3_tts::Qwen3TtsModel;
use crate::runtime::service::InferenceEngine;
use crate::tokenizer::Tokenizer;

impl InferenceEngine {
    /// Unload a model from memory.
    pub async fn unload_model(&self, variant: ModelVariant) -> Result<()> {
        if variant.is_asr() || variant.is_forced_aligner() {
            self.model_registry.unload_asr(variant).await;
        } else if variant.is_chat() {
            self.model_registry.unload_chat(variant).await;
        } else if variant.is_voxtral() {
            self.model_registry.unload_voxtral(variant).await;
        } else if variant.is_tts() {
            let mut model_guard = self.tts_model.write().await;
            *model_guard = None;
            let mut path_guard = self.loaded_model_path.write().await;
            *path_guard = None;
        }

        self.model_manager.unload_model(variant).await
    }

    /// Load a model for inference.
    pub async fn load_model(&self, variant: ModelVariant) -> Result<()> {
        if !self.model_manager.is_ready(variant).await {
            let info = self.model_manager.get_model_info(variant).await;
            if info.map(|i| i.local_path.is_none()).unwrap_or(true) {
                return Err(Error::ModelNotFound(format!(
                    "Model {} not downloaded. Please download it first.",
                    variant
                )));
            }
        }

        let model_path = self
            .model_manager
            .get_model_info(variant)
            .await
            .and_then(|i| i.local_path)
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;

        let backend_plan = self.backend_router.select(variant);
        info!(
            "Selected backend {:?} for {} ({})",
            backend_plan.backend, variant, backend_plan.reason
        );

        if matches!(backend_plan.backend, ExecutionBackend::MlxNative) {
            return Err(Error::MlxError(format!(
                "MLX runtime backend selected for {} but native MLX execution is not implemented yet",
                variant.dir_name()
            )));
        }

        if variant.is_asr() || variant.is_forced_aligner() {
            self.model_registry.load_asr(variant, &model_path).await?;
            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        if variant.is_chat() {
            self.model_registry.load_chat(variant, &model_path).await?;
            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        if variant.is_voxtral() {
            self.model_registry
                .load_voxtral(variant, &model_path)
                .await?;
            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        if variant.is_tts() {
            info!("Loading native TTS model from {:?}", model_path);
            let tts_model = Qwen3TtsModel::load(&model_path, self.device.clone())?;

            let mut model_guard = self.tts_model.write().await;
            *model_guard = Some(tts_model);
            let mut path_guard = self.loaded_model_path.write().await;
            *path_guard = Some(model_path);

            self.model_manager.mark_loaded(variant).await;
            return Ok(());
        }

        match Tokenizer::from_path(&model_path) {
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
            codec_guard.load_weights(&model_path)?;
        }

        Ok(())
    }
}
