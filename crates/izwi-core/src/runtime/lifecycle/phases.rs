use std::path::PathBuf;

use tracing::info;

use crate::backends::ExecutionBackend;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::runtime::service::RuntimeService;

pub(super) struct PreparedLoad {
    pub model_path: PathBuf,
}

impl RuntimeService {
    pub(super) async fn prepare_model_load(&self, variant: ModelVariant) -> Result<PreparedLoad> {
        self.ensure_model_is_downloaded(variant).await?;
        let model_path = self.resolve_model_path(variant).await?;

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

        Ok(PreparedLoad { model_path })
    }

    pub(super) async fn clear_active_tts_variant(&self) {
        let mut path_guard = self.loaded_model_path.write().await;
        *path_guard = None;

        let mut variant_guard = self.loaded_tts_variant.write().await;
        *variant_guard = None;
    }

    pub(super) async fn set_active_tts_variant(&self, variant: ModelVariant, model_path: PathBuf) {
        let mut path_guard = self.loaded_model_path.write().await;
        *path_guard = Some(model_path);

        let mut variant_guard = self.loaded_tts_variant.write().await;
        *variant_guard = Some(variant);
    }

    async fn ensure_model_is_downloaded(&self, variant: ModelVariant) -> Result<()> {
        if self.model_manager.is_ready(variant).await {
            return Ok(());
        }

        let info = self.model_manager.get_model_info(variant).await;
        if info.map(|i| i.local_path.is_none()).unwrap_or(true) {
            return Err(Error::ModelNotFound(format!(
                "Model {} not downloaded. Please download it first.",
                variant
            )));
        }

        Ok(())
    }

    async fn resolve_model_path(&self, variant: ModelVariant) -> Result<PathBuf> {
        self.model_manager
            .get_model_info(variant)
            .await
            .and_then(|i| i.local_path)
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))
    }
}
