//! Model registry to ensure models are loaded once and shared across the app.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{OnceCell, RwLock};
use tracing::info;

use crate::error::{Error, Result};
use crate::model::ModelVariant;

use super::device::DeviceProfile;
use super::qwen3_asr::Qwen3AsrModel;

#[derive(Clone)]
pub struct ModelRegistry {
    models_dir: PathBuf,
    device: DeviceProfile,
    asr_models: Arc<RwLock<HashMap<ModelVariant, Arc<OnceCell<Arc<Qwen3AsrModel>>>>>>,
}

impl ModelRegistry {
    pub fn new(models_dir: PathBuf, device: DeviceProfile) -> Self {
        Self {
            models_dir,
            device,
            asr_models: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn device(&self) -> &DeviceProfile {
        &self.device
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub async fn load_asr(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
    ) -> Result<Arc<Qwen3AsrModel>> {
        if !variant.is_asr() && !variant.is_forced_aligner() && !variant.is_voxtral() {
            return Err(Error::InvalidInput(format!(
                "Model variant {variant} is not an ASR, ForcedAligner, or Voxtral model"
            )));
        }

        // Voxtral models require vLLM for inference and are not loaded into native Candle runtime
        // They are managed externally; just return a placeholder for now
        if variant.is_voxtral() {
            return Err(Error::InvalidInput(
                "Voxtral models require vLLM for inference and are not supported by the native runtime".to_string()
            ));
        }

        let cell = {
            let mut guard = self.asr_models.write().await;
            guard
                .entry(variant)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        info!("Loading native ASR/ForcedAligner model {variant} from {model_dir:?}");

        let model = cell
            .get_or_try_init({
                let model_dir = model_dir.to_path_buf();
                let device = self.device.clone();
                move || async move {
                    tokio::task::spawn_blocking(move || Qwen3AsrModel::load(&model_dir, device))
                        .await
                        .map_err(|e| Error::ModelLoadError(e.to_string()))?
                        .map(Arc::new)
                }
            })
            .await?;

        Ok(model.clone())
    }

    pub async fn get_asr(&self, variant: ModelVariant) -> Option<Arc<Qwen3AsrModel>> {
        let guard = self.asr_models.read().await;
        guard.get(&variant).and_then(|cell| cell.get().cloned())
    }

    pub async fn unload_asr(&self, variant: ModelVariant) {
        let mut guard = self.asr_models.write().await;
        guard.remove(&variant);
    }
}
