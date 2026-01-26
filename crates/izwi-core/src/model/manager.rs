//! Model lifecycle management

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::info;

use crate::config::EngineConfig;
use crate::error::{Error, Result};
use crate::model::download::{DownloadProgress, ModelDownloader};
use crate::model::info::{ModelInfo, ModelStatus, ModelVariant};
use crate::model::weights::ModelWeights;

/// Manages model downloading, loading, and lifecycle
pub struct ModelManager {
    _config: EngineConfig,
    downloader: ModelDownloader,
    models: RwLock<HashMap<ModelVariant, ModelState>>,
}

struct ModelState {
    info: ModelInfo,
    weights: Option<Arc<ModelWeights>>,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new(config: EngineConfig) -> Result<Self> {
        let downloader = ModelDownloader::new(config.models_dir.clone())?;

        // Initialize model states
        let mut models = HashMap::new();
        for variant in ModelVariant::all() {
            let mut info = ModelInfo::new(*variant);

            // Check if already downloaded
            if downloader.is_downloaded(*variant) {
                info.status = ModelStatus::Downloaded;
                info.local_path = Some(downloader.model_path(*variant));
                info.size_bytes = downloader.get_cached_size(*variant);
            }

            models.insert(
                *variant,
                ModelState {
                    info,
                    weights: None,
                },
            );
        }

        Ok(Self {
            _config: config,
            downloader,
            models: RwLock::new(models),
        })
    }

    /// Get list of all available models with their status
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        let models = self.models.read().await;
        models.values().map(|s| s.info.clone()).collect()
    }

    /// Get info for a specific model
    pub async fn get_model_info(&self, variant: ModelVariant) -> Option<ModelInfo> {
        let models = self.models.read().await;
        models.get(&variant).map(|s| s.info.clone())
    }

    /// Download a model from HuggingFace
    pub async fn download_model(&self, variant: ModelVariant) -> Result<PathBuf> {
        // Update status to downloading
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info.status = ModelStatus::Downloading;
                state.info.download_progress = Some(0.0);
            }
        }

        // Perform download
        let result = tokio::task::spawn_blocking({
            let downloader = self.downloader.clone();
            move || downloader.download(variant)
        })
        .await
        .map_err(|e| Error::DownloadError(e.to_string()))??;

        // Update status
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info.status = ModelStatus::Downloaded;
                state.info.local_path = Some(result.clone());
                state.info.download_progress = Some(100.0);
                state.info.size_bytes = self.downloader.get_cached_size(variant);
            }
        }

        Ok(result)
    }

    /// Download a model with progress reporting
    pub async fn download_model_with_progress(
        &self,
        variant: ModelVariant,
        progress_tx: mpsc::Sender<DownloadProgress>,
    ) -> Result<PathBuf> {
        // Update status
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info.status = ModelStatus::Downloading;
            }
        }

        let result = self
            .downloader
            .download_with_progress(variant, progress_tx)
            .await?;

        // Update status
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info.status = ModelStatus::Downloaded;
                state.info.local_path = Some(result.clone());
                state.info.size_bytes = self.downloader.get_cached_size(variant);
            }
        }

        Ok(result)
    }

    /// Load a model into memory
    pub async fn load_model(&self, variant: ModelVariant) -> Result<Arc<ModelWeights>> {
        // Check if already loaded
        {
            let models = self.models.read().await;
            if let Some(state) = models.get(&variant) {
                if let Some(ref weights) = state.weights {
                    return Ok(weights.clone());
                }
            }
        }

        // Get model path
        let model_path = {
            let models = self.models.read().await;
            models
                .get(&variant)
                .and_then(|s| s.info.local_path.clone())
                .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?
        };

        // Update status
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info.status = ModelStatus::Loading;
            }
        }

        info!("Loading model {} from {:?}", variant, model_path);

        // Load weights (blocking operation)
        let weights = tokio::task::spawn_blocking(move || ModelWeights::load(&model_path))
            .await
            .map_err(|e| Error::ModelLoadError(e.to_string()))??;

        let weights = Arc::new(weights);

        // Store loaded weights
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info.status = ModelStatus::Ready;
                state.weights = Some(weights.clone());
            }
        }

        info!("Model {} loaded successfully", variant);
        Ok(weights)
    }

    /// Unload a model from memory
    pub async fn unload_model(&self, variant: ModelVariant) -> Result<()> {
        let mut models = self.models.write().await;
        if let Some(state) = models.get_mut(&variant) {
            state.weights = None;
            state.info.status = if state.info.local_path.is_some() {
                ModelStatus::Downloaded
            } else {
                ModelStatus::NotDownloaded
            };
        }
        Ok(())
    }

    /// Get loaded model weights
    pub async fn get_weights(&self, variant: ModelVariant) -> Option<Arc<ModelWeights>> {
        let models = self.models.read().await;
        models.get(&variant).and_then(|s| s.weights.clone())
    }

    /// Check if model is ready for inference
    pub async fn is_ready(&self, variant: ModelVariant) -> bool {
        let models = self.models.read().await;
        models
            .get(&variant)
            .map(|s| s.info.status == ModelStatus::Ready)
            .unwrap_or(false)
    }

    /// Delete downloaded model files
    pub async fn delete_model(&self, variant: ModelVariant) -> Result<()> {
        // Unload first
        self.unload_model(variant).await?;

        let model_path = self.downloader.model_path(variant);
        if model_path.exists() {
            std::fs::remove_dir_all(&model_path)?;
        }

        // Update status
        {
            let mut models = self.models.write().await;
            if let Some(state) = models.get_mut(&variant) {
                state.info = ModelInfo::new(variant);
            }
        }

        Ok(())
    }
}

// Make downloader cloneable for async tasks
impl Clone for ModelDownloader {
    fn clone(&self) -> Self {
        ModelDownloader::new(self.models_dir.clone()).unwrap()
    }
}
