//! Runtime service orchestrator.

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::{broadcast, RwLock};

use crate::audio::{AudioCodec, AudioEncoder, StreamingConfig};
use crate::backends::{BackendRouter, ExecutionBackend};
use crate::config::EngineConfig;
use crate::error::{Error, Result};
use crate::model::download::DownloadProgress;
use crate::model::{ModelInfo, ModelManager, ModelVariant};
use crate::models::qwen3_tts::Qwen3TtsModel;
use crate::models::{DeviceProfile, DeviceSelector, ModelRegistry};
use crate::runtime::tts_batcher::TtsBatcher;
use crate::tokenizer::Tokenizer;

/// Main inference engine runtime.
pub struct InferenceEngine {
    pub(crate) config: EngineConfig,
    pub(crate) backend_router: BackendRouter,
    pub(crate) model_manager: Arc<ModelManager>,
    pub(crate) model_registry: Arc<ModelRegistry>,
    pub(crate) tokenizer: RwLock<Option<Tokenizer>>,
    pub(crate) codec: RwLock<AudioCodec>,
    #[allow(dead_code)]
    pub(crate) streaming_config: StreamingConfig,
    pub(crate) tts_model: Arc<RwLock<Option<Qwen3TtsModel>>>,
    pub(crate) tts_batcher: Option<TtsBatcher>,
    pub(crate) loaded_model_path: RwLock<Option<PathBuf>>,
    pub(crate) loaded_tts_variant: RwLock<Option<ModelVariant>>,
    pub(crate) lfm2_fallback_tts_model: Arc<RwLock<Option<Qwen3TtsModel>>>,
    pub(crate) lfm2_fallback_tts_variant: RwLock<Option<ModelVariant>>,
    pub(crate) device: DeviceProfile,
}

impl InferenceEngine {
    /// Create a new inference engine.
    pub fn new(config: EngineConfig) -> Result<Self> {
        let model_manager = Arc::new(ModelManager::new(config.clone())?);

        let device = if cfg!(target_os = "macos") {
            let preference = if config.use_metal {
                Some("metal")
            } else {
                Some("cpu")
            };
            DeviceSelector::detect_with_preference(preference)?
        } else {
            DeviceSelector::detect()?
        };

        let model_registry = Arc::new(ModelRegistry::new(
            config.models_dir.clone(),
            device.clone(),
        ));

        let tts_model = Arc::new(RwLock::new(None));
        let lfm2_fallback_tts_model = Arc::new(RwLock::new(None));
        let tts_batcher = if config.max_batch_size > 1 {
            Some(TtsBatcher::new(config.clone(), tts_model.clone()))
        } else {
            None
        };

        let default_backend = if device.kind.is_metal() {
            ExecutionBackend::CandleMetal
        } else {
            ExecutionBackend::CandleNative
        };

        Ok(Self {
            config,
            backend_router: BackendRouter::from_env_with_default(default_backend),
            model_manager,
            model_registry,
            tokenizer: RwLock::new(None),
            codec: RwLock::new(AudioCodec::new()),
            streaming_config: StreamingConfig::default(),
            tts_model,
            tts_batcher,
            loaded_model_path: RwLock::new(None),
            loaded_tts_variant: RwLock::new(None),
            lfm2_fallback_tts_model,
            lfm2_fallback_tts_variant: RwLock::new(None),
            device,
        })
    }

    /// Get reference to model manager.
    pub fn model_manager(&self) -> &Arc<ModelManager> {
        &self.model_manager
    }

    /// List available models.
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        self.model_manager.list_models().await
    }

    /// Download a model.
    pub async fn download_model(&self, variant: ModelVariant) -> Result<()> {
        self.model_manager.download_model(variant).await?;
        Ok(())
    }

    /// Spawn a non-blocking background download.
    pub async fn spawn_download(
        &self,
        variant: ModelVariant,
    ) -> Result<broadcast::Receiver<DownloadProgress>> {
        self.model_manager.spawn_download(variant).await
    }

    /// Check if a download is active.
    pub async fn is_download_active(&self, variant: ModelVariant) -> bool {
        self.model_manager.is_download_active(variant).await
    }

    /// Get runtime configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Get codec sample rate.
    pub async fn sample_rate(&self) -> u32 {
        self.codec.read().await.sample_rate()
    }

    /// Create audio encoder.
    pub async fn audio_encoder(&self) -> AudioEncoder {
        let codec = self.codec.read().await;
        AudioEncoder::new(codec.sample_rate(), 1)
    }

    /// Get available speakers for loaded TTS model.
    pub async fn available_speakers(&self) -> Result<Vec<String>> {
        let loaded_variant = *self.loaded_tts_variant.read().await;
        if matches!(loaded_variant, Some(ModelVariant::Lfm2Audio15B)) {
            if let Some(model) = self
                .model_registry
                .get_lfm2(ModelVariant::Lfm2Audio15B)
                .await
            {
                return Ok(model.available_voices());
            }
        }

        let tts_model = self.tts_model.read().await;
        let model = tts_model
            .as_ref()
            .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;

        Ok(model.available_speakers().into_iter().cloned().collect())
    }
}
