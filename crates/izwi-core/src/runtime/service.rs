//! Runtime service orchestrator.

use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::{broadcast, RwLock};
use tokio::task::yield_now;

use crate::audio::{AudioCodec, AudioEncoder, StreamingConfig};
use crate::backends::{BackendRouter, ExecutionBackend};
use crate::config::EngineConfig;
use crate::engine::{
    Engine as CoreEngine, EngineCoreConfig, EngineCoreRequest, EngineOutput, StreamingOutput,
    WorkerConfig,
};
use crate::error::{Error, Result};
use crate::model::download::DownloadProgress;
use crate::model::{ModelInfo, ModelManager, ModelVariant};
use crate::models::qwen3_tts::Qwen3TtsModel;
use crate::models::{DeviceProfile, DeviceSelector, ModelRegistry};
use crate::tokenizer::Tokenizer;

/// Main inference engine runtime.
pub struct RuntimeService {
    pub(crate) config: EngineConfig,
    pub(crate) backend_router: BackendRouter,
    pub(crate) model_manager: Arc<ModelManager>,
    pub(crate) model_registry: Arc<ModelRegistry>,
    pub(crate) tokenizer: RwLock<Option<Tokenizer>>,
    pub(crate) codec: RwLock<AudioCodec>,
    #[allow(dead_code)]
    pub(crate) streaming_config: StreamingConfig,
    pub(crate) tts_model: Arc<RwLock<Option<Qwen3TtsModel>>>,
    pub(crate) core_engine: Arc<CoreEngine>,
    pub(crate) loaded_model_path: RwLock<Option<PathBuf>>,
    pub(crate) loaded_tts_variant: RwLock<Option<ModelVariant>>,
    pub(crate) device: DeviceProfile,
}

impl RuntimeService {
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
        let default_backend = if device.kind.is_metal() {
            ExecutionBackend::CandleMetal
        } else {
            ExecutionBackend::CandleNative
        };

        let mut core_config = EngineCoreConfig::for_qwen3_tts();
        core_config.models_dir = config.models_dir.clone();
        core_config.max_batch_size = config.max_batch_size.max(1);
        core_config.max_seq_len = config.max_sequence_length.max(1);
        core_config.use_metal = config.use_metal;
        core_config.num_threads = config.num_threads.max(1);
        core_config.block_size = config.kv_page_size.max(1);

        let mut worker_config = WorkerConfig::from(&core_config);
        worker_config.models_dir = config.models_dir.clone();
        worker_config.kv_page_size = config.kv_page_size.max(1);
        worker_config.shared_tts_model = Some(tts_model.clone());
        worker_config.model_registry = Some(model_registry.clone());
        worker_config.device = if config.use_metal {
            "mps".to_string()
        } else {
            "cpu".to_string()
        };
        let core_engine = Arc::new(CoreEngine::new_with_worker(core_config, worker_config)?);

        Ok(Self {
            config,
            backend_router: BackendRouter::from_env_with_default(default_backend),
            model_manager,
            model_registry,
            tokenizer: RwLock::new(None),
            codec: RwLock::new(AudioCodec::new()),
            streaming_config: StreamingConfig::default(),
            tts_model,
            core_engine,
            loaded_model_path: RwLock::new(None),
            loaded_tts_variant: RwLock::new(None),
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
        if let Some(variant) = loaded_variant.filter(|variant| variant.is_lfm2()) {
            if let Some(model) = self.model_registry.get_lfm2(variant).await {
                return Ok(model.available_voices());
            }
        }

        let tts_model = self.tts_model.read().await;
        let model = tts_model
            .as_ref()
            .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;

        Ok(model.available_speakers().into_iter().cloned().collect())
    }

    pub(crate) async fn run_streaming_request<F, Fut>(
        &self,
        mut request: EngineCoreRequest,
        mut on_chunk: F,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        request.streaming = true;
        let request_id = request.id.clone();
        let (stream_request_id, mut stream_rx) =
            self.core_engine.generate_streaming(request).await?;
        debug_assert_eq!(stream_request_id, request_id);

        let step_engine = self.core_engine.clone();
        let step_request_id = stream_request_id.clone();
        let step_task = tokio::spawn(async move {
            loop {
                let outputs = step_engine.step().await?;
                for output in outputs {
                    if output.request_id == step_request_id && output.is_finished {
                        return Ok(output);
                    }
                }

                if !step_engine.has_request(&step_request_id).await {
                    return Err(Error::InferenceError(format!(
                        "Request {} ended without terminal output",
                        step_request_id
                    )));
                }

                yield_now().await;
            }
        });

        while let Some(chunk) = stream_rx.recv().await {
            if chunk.request_id != stream_request_id {
                continue;
            }

            if let Err(err) = on_chunk(chunk).await {
                let _ = self.core_engine.abort_request(&stream_request_id).await;
                step_task.abort();
                return Err(err);
            }
        }

        step_task
            .await
            .map_err(|err| Error::InferenceError(format!("Streaming worker task failed: {err}")))?
    }
}
