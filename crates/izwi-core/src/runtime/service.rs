//! Runtime service orchestrator.

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::{broadcast, oneshot, Mutex, RwLock};
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
    completion_waiters: Arc<Mutex<HashMap<String, oneshot::Sender<Result<EngineOutput>>>>>,
    step_driver_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    step_driver_started: AtomicBool,
    pub(crate) loaded_model_path: RwLock<Option<PathBuf>>,
    pub(crate) loaded_tts_variant: RwLock<Option<ModelVariant>>,
    pub(crate) device: DeviceProfile,
}

struct PendingRequestGuard {
    request_id: String,
    core_engine: Arc<CoreEngine>,
    completion_waiters: Arc<Mutex<HashMap<String, oneshot::Sender<Result<EngineOutput>>>>>,
    active: bool,
}

impl PendingRequestGuard {
    fn new(
        request_id: String,
        core_engine: Arc<CoreEngine>,
        completion_waiters: Arc<Mutex<HashMap<String, oneshot::Sender<Result<EngineOutput>>>>>,
    ) -> Self {
        Self {
            request_id,
            core_engine,
            completion_waiters,
            active: true,
        }
    }

    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for PendingRequestGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        let request_id = self.request_id.clone();
        let engine = self.core_engine.clone();
        let waiters = self.completion_waiters.clone();

        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let mut guard = waiters.lock().await;
                guard.remove(&request_id);
                drop(guard);

                let _ = engine.abort_request(&request_id).await;
            });
        }
    }
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
            completion_waiters: Arc::new(Mutex::new(HashMap::new())),
            step_driver_task: Mutex::new(None),
            step_driver_started: AtomicBool::new(false),
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

    async fn ensure_step_driver_started(&self) {
        if self.step_driver_started.load(Ordering::Acquire) {
            return;
        }

        let mut guard = self.step_driver_task.lock().await;
        if self.step_driver_started.load(Ordering::Acquire) {
            return;
        }

        let engine = self.core_engine.clone();
        let waiters = self.completion_waiters.clone();
        let task = tokio::spawn(async move {
            loop {
                match engine.step().await {
                    Ok(outputs) => {
                        if outputs.is_empty() {
                            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                            continue;
                        }

                        for output in outputs {
                            if !output.is_finished {
                                continue;
                            }

                            let waiter = {
                                let mut w = waiters.lock().await;
                                w.remove(&output.request_id)
                            };

                            if let Some(tx) = waiter {
                                if let Some(err) = output.error.clone() {
                                    let _ = tx.send(Err(Error::InferenceError(err)));
                                } else {
                                    let _ = tx.send(Ok(output));
                                }
                            }
                        }
                    }
                    Err(err) => {
                        let mut w = waiters.lock().await;
                        let pending: Vec<_> = w.drain().collect();
                        drop(w);
                        for (_, tx) in pending {
                            let _ = tx.send(Err(Error::InferenceError(err.to_string())));
                        }
                        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
                    }
                }
            }
        });

        *guard = Some(task);
        self.step_driver_started.store(true, Ordering::Release);
    }

    async fn register_waiter(&self, request_id: &str) -> oneshot::Receiver<Result<EngineOutput>> {
        let (tx, rx) = oneshot::channel();
        let mut waiters = self.completion_waiters.lock().await;
        waiters.insert(request_id.to_string(), tx);
        rx
    }

    async fn remove_waiter(&self, request_id: &str) {
        let mut waiters = self.completion_waiters.lock().await;
        waiters.remove(request_id);
    }

    async fn await_completion(
        &self,
        request_id: &str,
        rx: oneshot::Receiver<Result<EngineOutput>>,
    ) -> Result<EngineOutput> {
        rx.await.map_err(|_| {
            Error::InferenceError(format!(
                "Request {} completion channel closed unexpectedly",
                request_id
            ))
        })?
    }

    pub(crate) async fn run_request(&self, request: EngineCoreRequest) -> Result<EngineOutput> {
        self.ensure_step_driver_started().await;

        let request_id = request.id.clone();
        let completion_rx = self.register_waiter(&request_id).await;

        if let Err(err) = self.core_engine.add_request(request).await {
            self.remove_waiter(&request_id).await;
            return Err(err);
        }

        let mut guard = PendingRequestGuard::new(
            request_id.clone(),
            self.core_engine.clone(),
            self.completion_waiters.clone(),
        );
        let output = self.await_completion(&request_id, completion_rx).await?;
        guard.disarm();
        Ok(output)
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
        self.ensure_step_driver_started().await;

        request.streaming = true;
        let request_id = request.id.clone();
        let completion_rx = self.register_waiter(&request_id).await;
        let (stream_request_id, mut stream_rx) =
            match self.core_engine.generate_streaming(request).await {
                Ok(v) => v,
                Err(err) => {
                    self.remove_waiter(&request_id).await;
                    return Err(err);
                }
            };
        debug_assert_eq!(stream_request_id, request_id);
        let mut guard = PendingRequestGuard::new(
            stream_request_id.clone(),
            self.core_engine.clone(),
            self.completion_waiters.clone(),
        );

        while let Some(chunk) = stream_rx.recv().await {
            if chunk.request_id != stream_request_id {
                continue;
            }

            if let Err(err) = on_chunk(chunk).await {
                self.remove_waiter(&stream_request_id).await;
                let _ = self.core_engine.abort_request(&stream_request_id).await;
                return Err(err);
            }
        }

        let output = self
            .await_completion(&stream_request_id, completion_rx)
            .await?;
        guard.disarm();
        // Allow pending tasks to progress before returning to upper layers.
        yield_now().await;
        Ok(output)
    }
}
