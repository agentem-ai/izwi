//! Model executor - handles forward pass execution.
//!
//! The executor abstracts the actual model inference, allowing for different
//! backends (Python bridge, native Rust, etc.) while providing a unified interface.

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::config::EngineCoreConfig;
use super::request::EngineCoreRequest;
use super::scheduler::ScheduledRequest;
use super::types::{AudioOutput, ModelType, TaskType};
use crate::error::{Error, Result};
use crate::inference::python_bridge::PythonBridge;

/// Configuration for the model executor.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Model type
    pub model_type: ModelType,
    /// Path to models directory
    pub models_dir: PathBuf,
    /// Device to use (cpu, mps, cuda)
    pub device: String,
    /// Data type (float32, float16, bfloat16)
    pub dtype: String,
    /// Number of threads
    pub num_threads: usize,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::Qwen3TTS,
            models_dir: dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("izwi")
                .join("models"),
            device: if cfg!(target_os = "macos") {
                "mps".to_string()
            } else {
                "cpu".to_string()
            },
            dtype: "float32".to_string(),
            num_threads: 4,
        }
    }
}

impl From<&EngineCoreConfig> for WorkerConfig {
    fn from(config: &EngineCoreConfig) -> Self {
        Self {
            model_type: config.model_type,
            models_dir: config.models_dir.clone(),
            device: if config.use_metal {
                "mps".to_string()
            } else {
                "cpu".to_string()
            },
            dtype: "float32".to_string(),
            num_threads: config.num_threads,
        }
    }
}

/// Output from the executor after a forward pass.
#[derive(Debug, Clone)]
pub struct ExecutorOutput {
    /// Request ID
    pub request_id: String,
    /// Generated audio samples
    pub audio: Option<AudioOutput>,
    /// Generated text (for ASR/chat)
    pub text: Option<String>,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Whether generation is complete
    pub finished: bool,
    /// Error if any
    pub error: Option<String>,
}

impl ExecutorOutput {
    pub fn error(request_id: String, error: impl Into<String>) -> Self {
        Self {
            request_id,
            audio: None,
            text: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: true,
            error: Some(error.into()),
        }
    }
}

/// Model executor trait - abstracts the model inference backend.
pub trait ModelExecutor: Send + Sync {
    /// Execute forward pass for scheduled requests.
    fn execute(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>>;

    /// Check if the executor is ready.
    fn is_ready(&self) -> bool;

    /// Initialize the executor (load models, etc.)
    fn initialize(&mut self) -> Result<()>;

    /// Shutdown the executor.
    fn shutdown(&mut self) -> Result<()>;
}

/// Python-based model executor using daemon processes.
pub struct PythonExecutor {
    config: WorkerConfig,
    tts_bridge: Arc<PythonBridge>,
    initialized: bool,
    /// Maximum concurrent requests to execute
    max_concurrent: usize,
}

impl PythonExecutor {
    /// Create a new Python executor.
    pub fn new(config: WorkerConfig) -> Self {
        Self {
            config,
            tts_bridge: Arc::new(PythonBridge::new()),
            initialized: false,
            max_concurrent: 4, // Limit concurrent requests to avoid overwhelming the daemon
        }
    }

    /// Create executor with custom concurrency limit.
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max.max(1);
        self
    }

    /// Execute a single TTS request via Qwen3-TTS.
    fn execute_qwen_tts(&self, request: &EngineCoreRequest) -> Result<ExecutorOutput> {
        let text = request
            .text
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("TTS requires text".into()))?;

        // Get model path - for now use a default
        let model_path = self
            .config
            .models_dir
            .join("Qwen3-TTS-12Hz-0.6B-CustomVoice");

        let speaker = request.params.speaker.as_deref();
        let voice_desc = request.voice_description.as_deref();
        let ref_audio = request.reference_audio.clone();
        let ref_text = request.reference_text.clone();

        let (samples, sample_rate) = self.tts_bridge.generate_with_clone(
            &model_path,
            text,
            speaker,
            Some("Auto"),
            voice_desc,
            ref_audio,
            ref_text,
        )?;

        Ok(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput::new(samples, sample_rate)),
            text: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: true,
            error: None,
        })
    }
}

impl ModelExecutor for PythonExecutor {
    fn execute(
        &self,
        requests: &[&EngineCoreRequest],
        _scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        if !self.initialized {
            return Err(Error::InferenceError("Executor not initialized".into()));
        }

        if requests.is_empty() {
            return Ok(Vec::new());
        }

        // For a single request, execute directly without async overhead
        if requests.len() == 1 {
            let request = requests[0];
            let result = match (&request.model_type, &request.task_type) {
                (ModelType::Qwen3TTS, TaskType::TTS) => self.execute_qwen_tts(request),
                _ => Err(Error::InferenceError(format!(
                    "Unsupported model/task combination: {:?}/{:?}",
                    request.model_type, request.task_type
                ))),
            };

            return match result {
                Ok(output) => Ok(vec![output]),
                Err(e) => {
                    warn!("Execution error for request {}: {}", request.id, e);
                    Ok(vec![ExecutorOutput::error(
                        request.id.clone(),
                        e.to_string(),
                    )])
                }
            };
        }

        // For multiple requests, execute concurrently in batches
        debug!(
            "Executing {} requests concurrently (max_concurrent: {})",
            requests.len(),
            self.max_concurrent
        );

        // Clone data needed for execution
        let request_data: Vec<_> = requests
            .iter()
            .map(|r| ExecutionTask {
                id: r.id.clone(),
                model_type: r.model_type,
                task_type: r.task_type,
                text: r.text.clone(),
                speaker: r.params.speaker.clone(),
                voice_description: r.voice_description.clone(),
                reference_audio: r.reference_audio.clone(),
                reference_text: r.reference_text.clone(),
            })
            .collect();

        let bridge = self.tts_bridge.clone();
        let models_dir = self.config.models_dir.clone();

        // Execute all tasks concurrently using rayon for CPU-bound work
        let outputs: Vec<ExecutorOutput> = request_data
            .into_iter()
            .map(|task| execute_single_task(&bridge, &models_dir, task))
            .collect();

        Ok(outputs)
    }

    fn is_ready(&self) -> bool {
        self.initialized
    }

    fn initialize(&mut self) -> Result<()> {
        info!(
            "Initializing Python executor (max_concurrent: {})",
            self.max_concurrent
        );

        // Start the TTS daemon
        self.tts_bridge.ensure_daemon_running()?;
        info!("TTS daemon ready");

        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down Python executor");

        if let Err(e) = self.tts_bridge.stop_daemon() {
            warn!("Error stopping TTS daemon: {}", e);
        }

        self.initialized = false;
        Ok(())
    }
}

/// Data needed for executing a single task.
#[derive(Clone)]
struct ExecutionTask {
    id: String,
    model_type: ModelType,
    task_type: TaskType,
    text: Option<String>,
    speaker: Option<String>,
    voice_description: Option<String>,
    reference_audio: Option<String>,
    reference_text: Option<String>,
}

/// Execute a single task using the Python bridge.
fn execute_single_task(
    bridge: &PythonBridge,
    models_dir: &PathBuf,
    task: ExecutionTask,
) -> ExecutorOutput {
    match (task.model_type, task.task_type) {
        (ModelType::Qwen3TTS, TaskType::TTS) => {
            let text = match &task.text {
                Some(t) => t,
                None => {
                    return ExecutorOutput::error(task.id, "TTS requires text");
                }
            };

            let model_path = models_dir.join("Qwen3-TTS-12Hz-0.6B-CustomVoice");

            match bridge.generate_with_clone(
                &model_path,
                text,
                task.speaker.as_deref(),
                Some("Auto"),
                task.voice_description.as_deref(),
                task.reference_audio.clone(),
                task.reference_text.clone(),
            ) {
                Ok((samples, sample_rate)) => ExecutorOutput {
                    request_id: task.id,
                    audio: Some(AudioOutput::new(samples, sample_rate)),
                    text: None,
                    tokens_processed: 0,
                    tokens_generated: 0,
                    finished: true,
                    error: None,
                },
                Err(e) => {
                    warn!("TTS execution error for {}: {}", task.id, e);
                    ExecutorOutput::error(task.id, e.to_string())
                }
            }
        }
        _ => ExecutorOutput::error(
            task.id,
            format!(
                "Unsupported model/task combination: {:?}/{:?}",
                task.model_type, task.task_type
            ),
        ),
    }
}

/// Unified executor that wraps a model executor implementation.
pub struct UnifiedExecutor {
    inner: Arc<RwLock<Box<dyn ModelExecutor>>>,
}

impl UnifiedExecutor {
    /// Create a new unified executor with Python backend.
    pub fn new_python(config: WorkerConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Box::new(PythonExecutor::new(config)))),
        }
    }

    /// Execute requests.
    pub async fn execute(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        let executor = self.inner.read().await;
        executor.execute(requests, scheduled)
    }

    /// Check if ready.
    pub async fn is_ready(&self) -> bool {
        let executor = self.inner.read().await;
        executor.is_ready()
    }

    /// Initialize.
    pub async fn initialize(&self) -> Result<()> {
        let mut executor = self.inner.write().await;
        executor.initialize()
    }

    /// Shutdown.
    pub async fn shutdown(&self) -> Result<()> {
        let mut executor = self.inner.write().await;
        executor.shutdown()
    }
}

/// Decode base64-encoded audio to samples.
fn decode_audio_base64(audio_b64: &str, _sample_rate: u32) -> Result<Vec<f32>> {
    use base64::Engine;
    use std::io::Cursor;

    let wav_bytes = base64::engine::general_purpose::STANDARD
        .decode(audio_b64)
        .map_err(|e| Error::InferenceError(format!("Failed to decode base64 audio: {}", e)))?;

    let cursor = Cursor::new(wav_bytes);
    let mut reader = hound::WavReader::new(cursor)
        .map_err(|e| Error::InferenceError(format!("Failed to parse WAV: {}", e)))?;

    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_config_default() {
        let config = WorkerConfig::default();
        assert_eq!(config.model_type, ModelType::Qwen3TTS);
    }
}
