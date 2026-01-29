//! Model executor - handles forward pass execution.
//!
//! The executor abstracts the actual model inference, allowing for different
//! backends (Python bridge, native Rust, etc.) while providing a unified interface.

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use super::config::EngineCoreConfig;
use super::request::EngineCoreRequest;
use super::scheduler::ScheduledRequest;
use super::types::{AudioOutput, ModelType, TaskType};
use crate::error::{Error, Result};
use crate::inference::lfm2_bridge::{LFM2Bridge, LFM2Response};
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
            model_type: ModelType::LFM2Audio,
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
    lfm2_bridge: LFM2Bridge,
    tts_bridge: PythonBridge,
    initialized: bool,
}

impl PythonExecutor {
    /// Create a new Python executor.
    pub fn new(config: WorkerConfig) -> Self {
        Self {
            config,
            lfm2_bridge: LFM2Bridge::new(),
            tts_bridge: PythonBridge::new(),
            initialized: false,
        }
    }

    /// Execute a single TTS request via LFM2.
    fn execute_lfm2_tts(&self, request: &EngineCoreRequest) -> Result<ExecutorOutput> {
        let text = request
            .text
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("TTS requires text".into()))?;

        let voice = request.params.voice.as_deref();
        let max_tokens = Some(request.params.max_tokens as u32);
        let audio_temp = request.params.audio_temperature;
        let audio_top_k = request.params.audio_top_k.map(|k| k as u32);

        let response =
            self.lfm2_bridge
                .generate_tts(text, voice, max_tokens, audio_temp, audio_top_k)?;

        self.lfm2_response_to_output(&request.id, response)
    }

    /// Execute ASR request via LFM2.
    fn execute_lfm2_asr(&self, request: &EngineCoreRequest) -> Result<ExecutorOutput> {
        let audio = request
            .audio_input
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("ASR requires audio input".into()))?;

        let max_tokens = Some(request.params.max_tokens as u32);

        let response = self.lfm2_bridge.transcribe(audio, max_tokens)?;

        self.lfm2_response_to_output(&request.id, response)
    }

    /// Execute audio chat request via LFM2.
    fn execute_lfm2_chat(&self, request: &EngineCoreRequest) -> Result<ExecutorOutput> {
        let audio = request.audio_input.as_deref();
        let text = request.text.as_deref();
        let max_tokens = Some(request.params.max_tokens as u32);
        let audio_temp = request.params.audio_temperature;
        let audio_top_k = request.params.audio_top_k.map(|k| k as u32);

        let response =
            self.lfm2_bridge
                .audio_chat(audio, text, max_tokens, audio_temp, audio_top_k)?;

        self.lfm2_response_to_output(&request.id, response)
    }

    /// Convert LFM2 response to executor output.
    fn lfm2_response_to_output(
        &self,
        request_id: &str,
        response: LFM2Response,
    ) -> Result<ExecutorOutput> {
        if let Some(error) = response.error {
            return Ok(ExecutorOutput::error(request_id.to_string(), error));
        }

        let audio = if let Some(audio_b64) = response.audio_base64 {
            let sample_rate = response.sample_rate.unwrap_or(24000);
            let samples = decode_audio_base64(&audio_b64, sample_rate)?;
            Some(AudioOutput::new(samples, sample_rate))
        } else {
            None
        };

        let text = response.text.or(response.transcription);

        Ok(ExecutorOutput {
            request_id: request_id.to_string(),
            audio,
            text,
            tokens_processed: 0, // Not tracked by daemon
            tokens_generated: 0, // Not tracked by daemon
            finished: true,
            error: None,
        })
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

        let mut outputs = Vec::with_capacity(requests.len());

        // Execute each request (no batching in Python daemon currently)
        for request in requests {
            let result = match (&request.model_type, &request.task_type) {
                (ModelType::LFM2Audio, TaskType::TTS) => self.execute_lfm2_tts(request),
                (ModelType::LFM2Audio, TaskType::ASR) => self.execute_lfm2_asr(request),
                (ModelType::LFM2Audio, TaskType::AudioChat) => self.execute_lfm2_chat(request),
                (ModelType::Qwen3TTS, TaskType::TTS) => self.execute_qwen_tts(request),
                _ => Err(Error::InferenceError(format!(
                    "Unsupported model/task combination: {:?}/{:?}",
                    request.model_type, request.task_type
                ))),
            };

            match result {
                Ok(output) => outputs.push(output),
                Err(e) => {
                    warn!("Execution error for request {}: {}", request.id, e);
                    outputs.push(ExecutorOutput::error(request.id.clone(), e.to_string()));
                }
            }
        }

        Ok(outputs)
    }

    fn is_ready(&self) -> bool {
        self.initialized
    }

    fn initialize(&mut self) -> Result<()> {
        info!("Initializing Python executor");

        // Start the appropriate daemon based on model type
        match self.config.model_type {
            ModelType::LFM2Audio => {
                self.lfm2_bridge.ensure_daemon_running()?;
                info!("LFM2 daemon ready");
            }
            ModelType::Qwen3TTS => {
                self.tts_bridge.ensure_daemon_running()?;
                info!("TTS daemon ready");
            }
        }

        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down Python executor");

        if let Err(e) = self.lfm2_bridge.stop_daemon() {
            warn!("Error stopping LFM2 daemon: {}", e);
        }

        if let Err(e) = self.tts_bridge.stop_daemon() {
            warn!("Error stopping TTS daemon: {}", e);
        }

        self.initialized = false;
        Ok(())
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
        assert_eq!(config.model_type, ModelType::LFM2Audio);
    }
}
