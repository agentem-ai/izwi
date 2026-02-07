//! Model executor - handles forward pass execution.
//!
//! This module provides executor abstractions for model inference.
//! Since native Rust models are now used, actual inference is handled
//! directly by the InferenceEngine. This module provides compatibility
//! types and helpers.

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

use super::config::EngineCoreConfig;
use super::request::EngineCoreRequest;
use super::scheduler::ScheduledRequest;
use super::types::{AudioOutput, ModelType};
use crate::error::{Error, Result};

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
/// Note: Actual TTS/ASR inference is now handled directly by InferenceEngine.
/// This trait is kept for compatibility and potential future use.
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

/// Stub executor that delegates to InferenceEngine.
/// Since native Rust models are used, the actual inference happens
/// in InferenceEngine. This executor serves as a compatibility layer.
pub struct NativeExecutor {
    config: WorkerConfig,
    initialized: bool,
}

impl NativeExecutor {
    /// Create a new native executor.
    pub fn new(config: WorkerConfig) -> Self {
        Self {
            config,
            initialized: false,
        }
    }
}

impl ModelExecutor for NativeExecutor {
    fn execute(
        &self,
        _requests: &[&EngineCoreRequest],
        _scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        if !self.initialized {
            return Err(Error::InferenceError("Executor not initialized".into()));
        }

        // Native TTS execution is handled directly by InferenceEngine
        // This stub returns an error directing users to use InferenceEngine
        Err(Error::InferenceError(
            "Use InferenceEngine for native TTS/ASR execution".into(),
        ))
    }

    fn is_ready(&self) -> bool {
        self.initialized
    }

    fn initialize(&mut self) -> Result<()> {
        info!("Initializing native executor");
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down native executor");
        self.initialized = false;
        Ok(())
    }
}

/// Unified executor that wraps a model executor implementation.
pub struct UnifiedExecutor {
    inner: Arc<RwLock<Box<dyn ModelExecutor>>>,
}

impl UnifiedExecutor {
    /// Create a new unified executor with native backend.
    pub fn new_native(config: WorkerConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Box::new(NativeExecutor::new(config)))),
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
pub fn decode_audio_base64(audio_b64: &str, _sample_rate: u32) -> Result<Vec<f32>> {
    use base64::Engine;
    use std::io::Cursor;

    let payload = if audio_b64.starts_with("data:") {
        audio_b64
            .split_once(',')
            .map(|(_, b64)| b64)
            .unwrap_or(audio_b64)
    } else {
        audio_b64
    };
    let normalized: String = payload.chars().filter(|c| !c.is_whitespace()).collect();

    let wav_bytes = base64::engine::general_purpose::STANDARD
        .decode(normalized.as_bytes())
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
