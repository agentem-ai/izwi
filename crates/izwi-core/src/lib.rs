//! Izwi Core - High-Performance Audio Inference Engine
//!
//! This crate provides a production-ready inference engine for audio models
//! (Qwen3-TTS, LFM2-Audio) on Apple Silicon and CPU devices.
//!
//! # Architecture
//!
//! The engine follows vLLM's architecture patterns with:
//! - Request scheduling with FCFS/priority policies
//! - Paged KV-cache memory management
//! - Continuous batching support
//! - Streaming output
//!
//! # Example
//!
//! ```ignore
//! use izwi_core::engine::{Engine, EngineCoreConfig, EngineCoreRequest};
//!
//! let config = EngineCoreConfig::for_lfm2();
//! let engine = Engine::new(config)?;
//!
//! let request = EngineCoreRequest::tts("Hello, world!");
//! let output = engine.generate(request).await?;
//! ```

pub mod audio;
pub mod backends;
pub mod catalog;
pub mod codecs;
pub mod config;
pub mod engine;
pub mod error;
pub mod families;
pub mod model;
pub mod models;
pub mod runtime;
pub mod tokenizer;

// Re-export main types from the new engine module
pub use engine::{
    Engine, EngineCore, EngineCoreConfig, EngineCoreRequest, EngineMetrics, EngineOutput,
    GenerationParams, KVCacheManager, ModelExecutor, OutputProcessor, RequestProcessor,
    RequestStatus, Scheduler, SchedulerConfig, SchedulingPolicy, StreamingOutput,
};

// Legacy re-exports for backward compatibility
pub use config::EngineConfig;
pub use error::{Error, Result};
pub use runtime::{
    AsrTranscription, ChatGeneration, ChunkStats, GenerationRequest, GenerationResult,
};
pub use runtime::{AudioChunk, GenerationConfig, InferenceEngine};

// Catalog/model metadata re-exports
pub use catalog::{
    parse_chat_model_variant, parse_model_variant, parse_tts_model_variant,
    resolve_asr_model_variant, DownloadProgress, ModelInfo, ModelManager, ModelStatus,
    ModelVariant,
};

// Native family/device registry re-exports
pub use families::{DeviceProfile, DeviceSelector, ModelRegistry};
