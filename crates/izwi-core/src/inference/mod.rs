//! Legacy inference module.
//!
//! Kept for backward compatibility while canonical runtime code now lives in
//! `crate::runtime`.

mod engine;
mod generation;
mod kv_cache;

pub use engine::{AsrTranscription, ChatGeneration, InferenceEngine};
pub use generation::{
    AudioChunk, ChunkStats, GenerationConfig, GenerationRequest, GenerationResult,
};
pub use kv_cache::KVCache;
