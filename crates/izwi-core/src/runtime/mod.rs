//! Runtime orchestration layer.
//!
//! This is the canonical request lifecycle module (similar to runtime engines
//! in vLLM/TGI/llama.cpp style systems), while legacy `inference` paths are
//! maintained as compatibility shims.

mod asr;
mod audio_io;
mod chat;
mod diarization;
mod lfm2;
mod model_router;
mod service;
mod tts;
mod types;

pub use service::{RuntimeService, RuntimeTelemetrySnapshot};
pub use types::{
    AsrTranscription, AudioChunk, ChatGeneration, ChunkStats, DiarizationConfig, DiarizationResult,
    DiarizationSegment, DiarizationTranscriptResult, DiarizationUtterance, DiarizationWord,
    GenerationConfig, GenerationRequest, GenerationResult, SpeechToSpeechGeneration,
};
