//! API routes and handlers

pub mod admin;
pub mod agent;
pub mod chat;
pub mod diarization;
pub mod internal;
pub mod openai;
pub mod request_context;
mod router;
pub mod saved_voices;
pub mod speech_history;
pub mod transcription;
pub(crate) mod tts_long_form;

pub use router::create_router;
