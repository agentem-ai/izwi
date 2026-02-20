//! API routes and handlers

pub mod admin;
pub mod chat;
pub mod diarization;
pub mod internal;
pub mod openai;
pub mod request_context;
mod router;
pub mod speech_history;
pub mod transcription;

pub use router::create_router;
