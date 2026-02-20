//! API routes and handlers

pub mod admin;
pub mod chat;
pub mod internal;
pub mod openai;
pub mod request_context;
pub mod transcription;
mod router;

pub use router::create_router;
