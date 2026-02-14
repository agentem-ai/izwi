//! API routes and handlers

pub mod admin;
pub mod internal;
pub mod openai;
pub mod request_context;
mod router;

pub use router::create_router;
