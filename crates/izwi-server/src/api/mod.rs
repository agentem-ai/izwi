//! API routes and handlers

pub mod admin;
pub mod internal;
pub mod openai;
mod router;

pub use router::create_router;
