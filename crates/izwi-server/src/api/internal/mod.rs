//! Internal API namespace.

pub mod health;

use axum::{routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/health", get(health::health_check))
}
