//! API routes and handlers

mod asr;
mod daemon;
mod health;
mod models;
mod tts;

use axum::{
    routing::{get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::state::AppState;

/// Create the main API router
pub fn create_router(state: AppState) -> Router {
    let api_routes = Router::new()
        // Health check
        .route("/health", get(health::health_check))
        // Daemon management
        .route("/daemon/status", get(daemon::get_status))
        .route("/daemon/start", post(daemon::start_daemon))
        .route("/daemon/stop", post(daemon::stop_daemon))
        .route("/daemon/preload", post(daemon::preload_model))
        // Model management
        .route("/models", get(models::list_models))
        .route("/models/:variant/download", post(models::download_model))
        .route(
            "/models/:variant/download/progress",
            get(models::download_progress_stream),
        )
        .route(
            "/models/:variant/download/cancel",
            post(models::cancel_download),
        )
        .route("/models/:variant/load", post(models::load_model))
        .route("/models/:variant/unload", post(models::unload_model))
        .route(
            "/models/:variant",
            get(models::get_model_info).delete(models::delete_model),
        )
        // TTS generation (Qwen3-TTS)
        .route("/tts/generate", post(tts::generate))
        .route("/tts/stream", post(tts::generate_stream))
        // Qwen3-ASR endpoints
        .route("/asr/status", get(asr::status))
        .route("/asr/start", post(asr::start_daemon))
        .route("/asr/stop", post(asr::stop_daemon))
        .route("/asr/transcribe", post(asr::transcribe))
        .route("/asr/transcribe/stream", post(asr::transcribe_stream));

    Router::new()
        .nest("/api/v1", api_routes)
        // Serve static files for UI
        .fallback_service(
            tower_http::services::ServeDir::new("ui/dist")
                .fallback(tower_http::services::ServeFile::new("ui/dist/index.html")),
        )
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state)
}
