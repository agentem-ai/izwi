//! API routes and handlers

mod asr;
mod chat;
mod health;
mod models;
mod tts;

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::state::AppState;

/// Create the main API router
pub fn create_router(state: AppState) -> Router {
    let max_upload_mb = std::env::var("IZWI_MAX_UPLOAD_MB")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(64);
    let max_upload_bytes = max_upload_mb.saturating_mul(1024 * 1024);

    let v1_routes = Router::new()
        // Health check
        .route("/health", get(health::health_check))
        // OpenAI-compatible model endpoints
        .route("/models", get(models::list_models_openai))
        .route("/models/:model", get(models::get_model_openai))
        // OpenAI-compatible audio and chat endpoints
        .route("/audio/speech", post(tts::speech))
        .route(
            "/audio/transcriptions",
            post(asr::transcriptions).layer(DefaultBodyLimit::max(max_upload_bytes)),
        )
        .route("/chat/completions", post(chat::completions))
        // Admin model management endpoints for local runtime operations
        .route("/admin/models", get(models::list_models))
        // Model management
        .route(
            "/admin/models/:variant/download",
            post(models::download_model),
        )
        .route(
            "/admin/models/:variant/download/progress",
            get(models::download_progress_stream),
        )
        .route(
            "/admin/models/:variant/download/cancel",
            post(models::cancel_download),
        )
        .route("/admin/models/:variant/load", post(models::load_model))
        .route("/admin/models/:variant/unload", post(models::unload_model))
        .route(
            "/admin/models/:variant",
            get(models::get_model_info).delete(models::delete_model),
        );

    Router::new()
        .nest("/v1", v1_routes)
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
