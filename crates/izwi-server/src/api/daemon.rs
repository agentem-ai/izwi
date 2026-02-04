//! Daemon management API endpoints
//!
//! Note: These endpoints are deprecated since the migration to native Rust models.
//! Native models don't use Python daemons, so these endpoints return
//! "not applicable" responses for backward compatibility.

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::state::AppState;

/// Daemon status response
#[derive(Debug, Serialize)]
pub struct DaemonStatus {
    pub running: bool,
    pub device: Option<String>,
    pub cached_models: Vec<String>,
    pub message: String,
}

/// Preload model request
#[derive(Debug, Deserialize)]
pub struct PreloadRequest {
    pub model_path: String,
}

/// Generic response
#[derive(Debug, Serialize)]
pub struct DaemonResponse {
    pub success: bool,
    pub message: String,
}

/// Get daemon status
/// 
/// Since native Rust models don't use daemons, this always returns
/// a status indicating the native engine is ready.
pub async fn get_status(State(_state): State<AppState>) -> Json<DaemonStatus> {
    info!("Daemon status check (native models - no daemon needed)");

    Json(DaemonStatus {
        running: true,
        device: Some("native".to_string()),
        cached_models: vec![],
        message: "Native Rust inference engine (no daemon)".to_string(),
    })
}

/// Start the daemon
/// 
/// Deprecated: Native Rust models don't use daemons.
pub async fn start_daemon(
    State(_state): State<AppState>,
) -> Result<Json<DaemonResponse>, (StatusCode, Json<DaemonResponse>)> {
    info!("Daemon start requested (native models - no daemon needed)");

    Ok(Json(DaemonResponse {
        success: true,
        message: "Native Rust inference engine active (no daemon required)".to_string(),
    }))
}

/// Stop the daemon
/// 
/// Deprecated: Native Rust models don't use daemons.
pub async fn stop_daemon(
    State(_state): State<AppState>,
) -> Result<Json<DaemonResponse>, (StatusCode, Json<DaemonResponse>)> {
    info!("Daemon stop requested (native models - no daemon needed)");

    Ok(Json(DaemonResponse {
        success: true,
        message: "Native Rust inference engine (no daemon to stop)".to_string(),
    }))
}

/// Preload a model into the daemon cache
/// 
/// Deprecated: Model loading is handled on-demand by the native engine.
pub async fn preload_model(
    State(_state): State<AppState>,
    Json(request): Json<PreloadRequest>,
) -> Result<Json<DaemonResponse>, (StatusCode, Json<DaemonResponse>)> {
    info!("Model preload requested: {} (native models handle loading on-demand)", request.model_path);

    Ok(Json(DaemonResponse {
        success: true,
        message: format!("Native engine will load '{}' on first use", request.model_path),
    }))
}
