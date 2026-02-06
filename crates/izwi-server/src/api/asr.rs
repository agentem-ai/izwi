//! Qwen3-ASR API endpoints for speech-to-text transcription (native).

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::info;

use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::model::ModelStatus;
use izwi_core::ModelVariant;

/// ASR transcription request
#[derive(Debug, Deserialize)]
pub struct TranscribeRequest {
    pub audio_base64: String,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
}

/// ASR transcription response
#[derive(Debug, Serialize)]
pub struct TranscribeResponse {
    pub transcription: String,
    pub language: Option<String>,
    pub stats: Option<AsrStats>,
}

/// ASR processing statistics
#[derive(Debug, Serialize)]
pub struct AsrStats {
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Audio duration in seconds (if available)
    pub audio_duration_secs: Option<f64>,
    /// Real-time factor (processing_time / audio_duration)
    pub rtf: Option<f64>,
}

/// ASR daemon status response
#[derive(Debug, Serialize)]
pub struct AsrStatusResponse {
    pub running: bool,
    pub status: String,
    pub device: Option<String>,
    pub cached_models: Vec<String>,
}

/// Get ASR native status
pub async fn status(State(state): State<AppState>) -> Result<Json<AsrStatusResponse>, ApiError> {
    let models = state.engine.model_manager().list_models().await;

    let cached_models: Vec<String> = models
        .into_iter()
        .filter(|m| m.variant.is_asr() && m.status == ModelStatus::Ready)
        .map(|m| m.variant.to_string())
        .collect();

    Ok(Json(AsrStatusResponse {
        running: !cached_models.is_empty(),
        status: if cached_models.is_empty() {
            "stopped".to_string()
        } else {
            "running".to_string()
        },
        device: None,
        cached_models,
    }))
}

/// Load the default ASR model (native)
pub async fn start_daemon(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    state.engine.load_model(ModelVariant::Qwen3Asr06B).await?;
    Ok(Json(serde_json::json!({
        "success": true,
        "message": "ASR model loaded"
    })))
}

/// Unload ASR models (native)
pub async fn stop_daemon(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    state.engine.unload_model(ModelVariant::Qwen3Asr06B).await?;
    let _ = state.engine.unload_model(ModelVariant::Qwen3Asr17B).await;
    Ok(Json(serde_json::json!({
        "success": true,
        "message": "ASR model unloaded"
    })))
}

/// Transcribe audio to text with backpressure
pub async fn transcribe(
    State(state): State<AppState>,
    Json(req): Json<TranscribeRequest>,
) -> Result<Json<TranscribeResponse>, ApiError> {
    info!("ASR request: {} bytes", req.audio_base64.len());

    // Acquire permit for concurrency limiting
    let _permit = state.acquire_permit().await;

    let start = Instant::now();
    let result = state
        .engine
        .asr_transcribe(
            &req.audio_base64,
            req.model_id.as_deref(),
            req.language.as_deref(),
        )
        .await?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let rtf = if result.duration_secs > 0.0 {
        Some((elapsed_ms / 1000.0) / result.duration_secs as f64)
    } else {
        None
    };

    Ok(Json(TranscribeResponse {
        transcription: result.text,
        language: result.language,
        stats: Some(AsrStats {
            processing_time_ms: elapsed_ms,
            audio_duration_secs: Some(result.duration_secs as f64),
            rtf,
        }),
    }))
}

/// Streaming transcription (not yet supported for native ASR)
pub async fn transcribe_stream(
    State(_state): State<AppState>,
    Json(_req): Json<TranscribeRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    Err(ApiError::bad_request(
        "Streaming ASR is not yet supported in native mode",
    ))
}
