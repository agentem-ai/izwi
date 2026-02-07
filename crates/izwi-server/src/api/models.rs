//! Model management API endpoints

use axum::{
    extract::{Path, State},
    response::Sse,
    Json,
};
use futures::stream::Stream;
use serde::Serialize;
use tracing::{info, warn};

use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::{parse_model_variant, ModelInfo, ModelVariant};

/// Response for model list
#[derive(Serialize)]
pub struct ModelsResponse {
    pub models: Vec<ModelInfo>,
}

#[derive(Serialize)]
pub struct OpenAiModelsResponse {
    pub object: &'static str,
    pub data: Vec<OpenAiModel>,
}

#[derive(Serialize)]
pub struct OpenAiModel {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

/// List all available models
pub async fn list_models(State(state): State<AppState>) -> Result<Json<ModelsResponse>, ApiError> {
    let models = state.engine.list_models().await;
    Ok(Json(ModelsResponse { models }))
}

/// OpenAI-compatible model listing.
pub async fn list_models_openai(
    State(state): State<AppState>,
) -> Result<Json<OpenAiModelsResponse>, ApiError> {
    let models = state.engine.list_models().await;
    let data = models
        .into_iter()
        .map(|model| OpenAiModel {
            id: model.variant.dir_name().to_string(),
            object: "model",
            owned_by: "agentem",
        })
        .collect();

    Ok(Json(OpenAiModelsResponse {
        object: "list",
        data,
    }))
}

/// OpenAI-compatible model retrieval.
pub async fn get_model_openai(Path(model): Path<String>) -> Result<Json<OpenAiModel>, ApiError> {
    let variant = parse_variant(&model)?;
    Ok(Json(OpenAiModel {
        id: variant.dir_name().to_string(),
        object: "model",
        owned_by: "agentem",
    }))
}

/// Get info for a specific model
pub async fn get_model_info(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<ModelInfo>, ApiError> {
    let variant = parse_variant(&variant)?;

    let info = state
        .engine
        .model_manager()
        .get_model_info(variant)
        .await
        .ok_or_else(|| ApiError::not_found("Model not found"))?;

    Ok(Json(info))
}

/// SSE progress event
#[derive(Serialize, Clone)]
struct ProgressEvent {
    variant: String,
    downloaded_bytes: u64,
    total_bytes: u64,
    current_file: String,
    current_file_downloaded: u64,
    current_file_total: u64,
    files_completed: usize,
    files_total: usize,
    percent: f32,
    status: String, // "downloading", "completed", "error"
}

/// Download progress response
#[derive(Serialize)]
pub struct DownloadResponse {
    pub status: &'static str,
    pub message: String,
}

/// Download a model from HuggingFace (non-blocking)
pub async fn download_model(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<DownloadResponse>, ApiError> {
    let variant = parse_variant(&variant)?;
    info!("Starting non-blocking download for model: {}", variant);

    // Check if already downloading
    if state.engine.is_download_active(variant).await {
        return Ok(Json(DownloadResponse {
            status: "downloading",
            message: format!("Model {} is already being downloaded", variant),
        }));
    }

    // Spawn non-blocking download
    state.engine.spawn_download(variant).await?;

    Ok(Json(DownloadResponse {
        status: "started",
        message: format!("Download of {} started in background", variant),
    }))
}

/// Load a model into memory
pub async fn load_model(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<DownloadResponse>, ApiError> {
    let variant = parse_variant(&variant)?;
    info!("Loading model: {}", variant);

    state.engine.load_model(variant).await?;

    Ok(Json(DownloadResponse {
        status: "loaded",
        message: format!("Model {} loaded successfully", variant),
    }))
}

/// Unload a model from memory
pub async fn unload_model(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<DownloadResponse>, ApiError> {
    let variant = parse_variant(&variant)?;
    info!("Unloading model: {}", variant);

    state.engine.unload_model(variant).await?;

    Ok(Json(DownloadResponse {
        status: "unloaded",
        message: format!("Model {} unloaded successfully", variant),
    }))
}

/// Delete a downloaded model from disk
pub async fn delete_model(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<DownloadResponse>, ApiError> {
    let variant = parse_variant(&variant)?;
    info!("Deleting model: {}", variant);

    // First unload if loaded
    let _ = state.engine.model_manager().unload_model(variant).await;

    // Delete the model files
    state.engine.model_manager().delete_model(variant).await?;

    Ok(Json(DownloadResponse {
        status: "deleted",
        message: format!("Model {} deleted successfully", variant),
    }))
}

/// Stream download progress as SSE
pub async fn download_progress_stream(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<
    Sse<impl Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>>,
    ApiError,
> {
    use axum::response::sse::Event;

    let variant = parse_variant(&variant)?;
    info!("Starting SSE progress stream for: {}", variant);

    // Get progress receiver from engine
    let mut progress_rx = state
        .engine
        .model_manager()
        .subscribe_progress(variant)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to subscribe to progress: {}", e)))?;

    let stream = async_stream::stream! {
        loop {
            match progress_rx.recv().await {
                Ok(progress) => {
                    let event = ProgressEvent {
                        variant: progress.variant.to_string(),
                        downloaded_bytes: progress.downloaded_bytes,
                        total_bytes: progress.total_bytes,
                        current_file: progress.current_file.clone(),
                        current_file_downloaded: progress.current_file_downloaded,
                        current_file_total: progress.current_file_total,
                        files_completed: progress.files_completed,
                        files_total: progress.files_total,
                        percent: progress.total_percent(),
                        status: if progress.files_completed >= progress.files_total {
                            "completed".to_string()
                        } else {
                            "downloading".to_string()
                        },
                    };

                    let json = serde_json::to_string(&event).unwrap_or_default();
                    yield Ok(Event::default().data(json));

                    // Stop if download is complete
                    if event.status == "completed" {
                        break;
                    }
                }
                Err(_) => {
                    // Channel closed or lagged, stop streaming
                    break;
                }
            }
        }
    };

    Ok(Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default()))
}

/// Cancel an active download
pub async fn cancel_download(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<DownloadResponse>, ApiError> {
    let variant = parse_variant(&variant)?;
    info!("Cancelling download for: {}", variant);

    // Check if download is active
    if !state.engine.is_download_active(variant).await {
        return Ok(Json(DownloadResponse {
            status: "not_active",
            message: format!("No active download for {}", variant),
        }));
    }

    // Cancel the download
    match state.engine.model_manager().cancel_download(variant).await {
        Ok(_) => Ok(Json(DownloadResponse {
            status: "cancelled",
            message: format!("Download of {} cancelled", variant),
        })),
        Err(e) => {
            warn!("Failed to cancel download: {}", e);
            Err(ApiError::internal(format!(
                "Failed to cancel download: {}",
                e
            )))
        }
    }
}

/// Parse model variant from string
fn parse_variant(s: &str) -> Result<ModelVariant, ApiError> {
    parse_model_variant(s).map_err(|err| ApiError::bad_request(err.to_string()))
}
