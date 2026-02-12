//! Model management API endpoints

use axum::{
    extract::{Path, State},
    response::Sse,
    Json,
};
use futures::stream::Stream;
use serde::Serialize;
use tokio::sync::broadcast::error::RecvError;
use tracing::{info, warn};

use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::model::download::DownloadState;
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
    let mut models: Vec<ModelInfo> = state
        .engine
        .list_models()
        .await
        .into_iter()
        .filter(|model| model.enabled)
        .collect();
    models.sort_by_key(model_sort_key);
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
        let mut last_event: Option<ProgressEvent> = None;
        loop {
            match progress_rx.recv().await {
                Ok(progress) => {
                    let is_completed = progress.files_total > 0
                        && progress.files_completed >= progress.files_total
                        && (progress.total_bytes == 0
                            || progress.downloaded_bytes >= progress.total_bytes);

                    let event = ProgressEvent {
                        variant: progress.variant.dir_name().to_string(),
                        downloaded_bytes: progress.downloaded_bytes,
                        total_bytes: progress.total_bytes,
                        current_file: progress.current_file.clone(),
                        current_file_downloaded: progress.current_file_downloaded,
                        current_file_total: progress.current_file_total,
                        files_completed: progress.files_completed,
                        files_total: progress.files_total,
                        percent: progress.total_percent(),
                        status: if is_completed {
                            "completed".to_string()
                        } else {
                            "downloading".to_string()
                        },
                    };

                    let json = serde_json::to_string(&event).unwrap_or_default();
                    yield Ok(Event::default().data(json));
                    last_event = Some(event.clone());

                    // Stop if download is complete
                    if event.status == "completed" {
                        break;
                    }
                }
                Err(RecvError::Lagged(skipped)) => {
                    warn!(
                        "Download progress stream lagged for {} (skipped {} updates); continuing",
                        variant, skipped
                    );
                    continue;
                }
                Err(RecvError::Closed) => {
                    // Channel closed: the download task has exited. Emit one final state
                    // so clients do not see a silent stream cutoff.
                    let final_state = state
                        .engine
                        .model_manager()
                        .get_download_state(variant)
                        .await;

                    match final_state {
                        DownloadState::Downloaded => {
                            if let Some(mut event) = last_event.clone() {
                                event.status = "completed".to_string();
                                if event.total_bytes > 0 {
                                    event.downloaded_bytes = event.total_bytes;
                                    event.percent = 100.0;
                                }
                                event.files_completed = event.files_total.max(event.files_completed);
                                event.current_file = String::new();
                                event.current_file_downloaded = 0;
                                event.current_file_total = 0;
                                let json = serde_json::to_string(&event).unwrap_or_default();
                                yield Ok(Event::default().data(json));
                            } else {
                                let event = ProgressEvent {
                                    variant: variant.dir_name().to_string(),
                                    downloaded_bytes: 0,
                                    total_bytes: 0,
                                    current_file: String::new(),
                                    current_file_downloaded: 0,
                                    current_file_total: 0,
                                    files_completed: 0,
                                    files_total: 0,
                                    percent: 100.0,
                                    status: "completed".to_string(),
                                };
                                let json = serde_json::to_string(&event).unwrap_or_default();
                                yield Ok(Event::default().data(json));
                            }
                        }
                        DownloadState::Error => {
                            let mut event = last_event.clone().unwrap_or(ProgressEvent {
                                variant: variant.dir_name().to_string(),
                                downloaded_bytes: 0,
                                total_bytes: 0,
                                current_file: String::new(),
                                current_file_downloaded: 0,
                                current_file_total: 0,
                                files_completed: 0,
                                files_total: 0,
                                percent: 0.0,
                                status: "error".to_string(),
                            });
                            event.status = "error".to_string();
                            let json = serde_json::to_string(&event).unwrap_or_default();
                            yield Ok(Event::default().data(json));
                        }
                        DownloadState::NotDownloaded => {
                            let mut event = last_event.clone().unwrap_or(ProgressEvent {
                                variant: variant.dir_name().to_string(),
                                downloaded_bytes: 0,
                                total_bytes: 0,
                                current_file: String::new(),
                                current_file_downloaded: 0,
                                current_file_total: 0,
                                files_completed: 0,
                                files_total: 0,
                                percent: 0.0,
                                status: "cancelled".to_string(),
                            });
                            event.status = "cancelled".to_string();
                            let json = serde_json::to_string(&event).unwrap_or_default();
                            yield Ok(Event::default().data(json));
                        }
                        DownloadState::Downloading => {}
                    }
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

fn model_sort_key(model: &ModelInfo) -> (u8, u8, u8, String) {
    let variant = model.variant;
    let dir_name = variant.dir_name();
    (
        model_type_rank(variant),
        model_size_rank(dir_name),
        model_precision_rank(dir_name),
        dir_name.to_string(),
    )
}

fn model_type_rank(variant: ModelVariant) -> u8 {
    if variant.is_asr() {
        0
    } else if variant.is_tts() {
        1
    } else if variant.is_chat() {
        2
    } else if variant.is_forced_aligner() {
        3
    } else if variant.is_tokenizer() {
        4
    } else if variant.is_lfm2() {
        5
    } else if variant.is_voxtral() {
        6
    } else {
        7
    }
}

fn model_size_rank(dir_name: &str) -> u8 {
    if dir_name.contains("0.6B") {
        0
    } else if dir_name.contains("1.7B") {
        1
    } else {
        2
    }
}

fn model_precision_rank(dir_name: &str) -> u8 {
    if dir_name.contains("-4bit") {
        2
    } else if dir_name.contains("-8bit") {
        3
    } else if dir_name.contains("-bf16") {
        1
    } else {
        0
    }
}
