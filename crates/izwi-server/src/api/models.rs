//! Model management API endpoints

use axum::{
    extract::{Path, State},
    response::Sse,
    Json,
};
use futures::stream::Stream;
use serde::Serialize;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::{ModelInfo, ModelVariant};

/// Response for model list
#[derive(Serialize)]
pub struct ModelsResponse {
    pub models: Vec<ModelInfo>,
}

/// List all available models
pub async fn list_models(State(state): State<AppState>) -> Result<Json<ModelsResponse>, ApiError> {
    let models = state.engine.list_models().await;
    Ok(Json(ModelsResponse { models }))
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
    // Exact matches for HuggingFace model names (used in URLs)
    match s {
        "Qwen3-TTS-12Hz-0.6B-Base" => return Ok(ModelVariant::Qwen3Tts12Hz06BBase),
        "Qwen3-TTS-12Hz-0.6B-Base-4bit" => return Ok(ModelVariant::Qwen3Tts12Hz06BBase4Bit),
        "Qwen3-TTS-12Hz-0.6B-Base-8bit" => return Ok(ModelVariant::Qwen3Tts12Hz06BBase8Bit),
        "Qwen3-TTS-12Hz-0.6B-Base-bf16" => return Ok(ModelVariant::Qwen3Tts12Hz06BBaseBf16),
        "Qwen3-TTS-12Hz-0.6B-CustomVoice" => return Ok(ModelVariant::Qwen3Tts12Hz06BCustomVoice),
        "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit" => {
            return Ok(ModelVariant::Qwen3Tts12Hz06BCustomVoice4Bit)
        }
        "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit" => {
            return Ok(ModelVariant::Qwen3Tts12Hz06BCustomVoice8Bit)
        }
        "Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16" => {
            return Ok(ModelVariant::Qwen3Tts12Hz06BCustomVoiceBf16)
        }
        "Qwen3-TTS-12Hz-1.7B-Base" => return Ok(ModelVariant::Qwen3Tts12Hz17BBase),
        "Qwen3-TTS-12Hz-1.7B-CustomVoice" => return Ok(ModelVariant::Qwen3Tts12Hz17BCustomVoice),
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign" => return Ok(ModelVariant::Qwen3Tts12Hz17BVoiceDesign),
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit" => {
            return Ok(ModelVariant::Qwen3Tts12Hz17BVoiceDesign4Bit)
        }
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit" => {
            return Ok(ModelVariant::Qwen3Tts12Hz17BVoiceDesign8Bit)
        }
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16" => {
            return Ok(ModelVariant::Qwen3Tts12Hz17BVoiceDesignBf16)
        }
        "Qwen3-TTS-Tokenizer-12Hz" => return Ok(ModelVariant::Qwen3TtsTokenizer12Hz),
        "LFM2-Audio-1.5B" => return Ok(ModelVariant::Lfm2Audio15B),
        "Qwen3-ASR-0.6B" => return Ok(ModelVariant::Qwen3Asr06B),
        "Qwen3-ASR-0.6B-4bit" => return Ok(ModelVariant::Qwen3Asr06B4Bit),
        "Qwen3-ASR-0.6B-8bit" => return Ok(ModelVariant::Qwen3Asr06B8Bit),
        "Qwen3-ASR-0.6B-bf16" => return Ok(ModelVariant::Qwen3Asr06BBf16),
        "Qwen3-ASR-1.7B" => return Ok(ModelVariant::Qwen3Asr17B),
        "Qwen3-ASR-1.7B-4bit" => return Ok(ModelVariant::Qwen3Asr17B4Bit),
        "Qwen3-ASR-1.7B-8bit" => return Ok(ModelVariant::Qwen3Asr17B8Bit),
        "Qwen3-ASR-1.7B-bf16" => return Ok(ModelVariant::Qwen3Asr17BBf16),
        "Qwen3-0.6B-4bit" => return Ok(ModelVariant::Qwen306B4Bit),
        "Voxtral-Mini-4B-Realtime-2602" => return Ok(ModelVariant::VoxtralMini4BRealtime2602),
        _ => {}
    }

    // Fallback: normalize and try pattern matching
    let normalized = s.to_lowercase().replace("-", "_").replace(".", "");

    // Qwen3-ASR models (check before TTS to avoid conflicts)
    if normalized.contains("qwen3") && normalized.contains("asr") {
        let is_06b = normalized.contains("06b");
        let is_17b = normalized.contains("17b");
        let target_variant = if is_06b {
            if normalized.contains("4bit") {
                ModelVariant::Qwen3Asr06B4Bit
            } else if normalized.contains("8bit") {
                ModelVariant::Qwen3Asr06B8Bit
            } else if normalized.contains("bf16") {
                ModelVariant::Qwen3Asr06BBf16
            } else {
                ModelVariant::Qwen3Asr06B
            }
        } else if is_17b {
            if normalized.contains("4bit") {
                ModelVariant::Qwen3Asr17B4Bit
            } else if normalized.contains("8bit") {
                ModelVariant::Qwen3Asr17B8Bit
            } else if normalized.contains("bf16") {
                ModelVariant::Qwen3Asr17BBf16
            } else {
                ModelVariant::Qwen3Asr17B
            }
        } else {
            return Err(ApiError::bad_request(format!("Unknown ASR variant: {}", s)));
        };
        return Ok(target_variant);
    }

    if normalized.contains("06b") && normalized.contains("base") {
        if normalized.contains("4bit") {
            return Ok(ModelVariant::Qwen3Tts12Hz06BBase4Bit);
        } else if normalized.contains("8bit") {
            return Ok(ModelVariant::Qwen3Tts12Hz06BBase8Bit);
        } else if normalized.contains("bf16") {
            return Ok(ModelVariant::Qwen3Tts12Hz06BBaseBf16);
        }
        return Ok(ModelVariant::Qwen3Tts12Hz06BBase);
    }
    if normalized.contains("06b") && normalized.contains("custom") {
        if normalized.contains("4bit") {
            return Ok(ModelVariant::Qwen3Tts12Hz06BCustomVoice4Bit);
        } else if normalized.contains("8bit") {
            return Ok(ModelVariant::Qwen3Tts12Hz06BCustomVoice8Bit);
        } else if normalized.contains("bf16") {
            return Ok(ModelVariant::Qwen3Tts12Hz06BCustomVoiceBf16);
        }
        return Ok(ModelVariant::Qwen3Tts12Hz06BCustomVoice);
    }
    if normalized.contains("17b")
        && normalized.contains("base")
        && !normalized.contains("custom")
        && !normalized.contains("voice")
    {
        return Ok(ModelVariant::Qwen3Tts12Hz17BBase);
    }
    if normalized.contains("17b") && normalized.contains("customvoice") {
        return Ok(ModelVariant::Qwen3Tts12Hz17BCustomVoice);
    }
    if normalized.contains("17b") && normalized.contains("voicedesign") {
        if normalized.contains("4bit") {
            return Ok(ModelVariant::Qwen3Tts12Hz17BVoiceDesign4Bit);
        } else if normalized.contains("8bit") {
            return Ok(ModelVariant::Qwen3Tts12Hz17BVoiceDesign8Bit);
        } else if normalized.contains("bf16") {
            return Ok(ModelVariant::Qwen3Tts12Hz17BVoiceDesignBf16);
        }
        return Ok(ModelVariant::Qwen3Tts12Hz17BVoiceDesign);
    }
    if normalized.contains("tokenizer") {
        return Ok(ModelVariant::Qwen3TtsTokenizer12Hz);
    }
    if normalized.contains("lfm2") && normalized.contains("audio") {
        return Ok(ModelVariant::Lfm2Audio15B);
    }
    if normalized.contains("qwen3")
        && normalized.contains("06b")
        && normalized.contains("4bit")
        && !normalized.contains("tts")
        && !normalized.contains("asr")
    {
        return Ok(ModelVariant::Qwen306B4Bit);
    }
    if normalized.contains("voxtral") {
        return Ok(ModelVariant::VoxtralMini4BRealtime2602);
    }

    Err(ApiError::bad_request(format!(
        "Unknown model variant: {}. Valid variants: Qwen3-TTS-12Hz-0.6B-Base, Qwen3-TTS-12Hz-0.6B-CustomVoice, Qwen3-TTS-12Hz-1.7B-Base, Qwen3-TTS-12Hz-1.7B-CustomVoice, Qwen3-TTS-12Hz-1.7B-VoiceDesign, Qwen3-TTS-Tokenizer-12Hz, LFM2-Audio-1.5B, Qwen3-ASR-0.6B, Qwen3-ASR-1.7B, Qwen3-0.6B-4bit, Voxtral-Mini-4B-Realtime-2602",
        s
    )))
}
