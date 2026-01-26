//! Model management API endpoints

use axum::{
    extract::{Path, State},
    Json,
};
use serde::Serialize;
use tracing::info;

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
    let engine = state.engine.read().await;
    let models = engine.list_models().await;
    Ok(Json(ModelsResponse { models }))
}

/// Get info for a specific model
pub async fn get_model_info(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<ModelInfo>, ApiError> {
    let variant = parse_variant(&variant)?;
    let engine = state.engine.read().await;

    let info = engine
        .model_manager()
        .get_model_info(variant)
        .await
        .ok_or_else(|| ApiError::not_found("Model not found"))?;

    Ok(Json(info))
}

/// Download progress response
#[derive(Serialize)]
pub struct DownloadResponse {
    pub status: &'static str,
    pub message: String,
}

/// Download a model from HuggingFace
pub async fn download_model(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<DownloadResponse>, ApiError> {
    let variant = parse_variant(&variant)?;
    info!("Downloading model: {}", variant);

    let engine = state.engine.read().await;
    engine.download_model(variant).await?;

    Ok(Json(DownloadResponse {
        status: "completed",
        message: format!("Model {} downloaded successfully", variant),
    }))
}

/// Load a model into memory
pub async fn load_model(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<DownloadResponse>, ApiError> {
    let variant = parse_variant(&variant)?;
    info!("Loading model: {}", variant);

    let mut engine = state.engine.write().await;
    engine.load_model(variant).await?;

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

    let engine = state.engine.read().await;
    engine.model_manager().unload_model(variant).await?;

    Ok(Json(DownloadResponse {
        status: "unloaded",
        message: format!("Model {} unloaded successfully", variant),
    }))
}

/// Parse model variant from string
fn parse_variant(s: &str) -> Result<ModelVariant, ApiError> {
    // Exact matches for HuggingFace model names (used in URLs)
    match s {
        "Qwen3-TTS-12Hz-0.6B-Base" => return Ok(ModelVariant::Qwen3Tts12Hz06BBase),
        "Qwen3-TTS-12Hz-0.6B-CustomVoice" => return Ok(ModelVariant::Qwen3Tts12Hz06BCustomVoice),
        "Qwen3-TTS-12Hz-1.7B-Base" => return Ok(ModelVariant::Qwen3Tts12Hz17BBase),
        "Qwen3-TTS-12Hz-1.7B-CustomVoice" => return Ok(ModelVariant::Qwen3Tts12Hz17BCustomVoice),
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign" => return Ok(ModelVariant::Qwen3Tts12Hz17BVoiceDesign),
        "Qwen3-TTS-Tokenizer-12Hz" => return Ok(ModelVariant::Qwen3TtsTokenizer12Hz),
        _ => {}
    }

    // Fallback: normalize and try pattern matching
    let normalized = s.to_lowercase().replace("-", "_").replace(".", "");

    if normalized.contains("06b") && normalized.contains("base") && !normalized.contains("custom") {
        return Ok(ModelVariant::Qwen3Tts12Hz06BBase);
    }
    if normalized.contains("06b") && normalized.contains("custom") {
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
        return Ok(ModelVariant::Qwen3Tts12Hz17BVoiceDesign);
    }
    if normalized.contains("tokenizer") {
        return Ok(ModelVariant::Qwen3TtsTokenizer12Hz);
    }

    Err(ApiError::bad_request(format!(
        "Unknown model variant: {}. Valid variants: Qwen3-TTS-12Hz-0.6B-Base, Qwen3-TTS-12Hz-0.6B-CustomVoice, Qwen3-TTS-12Hz-1.7B-Base, Qwen3-TTS-12Hz-1.7B-CustomVoice, Qwen3-TTS-12Hz-1.7B-VoiceDesign, Qwen3-TTS-Tokenizer-12Hz",
        s
    )))
}
