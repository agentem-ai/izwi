//! OpenAI-compatible speech synthesis endpoints.

use axum::{
    body::Body,
    extract::State,
    http::{header, Response},
    Json,
};
use futures::StreamExt;
use serde::Deserialize;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::info;

use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::audio::AudioFormat;
use izwi_core::{parse_tts_model_variant, AudioChunk, GenerationConfig, GenerationRequest};

/// OpenAI-compatible speech synthesis request.
#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    /// OpenAI field name for TTS model.
    pub model: String,
    /// OpenAI field name for text input.
    pub input: String,
    /// OpenAI-style voice selection.
    #[serde(default)]
    pub voice: Option<String>,
    /// OpenAI response format (`wav` and `pcm` currently supported).
    #[serde(default)]
    pub response_format: Option<String>,
    /// OpenAI speed.
    #[serde(default)]
    pub speed: Option<f32>,
    /// Optional language hint (e.g. "Auto", "English", "Chinese").
    #[serde(default)]
    pub language: Option<String>,
    /// Optional sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Optional max token budget.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// If true, stream chunked audio from same endpoint.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Optional voice design prompt.
    #[serde(default)]
    pub instructions: Option<String>,
    /// Optional reference audio (base64) for voice cloning.
    #[serde(default)]
    pub reference_audio: Option<String>,
    /// Optional reference transcript for cloning.
    #[serde(default)]
    pub reference_text: Option<String>,
}

pub async fn speech(
    State(state): State<AppState>,
    Json(req): Json<SpeechRequest>,
) -> Result<Response<Body>, ApiError> {
    info!("OpenAI speech request: {} chars", req.input.len());

    let variant = parse_tts_model_variant(&req.model)
        .map_err(|err| ApiError::bad_request(format!("Unsupported TTS model: {}", err)))?;
    state.engine.load_model(variant).await?;

    if req.stream.unwrap_or(false) {
        return stream_speech(state, req).await;
    }

    let _permit = state.acquire_permit().await;

    let timeout = Duration::from_secs(state.request_timeout_secs);
    let format = parse_response_format(req.response_format.as_deref().unwrap_or("wav"))?;

    let result = tokio::time::timeout(timeout, async {
        let mut gen_config = GenerationConfig {
            streaming: false,
            ..GenerationConfig::default()
        };
        if let Some(temp) = req.temperature {
            gen_config.temperature = temp;
        }
        if let Some(speed) = req.speed {
            gen_config.speed = speed;
        }
        if let Some(max_tokens) = req.max_tokens {
            gen_config.max_tokens = max_tokens;
        }
        gen_config.speaker = req.voice.clone();

        let gen_request = GenerationRequest {
            id: uuid::Uuid::new_v4().to_string(),
            text: req.input.clone(),
            config: gen_config,
            language: req.language.clone(),
            reference_audio: req.reference_audio.clone(),
            reference_text: req.reference_text.clone(),
            voice_description: req.instructions.clone(),
        };

        state.engine.generate(gen_request).await
    })
    .await
    .map_err(|_| ApiError::internal("Request timeout"))??;

    let encoder = state.engine.audio_encoder().await;
    let samples = result.samples.clone();
    let audio_bytes = tokio::task::spawn_blocking(move || encoder.encode(&samples, format))
        .await
        .map_err(|e| ApiError::internal(format!("Audio encoding failed: {}", e)))??;

    let content_type = izwi_core::audio::AudioEncoder::content_type(format);
    let duration_secs = result.duration_secs();
    let generation_time_ms = result.total_time_ms;
    let rtf = result.rtf();
    let tokens_generated = result.total_tokens;

    Ok(Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .header("X-Generation-Time-Ms", format!("{:.1}", generation_time_ms))
        .header("X-Audio-Duration-Secs", format!("{:.2}", duration_secs))
        .header("X-RTF", format!("{:.3}", rtf))
        .header("X-Tokens-Generated", tokens_generated.to_string())
        .header(
            "Access-Control-Expose-Headers",
            "X-Generation-Time-Ms, X-Audio-Duration-Secs, X-RTF, X-Tokens-Generated",
        )
        .body(Body::from(audio_bytes))
        .unwrap())
}

async fn stream_speech(state: AppState, req: SpeechRequest) -> Result<Response<Body>, ApiError> {
    let format = parse_response_format(req.response_format.as_deref().unwrap_or("wav"))?;

    let mut gen_config = GenerationConfig {
        streaming: true,
        ..GenerationConfig::default()
    };
    if let Some(temp) = req.temperature {
        gen_config.temperature = temp;
    }
    if let Some(speed) = req.speed {
        gen_config.speed = speed;
    }
    if let Some(max_tokens) = req.max_tokens {
        gen_config.max_tokens = max_tokens;
    }
    gen_config.speaker = req.voice.clone();

    let gen_request = GenerationRequest {
        id: uuid::Uuid::new_v4().to_string(),
        text: req.input.clone(),
        config: gen_config,
        language: req.language.clone(),
        reference_audio: req.reference_audio.clone(),
        reference_text: req.reference_text.clone(),
        voice_description: req.instructions.clone(),
    };

    let sample_rate = state.engine.sample_rate().await;
    let (tx, rx) = mpsc::channel::<AudioChunk>(32);

    let engine = state.engine.clone();
    let timeout = Duration::from_secs(state.request_timeout_secs);
    tokio::spawn(async move {
        let result = tokio::time::timeout(timeout, async {
            engine.generate_streaming(gen_request, tx).await
        })
        .await;

        if result.is_err() {
            tracing::error!("Streaming generation timeout");
        } else if let Err(err) = result.unwrap_or_else(|_| Ok(())) {
            tracing::error!("Streaming generation error: {}", err);
        }
    });

    let encoder = izwi_core::audio::AudioEncoder::new(sample_rate, 1);
    let stream = ReceiverStream::new(rx).map(move |chunk| {
        let bytes = encoder.encode(&chunk.samples, format).unwrap_or_default();
        Ok::<_, std::convert::Infallible>(bytes)
    });

    let content_type = izwi_core::audio::AudioEncoder::content_type(format);

    Ok(Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .header(header::TRANSFER_ENCODING, "chunked")
        .body(Body::from_stream(stream))
        .unwrap())
}

fn parse_response_format(format: &str) -> Result<AudioFormat, ApiError> {
    match format.to_ascii_lowercase().as_str() {
        "wav" => Ok(AudioFormat::Wav),
        "pcm" => Ok(AudioFormat::RawI16),
        "raw_f32" | "pcm_f32" => Ok(AudioFormat::RawF32),
        "raw_i16" | "pcm_i16" => Ok(AudioFormat::RawI16),
        unsupported => Err(ApiError::bad_request(format!(
            "Unsupported response_format: {}. Supported formats: wav, pcm",
            unsupported
        ))),
    }
}
