//! TTS generation API endpoints

use axum::{
    body::Body,
    extract::State,
    http::{header, Response},
    Json,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::info;

use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::audio::AudioFormat;
use izwi_core::inference::{AudioChunk, GenerationConfig, GenerationRequest};

/// TTS generation request
#[derive(Debug, Deserialize)]
pub struct TTSRequest {
    /// Text to synthesize
    pub text: String,

    /// Speaker/voice ID
    #[serde(default)]
    pub speaker: Option<String>,

    /// Voice description (for voice design)
    #[serde(default)]
    pub voice_description: Option<String>,

    /// Reference audio for voice cloning (base64)
    #[serde(default)]
    pub reference_audio: Option<String>,

    /// Reference text (transcript of reference audio)
    #[serde(default)]
    pub reference_text: Option<String>,

    /// Output format (wav, raw_f32, raw_i16)
    #[serde(default = "default_format")]
    pub format: String,

    /// Temperature for sampling
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Speed factor
    #[serde(default)]
    pub speed: Option<f32>,
}

fn default_format() -> String {
    "wav".to_string()
}

/// TTS generation response (non-streaming)
#[derive(Serialize)]
pub struct TTSResponse {
    pub request_id: String,
    pub audio: String, // base64 encoded
    pub format: String,
    pub sample_rate: u32,
    pub duration_secs: f32,
    pub stats: TTSStats,
}

#[derive(Serialize)]
pub struct TTSStats {
    pub tokens_generated: usize,
    pub generation_time_ms: f32,
    pub rtf: f32,
}

/// Generate audio (non-streaming)
pub async fn generate(
    State(state): State<AppState>,
    Json(req): Json<TTSRequest>,
) -> Result<Response<Body>, ApiError> {
    info!("TTS request: {} chars", req.text.len());

    let engine = state.engine.read().await;

    // Build generation request
    let mut gen_config = GenerationConfig::default();
    gen_config.streaming = false;
    if let Some(t) = req.temperature {
        gen_config.temperature = t;
    }
    if let Some(s) = req.speed {
        gen_config.speed = s;
    }
    gen_config.speaker = req.speaker.clone();

    let gen_request = GenerationRequest {
        id: uuid::Uuid::new_v4().to_string(),
        text: req.text,
        config: gen_config,
        reference_audio: req.reference_audio,
        reference_text: req.reference_text,
        voice_description: req.voice_description,
    };

    // Generate audio
    let result = engine.generate(gen_request).await?;

    // Encode to requested format
    let format = parse_format(&req.format)?;
    let encoder = engine.audio_encoder();
    let audio_bytes = encoder.encode(&result.samples, format)?;

    // Return based on format
    let content_type = izwi_core::audio::AudioEncoder::content_type(format);

    if format == AudioFormat::Wav {
        // Return as binary WAV file
        Ok(Response::builder()
            .header(header::CONTENT_TYPE, content_type)
            .header(
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"speech.wav\"",
            )
            .body(Body::from(audio_bytes))
            .unwrap())
    } else {
        // Return as JSON with base64 audio
        use base64::Engine;
        let response = TTSResponse {
            request_id: result.request_id.clone(),
            audio: base64::engine::general_purpose::STANDARD.encode(&audio_bytes),
            format: req.format,
            sample_rate: result.sample_rate,
            duration_secs: result.duration_secs(),
            stats: TTSStats {
                tokens_generated: result.total_tokens,
                generation_time_ms: result.total_time_ms,
                rtf: result.rtf(),
            },
        };
        Ok(Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&response).unwrap()))
            .unwrap())
    }
}

/// Generate audio with streaming
pub async fn generate_stream(
    State(state): State<AppState>,
    Json(req): Json<TTSRequest>,
) -> Result<Response<Body>, ApiError> {
    info!("Streaming TTS request: {} chars", req.text.len());

    let engine = state.engine.read().await;

    // Build generation request
    let mut gen_config = GenerationConfig::default();
    gen_config.streaming = true;
    if let Some(t) = req.temperature {
        gen_config.temperature = t;
    }
    if let Some(s) = req.speed {
        gen_config.speed = s;
    }
    gen_config.speaker = req.speaker.clone();

    let gen_request = GenerationRequest {
        id: uuid::Uuid::new_v4().to_string(),
        text: req.text,
        config: gen_config,
        reference_audio: req.reference_audio,
        reference_text: req.reference_text,
        voice_description: req.voice_description,
    };

    let format = parse_format(&req.format)?;
    let sample_rate = engine.sample_rate();

    // Create channel for streaming chunks
    let (tx, rx) = mpsc::channel::<AudioChunk>(32);

    // Spawn generation task
    let engine_clone = state.engine.clone();
    let request_clone = gen_request.clone();
    tokio::spawn(async move {
        let engine = engine_clone.read().await;
        if let Err(e) = engine.generate_streaming(request_clone, tx).await {
            tracing::error!("Streaming generation error: {}", e);
        }
    });

    // Create stream from receiver
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

fn parse_format(s: &str) -> Result<AudioFormat, ApiError> {
    match s.to_lowercase().as_str() {
        "wav" => Ok(AudioFormat::Wav),
        "raw_f32" | "pcm_f32" => Ok(AudioFormat::RawF32),
        "raw_i16" | "pcm_i16" => Ok(AudioFormat::RawI16),
        _ => Err(ApiError::bad_request(format!(
            "Unknown audio format: {}",
            s
        ))),
    }
}
