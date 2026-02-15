//! OpenAI-compatible diarization endpoint.

use axum::{
    body::Body,
    extract::{Multipart, Request, State},
    http::{header, StatusCode},
    response::Response,
    Json, RequestExt,
};
use base64::Engine;
use std::time::Instant;

use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::{DiarizationConfig, DiarizationResult, DiarizationSegment};

#[derive(Debug, Default)]
struct DiarizationRequest {
    audio_base64: Option<String>,
    model: Option<String>,
    response_format: Option<String>,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
    min_speech_duration_ms: Option<f32>,
    min_silence_duration_ms: Option<f32>,
    stream: bool,
}

#[derive(Debug, serde::Serialize)]
struct JsonSegment {
    speaker: String,
    start: f32,
    end: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    confidence: Option<f32>,
}

#[derive(Debug, serde::Serialize)]
struct JsonDiarizationResponse {
    segments: Vec<JsonSegment>,
}

#[derive(Debug, serde::Serialize)]
struct VerboseJsonDiarizationResponse {
    segments: Vec<JsonSegment>,
    speaker_count: usize,
    duration: f32,
    processing_time_ms: f64,
    rtf: Option<f64>,
}

pub async fn diarizations(
    State(state): State<AppState>,
    req: Request,
) -> Result<Response<Body>, ApiError> {
    let mut req = parse_diarization_request(req).await?;
    if req.stream {
        return Err(ApiError::bad_request(
            "Streaming diarization is not supported for /v1/audio/diarizations",
        ));
    }

    let audio_base64 = req
        .audio_base64
        .take()
        .ok_or_else(|| ApiError::bad_request("Missing audio input (`file` or `audio_base64`)"))?;

    let _permit = state.acquire_permit().await;

    let config = DiarizationConfig {
        min_speakers: req.min_speakers,
        max_speakers: req.max_speakers,
        min_speech_duration_ms: req.min_speech_duration_ms,
        min_silence_duration_ms: req.min_silence_duration_ms,
    };

    let started = Instant::now();
    let output = state
        .runtime
        .diarize(&audio_base64, req.model.as_deref(), &config)
        .await?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let rtf = if output.duration_secs > 0.0 {
        Some((elapsed_ms / 1000.0) / output.duration_secs as f64)
    } else {
        None
    };

    let response_format = req
        .response_format
        .as_deref()
        .unwrap_or("json")
        .to_ascii_lowercase();

    let segments = map_segments(&output.segments);

    match response_format.as_str() {
        "json" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_string(&JsonDiarizationResponse { segments }).unwrap(),
            ))
            .unwrap()),
        "verbose_json" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_string(&VerboseJsonDiarizationResponse {
                    segments,
                    speaker_count: output.speaker_count,
                    duration: output.duration_secs,
                    processing_time_ms: elapsed_ms,
                    rtf,
                })
                .unwrap(),
            ))
            .unwrap()),
        "text" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .body(Body::from(format_segments_text(&output)))
            .unwrap()),
        other => Err(ApiError::bad_request(format!(
            "Unsupported response_format: {}. Supported: json, verbose_json, text",
            other
        ))),
    }
}

fn map_segments(segments: &[DiarizationSegment]) -> Vec<JsonSegment> {
    segments
        .iter()
        .map(|segment| JsonSegment {
            speaker: segment.speaker.clone(),
            start: segment.start_secs,
            end: segment.end_secs,
            confidence: segment.confidence,
        })
        .collect()
}

fn format_segments_text(output: &DiarizationResult) -> String {
    let mut out = String::new();
    for segment in &output.segments {
        let duration = (segment.end_secs - segment.start_secs).max(0.0);
        out.push_str(&format!(
            "SPEAKER unknown 1 {:.3} {:.3} <NA> <NA> {} <NA> <NA>\n",
            segment.start_secs, duration, segment.speaker
        ));
    }
    out
}

#[derive(Debug, serde::Deserialize)]
struct JsonRequestBody {
    audio_base64: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    min_speakers: Option<usize>,
    #[serde(default)]
    max_speakers: Option<usize>,
    #[serde(default)]
    min_speech_duration_ms: Option<f32>,
    #[serde(default)]
    min_silence_duration_ms: Option<f32>,
    #[serde(default)]
    stream: Option<bool>,
}

async fn parse_diarization_request(req: Request) -> Result<DiarizationRequest, ApiError> {
    let content_type = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if content_type.starts_with("application/json") {
        let Json(payload) = req
            .extract::<Json<JsonRequestBody>, _>()
            .await
            .map_err(|e| ApiError::bad_request(format!("Invalid JSON payload: {e}")))?;

        return Ok(DiarizationRequest {
            audio_base64: Some(payload.audio_base64),
            model: payload.model,
            response_format: payload.response_format,
            min_speakers: payload.min_speakers,
            max_speakers: payload.max_speakers,
            min_speech_duration_ms: payload.min_speech_duration_ms,
            min_silence_duration_ms: payload.min_silence_duration_ms,
            stream: payload.stream.unwrap_or(false),
        });
    }

    if content_type.starts_with("multipart/form-data") {
        let mut multipart = req
            .extract::<Multipart, _>()
            .await
            .map_err(|e| ApiError::bad_request(format!("Invalid multipart payload: {e}")))?;

        let mut out = DiarizationRequest::default();

        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|e| ApiError::bad_request(format!("Failed reading multipart field: {e}")))?
        {
            let name = field.name().unwrap_or_default().to_string();
            match name.as_str() {
                "file" | "audio" => {
                    let bytes = field.bytes().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart '{}' field: {}",
                            name, e
                        ))
                    })?;
                    if !bytes.is_empty() {
                        out.audio_base64 =
                            Some(base64::engine::general_purpose::STANDARD.encode(&bytes));
                    }
                }
                "audio_base64" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'audio_base64' field: {}",
                            e
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.audio_base64 = Some(text.trim().to_string());
                    }
                }
                "model" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'model' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.model = Some(text.trim().to_string());
                    }
                }
                "response_format" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'response_format' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.response_format = Some(text.trim().to_string());
                    }
                }
                "min_speakers" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'min_speakers' field: {e}"
                        ))
                    })?;
                    out.min_speakers = text.trim().parse::<usize>().ok();
                }
                "max_speakers" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'max_speakers' field: {e}"
                        ))
                    })?;
                    out.max_speakers = text.trim().parse::<usize>().ok();
                }
                "min_speech_duration_ms" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'min_speech_duration_ms' field: {e}"
                        ))
                    })?;
                    out.min_speech_duration_ms = text.trim().parse::<f32>().ok();
                }
                "min_silence_duration_ms" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'min_silence_duration_ms' field: {e}"
                        ))
                    })?;
                    out.min_silence_duration_ms = text.trim().parse::<f32>().ok();
                }
                "stream" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'stream' field: {e}"
                        ))
                    })?;
                    out.stream = matches!(
                        text.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    );
                }
                _ => {}
            }
        }

        return Ok(out);
    }

    Err(ApiError {
        status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
        message: "Expected `Content-Type: application/json` or `multipart/form-data`".to_string(),
    })
}
