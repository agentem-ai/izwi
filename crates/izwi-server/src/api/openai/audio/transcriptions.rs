//! OpenAI-compatible transcription endpoints.

use axum::{
    body::Body,
    extract::{Multipart, Request, State},
    http::{header, StatusCode},
    response::Response,
    Json, RequestExt,
};
use base64::Engine;
use std::convert::Infallible;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::info;

use crate::error::ApiError;
use crate::state::AppState;

#[derive(Debug, Default)]
struct TranscriptionRequest {
    audio_base64: Option<String>,
    model: Option<String>,
    language: Option<String>,
    response_format: Option<String>,
    stream: bool,
    // Accepted for compatibility; currently not used by runtime.
    _prompt: Option<String>,
    _temperature: Option<f32>,
    _timestamp_granularities: Option<Vec<String>>,
}

#[derive(Debug, serde::Serialize)]
struct JsonTranscriptionResponse {
    text: String,
}

#[derive(Debug, serde::Serialize)]
struct VerboseJsonTranscriptionResponse {
    text: String,
    language: Option<String>,
    duration: f32,
    processing_time_ms: f64,
    rtf: Option<f64>,
}

#[derive(Debug, serde::Serialize)]
struct StreamEvent {
    event: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_duration_secs: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

pub async fn transcriptions(
    State(state): State<AppState>,
    req: Request,
) -> Result<Response<Body>, ApiError> {
    let mut req = parse_transcription_request(req).await?;
    let audio_base64 = req
        .audio_base64
        .take()
        .ok_or_else(|| ApiError::bad_request("Missing audio input (`file` or `audio_base64`)"))?;

    info!("OpenAI transcription request: {} bytes", audio_base64.len());

    if req.stream {
        return transcriptions_stream(state, req, audio_base64).await;
    }

    let _permit = state.acquire_permit().await;

    let started = Instant::now();
    let output = state
        .engine
        .asr_transcribe(&audio_base64, req.model.as_deref(), req.language.as_deref())
        .await?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    let response_format = req
        .response_format
        .as_deref()
        .unwrap_or("json")
        .to_ascii_lowercase();

    let rtf = if output.duration_secs > 0.0 {
        Some((elapsed_ms / 1000.0) / output.duration_secs as f64)
    } else {
        None
    };

    match response_format.as_str() {
        "json" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_string(&JsonTranscriptionResponse { text: output.text }).unwrap(),
            ))
            .unwrap()),
        "verbose_json" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_string(&VerboseJsonTranscriptionResponse {
                    text: output.text,
                    language: output.language,
                    duration: output.duration_secs,
                    processing_time_ms: elapsed_ms,
                    rtf,
                })
                .unwrap(),
            ))
            .unwrap()),
        "text" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .body(Body::from(output.text))
            .unwrap()),
        "srt" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .body(Body::from(format_srt(&output.text, output.duration_secs)))
            .unwrap()),
        "vtt" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/vtt; charset=utf-8")
            .body(Body::from(format_vtt(&output.text, output.duration_secs)))
            .unwrap()),
        other => Err(ApiError::bad_request(format!(
            "Unsupported response_format: {}. Supported: json, verbose_json, text, srt, vtt",
            other
        ))),
    }
}

async fn transcriptions_stream(
    state: AppState,
    req: TranscriptionRequest,
    audio_base64: String,
) -> Result<Response<Body>, ApiError> {
    let timeout = Duration::from_secs(state.request_timeout_secs);
    let model = req.model;
    let language = req.language;

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<String>();
    let engine = state.engine.clone();
    let semaphore = state.request_semaphore.clone();

    tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                let err = StreamEvent {
                    event: "error",
                    text: None,
                    delta: None,
                    language: None,
                    audio_duration_secs: None,
                    error: Some("Server is shutting down".to_string()),
                };
                let _ = event_tx.send(serde_json::to_string(&err).unwrap_or_default());

                let done = StreamEvent {
                    event: "done",
                    text: None,
                    delta: None,
                    language: None,
                    audio_duration_secs: None,
                    error: None,
                };
                let _ = event_tx.send(serde_json::to_string(&done).unwrap_or_default());
                return;
            }
        };

        let start = StreamEvent {
            event: "start",
            text: None,
            delta: None,
            language: None,
            audio_duration_secs: None,
            error: None,
        };
        let _ = event_tx.send(serde_json::to_string(&start).unwrap_or_default());

        let delta_tx = event_tx.clone();
        let result = tokio::time::timeout(timeout, async {
            engine
                .asr_transcribe_streaming(
                    &audio_base64,
                    model.as_deref(),
                    language.as_deref(),
                    move |delta| {
                        let event = StreamEvent {
                            event: "delta",
                            text: None,
                            delta: Some(delta),
                            language: None,
                            audio_duration_secs: None,
                            error: None,
                        };
                        let _ = delta_tx.send(serde_json::to_string(&event).unwrap_or_default());
                    },
                )
                .await
        })
        .await;

        match result {
            Ok(Ok(output)) => {
                let final_event = StreamEvent {
                    event: "final",
                    text: Some(output.text),
                    delta: None,
                    language: output.language,
                    audio_duration_secs: Some(output.duration_secs),
                    error: None,
                };
                let _ = event_tx.send(serde_json::to_string(&final_event).unwrap_or_default());
            }
            Ok(Err(err)) => {
                let error_event = StreamEvent {
                    event: "error",
                    text: None,
                    delta: None,
                    language: None,
                    audio_duration_secs: None,
                    error: Some(err.to_string()),
                };
                let _ = event_tx.send(serde_json::to_string(&error_event).unwrap_or_default());
            }
            Err(_) => {
                let error_event = StreamEvent {
                    event: "error",
                    text: None,
                    delta: None,
                    language: None,
                    audio_duration_secs: None,
                    error: Some("Transcription request timed out".to_string()),
                };
                let _ = event_tx.send(serde_json::to_string(&error_event).unwrap_or_default());
            }
        }

        let done = StreamEvent {
            event: "done",
            text: None,
            delta: None,
            language: None,
            audio_duration_secs: None,
            error: None,
        };
        let _ = event_tx.send(serde_json::to_string(&done).unwrap_or_default());
    });

    let stream = async_stream::stream! {
        while let Some(payload) = event_rx.recv().await {
            yield Ok::<_, Infallible>(format!("data: {payload}\n\n"));
        }
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap())
}

#[derive(Debug, serde::Deserialize)]
struct JsonRequestBody {
    audio_base64: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    timestamp_granularities: Option<Vec<String>>,
}

async fn parse_transcription_request(req: Request) -> Result<TranscriptionRequest, ApiError> {
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

        return Ok(TranscriptionRequest {
            audio_base64: Some(payload.audio_base64),
            model: payload.model,
            language: payload.language,
            response_format: payload.response_format,
            stream: payload.stream.unwrap_or(false),
            _prompt: payload.prompt,
            _temperature: payload.temperature,
            _timestamp_granularities: payload.timestamp_granularities,
        });
    }

    if content_type.starts_with("multipart/form-data") {
        let mut multipart = req
            .extract::<Multipart, _>()
            .await
            .map_err(|e| ApiError::bad_request(format!("Invalid multipart payload: {e}")))?;

        let mut out = TranscriptionRequest::default();

        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|e| ApiError::bad_request(format!("Failed reading multipart field: {e}")))?
        {
            let name = field.name().unwrap_or_default().to_string();
            match name.as_str() {
                "file" | "audio" => {
                    let bytes = field
                        .bytes()
                        .await
                        .map_err(|e| multipart_field_error(&name, &e.to_string()))?;
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
                        out.audio_base64 = Some(text);
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
                "language" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'language' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out.language = Some(text.trim().to_string());
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
                "prompt" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'prompt' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out._prompt = Some(text.trim().to_string());
                    }
                }
                "temperature" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'temperature' field: {e}"
                        ))
                    })?;
                    out._temperature = text.trim().parse::<f32>().ok();
                }
                "timestamp_granularities[]" => {
                    let text = field.text().await.map_err(|e| {
                        ApiError::bad_request(format!(
                            "Failed reading multipart 'timestamp_granularities[]' field: {e}"
                        ))
                    })?;
                    if !text.trim().is_empty() {
                        out._timestamp_granularities
                            .get_or_insert_with(Vec::new)
                            .push(text.trim().to_string());
                    }
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

fn multipart_field_error(field_name: &str, raw: &str) -> ApiError {
    let lowered = raw.to_ascii_lowercase();
    if lowered.contains("multipart/form-data") {
        return ApiError::bad_request(format!(
            "Failed reading multipart '{}' field: {}. \
This is commonly caused by oversized uploads or malformed multipart boundaries. \
Ensure `Content-Type` includes a valid boundary (let your HTTP client set it automatically for FormData) and keep payload under 64 MiB.",
            field_name, raw
        ));
    }

    ApiError::bad_request(format!(
        "Failed reading multipart '{}' field: {}",
        field_name, raw
    ))
}

fn format_srt(text: &str, duration_secs: f32) -> String {
    format!(
        "1\n{} --> {}\n{}\n",
        secs_to_srt(0.0),
        secs_to_srt(duration_secs.max(0.1)),
        text.trim()
    )
}

fn format_vtt(text: &str, duration_secs: f32) -> String {
    format!(
        "WEBVTT\n\n{} --> {}\n{}\n",
        secs_to_vtt(0.0),
        secs_to_vtt(duration_secs.max(0.1)),
        text.trim()
    )
}

fn secs_to_srt(secs: f32) -> String {
    let total_ms = (secs.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_sec = total_ms / 1000;
    let s = total_sec % 60;
    let total_min = total_sec / 60;
    let m = total_min % 60;
    let h = total_min / 60;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

fn secs_to_vtt(secs: f32) -> String {
    secs_to_srt(secs).replace(',', ".")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_srt_and_vtt() {
        let srt = format_srt("hello", 1.23);
        let vtt = format_vtt("hello", 1.23);
        assert!(srt.contains("-->"));
        assert!(vtt.starts_with("WEBVTT"));
    }
}
