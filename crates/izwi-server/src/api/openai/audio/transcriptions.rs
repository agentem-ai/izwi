//! OpenAI-compatible transcription endpoints.

use axum::{
    body::Body,
    extract::{Extension, Multipart, Request, State},
    http::{header, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json, RequestExt,
};
use base64::Engine;
use std::convert::Infallible;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::info;

use crate::api::transcription::support::{
    build_segments, build_words, format_srt, format_vtt,
};
use crate::api::request_context::RequestContext;
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
    words: Vec<VerboseJsonWordResponse>,
    segments: Vec<VerboseJsonSegmentResponse>,
}

#[derive(Debug, serde::Serialize)]
struct VerboseJsonWordResponse {
    word: String,
    start: f32,
    end: f32,
}

#[derive(Debug, serde::Serialize)]
struct VerboseJsonSegmentResponse {
    start: f32,
    end: f32,
    text: String,
    word_start: usize,
    word_end: usize,
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
    Extension(ctx): Extension<RequestContext>,
    req: Request,
) -> Result<Response<Body>, ApiError> {
    let mut req = parse_transcription_request(req).await?;
    let audio_base64 = req
        .audio_base64
        .take()
        .ok_or_else(|| ApiError::bad_request("Missing audio input (`file` or `audio_base64`)"))?;

    info!("OpenAI transcription request: {} bytes", audio_base64.len());

    if req.stream {
        return transcriptions_stream(state, req, audio_base64, ctx.correlation_id).await;
    }

    let _permit = state.acquire_permit().await;

    let started = Instant::now();
    let output = state
        .runtime
        .asr_transcribe_streaming_with_correlation(
            &audio_base64,
            req.model.as_deref(),
            req.language.as_deref(),
            Some(&ctx.correlation_id),
            |_delta| {},
        )
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
    let resolved_language = output.language.clone().or(req.language.clone());
    let needs_timestamps = matches!(response_format.as_str(), "verbose_json" | "srt" | "vtt");
    let (words, segments) = if needs_timestamps {
        build_alignment_details(
            &state,
            &audio_base64,
            output.text.as_str(),
            resolved_language.as_deref(),
            output.duration_secs,
        )
        .await
    } else {
        (Vec::new(), Vec::new())
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
                    language: resolved_language,
                    duration: output.duration_secs,
                    processing_time_ms: elapsed_ms,
                    rtf,
                    words: words
                        .iter()
                        .map(|word| VerboseJsonWordResponse {
                            word: word.word.clone(),
                            start: word.start_secs,
                            end: word.end_secs,
                        })
                        .collect(),
                    segments: segments
                        .iter()
                        .map(|segment| VerboseJsonSegmentResponse {
                            start: segment.start_secs,
                            end: segment.end_secs,
                            text: segment.text.clone(),
                            word_start: segment.word_start,
                            word_end: segment.word_end,
                        })
                        .collect(),
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
            .body(Body::from(format_srt(
                &segments,
                output.text.as_str(),
                output.duration_secs,
            )))
            .unwrap()),
        "vtt" => Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/vtt; charset=utf-8")
            .body(Body::from(format_vtt(
                &segments,
                output.text.as_str(),
                output.duration_secs,
            )))
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
    correlation_id: String,
) -> Result<Response<Body>, ApiError> {
    let timeout = Duration::from_secs(state.request_timeout_secs);
    let model = req.model;
    let language = req.language;

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<String>();
    let engine = state.runtime.clone();
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
                .asr_transcribe_streaming_with_correlation(
                    &audio_base64,
                    model.as_deref(),
                    language.as_deref(),
                    Some(correlation_id.as_str()),
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
            yield Ok::<_, Infallible>(Event::default().data(payload));
        }
    };

    let mut response = Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response();
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-cache"),
    );
    response.headers_mut().insert(
        header::CONNECTION,
        header::HeaderValue::from_static("keep-alive"),
    );
    response
        .headers_mut()
        .insert("x-accel-buffering", header::HeaderValue::from_static("no"));
    Ok(response)
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

async fn build_alignment_details(
    state: &AppState,
    audio_base64: &str,
    text: &str,
    language: Option<&str>,
    duration_secs: f32,
) -> (
    Vec<crate::transcription_store::TranscriptionWordRecord>,
    Vec<crate::transcription_store::TranscriptionSegmentRecord>,
) {
    if text.trim().is_empty() {
        return (Vec::new(), Vec::new());
    }

    let words = match state
        .runtime
        .force_align_with_model_and_language(
            audio_base64,
            text,
            language,
            None,
        )
        .await
    {
        Ok(aligned_words) => build_words(&aligned_words),
        Err(_) => Vec::new(),
    };
    let segments = build_segments(&words, text, duration_secs);
    (words, segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_srt_and_vtt() {
        let segments = vec![crate::transcription_store::TranscriptionSegmentRecord {
            start_secs: 0.0,
            end_secs: 1.23,
            text: "hello".to_string(),
            word_start: 0,
            word_end: 1,
        }];
        let srt = format_srt(&segments, "hello", 1.23);
        let vtt = format_vtt(&segments, "hello", 1.23);
        assert!(srt.contains("-->"));
        assert!(vtt.starts_with("WEBVTT"));
    }
}
