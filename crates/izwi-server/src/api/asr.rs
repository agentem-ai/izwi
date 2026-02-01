//! Qwen3-ASR API endpoints for speech-to-text transcription

use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    Json,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{info, warn};

use crate::error::ApiError;
use crate::state::AppState;

const ASR_SOCKET_PATH: &str = "/tmp/izwi_qwen3_asr_daemon.sock";

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

/// Send a message to the ASR daemon via Unix socket
fn send_daemon_message(message: &serde_json::Value) -> Result<serde_json::Value, ApiError> {
    let mut stream = UnixStream::connect(ASR_SOCKET_PATH)
        .map_err(|e| ApiError::internal(format!("ASR daemon not running: {}", e)))?;

    stream.set_read_timeout(Some(Duration::from_secs(120))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(30))).ok();

    let msg_bytes = serde_json::to_vec(message)
        .map_err(|e| ApiError::internal(format!("Failed to serialize message: {}", e)))?;

    let length = (msg_bytes.len() as u32).to_be_bytes();
    stream
        .write_all(&length)
        .map_err(|e| ApiError::internal(format!("Failed to write length: {}", e)))?;
    stream
        .write_all(&msg_bytes)
        .map_err(|e| ApiError::internal(format!("Failed to write message: {}", e)))?;

    let mut length_buf = [0u8; 4];
    stream
        .read_exact(&mut length_buf)
        .map_err(|e| ApiError::internal(format!("Failed to read response length: {}", e)))?;
    let response_length = u32::from_be_bytes(length_buf) as usize;

    let mut response_buf = vec![0u8; response_length];
    stream
        .read_exact(&mut response_buf)
        .map_err(|e| ApiError::internal(format!("Failed to read response: {}", e)))?;

    serde_json::from_slice(&response_buf)
        .map_err(|e| ApiError::internal(format!("Failed to parse response: {}", e)))
}

/// Check if the ASR daemon is running
fn is_daemon_running() -> bool {
    Path::new(ASR_SOCKET_PATH).exists() && UnixStream::connect(ASR_SOCKET_PATH).is_ok()
}

/// Get ASR daemon status
pub async fn status(State(_state): State<AppState>) -> Result<Json<AsrStatusResponse>, ApiError> {
    if !is_daemon_running() {
        return Ok(Json(AsrStatusResponse {
            running: false,
            status: "stopped".to_string(),
            device: None,
            cached_models: vec![],
        }));
    }

    let message = serde_json::json!({
        "command": "status"
    });

    match send_daemon_message(&message) {
        Ok(response) => {
            let device = response
                .get("device")
                .and_then(|v| v.as_str())
                .map(String::from);
            let cached_models = response
                .get("cached_models")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            Ok(Json(AsrStatusResponse {
                running: true,
                status: "running".to_string(),
                device,
                cached_models,
            }))
        }
        Err(_) => Ok(Json(AsrStatusResponse {
            running: false,
            status: "error".to_string(),
            device: None,
            cached_models: vec![],
        })),
    }
}

/// Start the ASR daemon
pub async fn start_daemon(
    State(_state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if is_daemon_running() {
        return Ok(Json(serde_json::json!({
            "success": true,
            "message": "ASR daemon already running"
        })));
    }

    info!("Starting Qwen3-ASR daemon");

    let scripts_dir = std::env::current_dir().unwrap_or_default().join("scripts");
    let daemon_script = scripts_dir.join("qwen3_asr_daemon.py");

    if !daemon_script.exists() {
        return Err(ApiError::internal(format!(
            "ASR daemon script not found: {:?}",
            daemon_script
        )));
    }

    let _child = Command::new("python3")
        .arg(&daemon_script)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| ApiError::internal(format!("Failed to start ASR daemon: {}", e)))?;

    // Wait for daemon to start
    for _ in 0..50 {
        std::thread::sleep(Duration::from_millis(100));
        if is_daemon_running() {
            return Ok(Json(serde_json::json!({
                "success": true,
                "message": "ASR daemon started"
            })));
        }
    }

    Err(ApiError::internal(
        "ASR daemon failed to start within timeout",
    ))
}

/// Stop the ASR daemon
pub async fn stop_daemon(
    State(_state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if !is_daemon_running() {
        return Ok(Json(serde_json::json!({
            "success": true,
            "message": "ASR daemon not running"
        })));
    }

    let message = serde_json::json!({
        "command": "shutdown"
    });

    match send_daemon_message(&message) {
        Ok(_) => Ok(Json(serde_json::json!({
            "success": true,
            "message": "ASR daemon stopped"
        }))),
        Err(_) => {
            warn!("Error stopping ASR daemon");
            Ok(Json(serde_json::json!({
                "success": true,
                "message": "ASR daemon stop requested"
            })))
        }
    }
}

/// Streaming transcription event types
#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TranscribeStreamEvent {
    Start {
        audio_duration_secs: Option<f64>,
    },
    Partial {
        text: String,
        is_final: bool,
    },
    Final {
        text: String,
        language: Option<String>,
        audio_duration_secs: Option<f64>,
    },
    Error {
        error: String,
    },
    Done,
}

/// Stream transcription with SSE - sends partial results as text is decoded
pub async fn transcribe_stream(
    State(_state): State<AppState>,
    Json(request): Json<TranscribeRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    if !is_daemon_running() {
        return Err(ApiError::internal(
            "ASR daemon not running. Please start it first.",
        ));
    }

    // Create an async stream that reads from the daemon using tokio async I/O
    let stream = async_stream::stream! {
        // Connect to daemon using tokio's async UnixStream
        let stream_result = tokio::net::UnixStream::connect(ASR_SOCKET_PATH).await;
        let mut daemon_stream = match stream_result {
            Ok(s) => s,
            Err(e) => {
                let event = TranscribeStreamEvent::Error {
                    error: format!("Failed to connect to ASR daemon: {}", e),
                };
                yield Ok(Event::default().json_data(event).unwrap());
                yield Ok(Event::default().json_data(TranscribeStreamEvent::Done).unwrap());
                return;
            }
        };

        // Send streaming transcription request
        let message = serde_json::json!({
            "command": "transcribe_stream",
            "audio_base64": request.audio_base64,
            "model_id": request.model_id,
            "language": request.language,
        });

        let msg_bytes = match serde_json::to_vec(&message) {
            Ok(b) => b,
            Err(e) => {
                let event = TranscribeStreamEvent::Error {
                    error: format!("Failed to serialize request: {}", e),
                };
                yield Ok(Event::default().json_data(event).unwrap());
                yield Ok(Event::default().json_data(TranscribeStreamEvent::Done).unwrap());
                return;
            }
        };

        let length = (msg_bytes.len() as u32).to_be_bytes();
        if daemon_stream.write_all(&length).await.is_err()
            || daemon_stream.write_all(&msg_bytes).await.is_err()
            || daemon_stream.flush().await.is_err()
        {
            let event = TranscribeStreamEvent::Error {
                error: "Failed to send request to daemon".to_string(),
            };
            yield Ok(Event::default().json_data(event).unwrap());
            yield Ok(Event::default().json_data(TranscribeStreamEvent::Done).unwrap());
            return;
        }

        // Read streaming responses using async I/O
        // This properly yields to the async runtime between reads
        loop {
            let mut length_buf = [0u8; 4];
            if daemon_stream.read_exact(&mut length_buf).await.is_err() {
                break;
            }
            let response_length = u32::from_be_bytes(length_buf) as usize;

            let mut response_buf = vec![0u8; response_length];
            if daemon_stream.read_exact(&mut response_buf).await.is_err() {
                break;
            }

            let response: serde_json::Value = match serde_json::from_slice(&response_buf) {
                Ok(r) => r,
                Err(_) => break,
            };

            let event_type = response.get("event").and_then(|v| v.as_str()).unwrap_or("");

            let sse_event = match event_type {
                "start" => {
                    let audio_duration_secs = response.get("audio_duration_secs").and_then(|v| v.as_f64());
                    TranscribeStreamEvent::Start { audio_duration_secs }
                }
                "partial" => {
                    let text = response.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let is_final = response.get("is_final").and_then(|v| v.as_bool()).unwrap_or(false);
                    TranscribeStreamEvent::Partial { text, is_final }
                }
                "final" => {
                    let text = response.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let language = response.get("language").and_then(|v| v.as_str()).map(String::from);
                    let audio_duration_secs = response.get("audio_duration_secs").and_then(|v| v.as_f64());
                    TranscribeStreamEvent::Final { text, language, audio_duration_secs }
                }
                "error" => {
                    let error = response.get("error").and_then(|v| v.as_str()).unwrap_or("Unknown error").to_string();
                    TranscribeStreamEvent::Error { error }
                }
                "done" => {
                    yield Ok(Event::default().json_data(TranscribeStreamEvent::Done).unwrap());
                    break;
                }
                _ => continue,
            };

            yield Ok(Event::default().json_data(sse_event).unwrap());
        }
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// Transcribe audio to text
pub async fn transcribe(
    State(_state): State<AppState>,
    Json(request): Json<TranscribeRequest>,
) -> Result<Json<TranscribeResponse>, ApiError> {
    use std::time::Instant;

    if !is_daemon_running() {
        return Err(ApiError::internal(
            "ASR daemon not running. Please start it first.",
        ));
    }

    let start_time = Instant::now();

    let message = serde_json::json!({
        "command": "transcribe",
        "audio_base64": request.audio_base64,
        "model_id": request.model_id,
        "language": request.language,
    });

    let response = send_daemon_message(&message)?;

    let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    if let Some(error) = response.get("error").and_then(|v| v.as_str()) {
        return Err(ApiError::internal(error.to_string()));
    }

    let transcription = response
        .get("transcription")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let language = response
        .get("language")
        .and_then(|v| v.as_str())
        .map(String::from);

    // Extract audio duration from daemon response if available
    let audio_duration_secs = response.get("audio_duration_secs").and_then(|v| v.as_f64());

    // Calculate RTF if we have audio duration
    let rtf = audio_duration_secs.map(|dur| {
        if dur > 0.0 {
            (processing_time_ms / 1000.0) / dur
        } else {
            0.0
        }
    });

    Ok(Json(TranscribeResponse {
        transcription,
        language,
        stats: Some(AsrStats {
            processing_time_ms,
            audio_duration_secs,
            rtf,
        }),
    }))
}
