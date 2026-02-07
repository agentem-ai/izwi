//! Text chat API endpoints.

use std::convert::Infallible;
use std::time::Duration;

use axum::{
    extract::State,
    response::{sse::Event, Sse},
    Json,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::models::qwen3_chat::ChatMessage;
use izwi_core::ModelVariant;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model_id: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub model_id: String,
    pub message: ChatMessageResponse,
    pub stats: ChatCompletionStats,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageResponse {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionStats {
    pub tokens_generated: usize,
    pub generation_time_ms: f64,
}

#[derive(Debug, Serialize)]
struct ChatStreamEvent {
    event: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stats: Option<ChatCompletionStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn max_new_tokens(value: Option<usize>) -> usize {
    value.unwrap_or(1536).clamp(1, 4096)
}

fn parse_chat_model(model_id: Option<&str>) -> Result<ModelVariant, ApiError> {
    match model_id.unwrap_or("Qwen3-0.6B-4bit") {
        "Qwen3-0.6B-4bit" => Ok(ModelVariant::Qwen306B4Bit),
        other => Err(ApiError::bad_request(format!(
            "Unsupported chat model: {other}. Supported: Qwen3-0.6B-4bit"
        ))),
    }
}

pub async fn complete(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::bad_request(
            "Chat request must include at least one message",
        ));
    }

    let variant = parse_chat_model(req.model_id.as_deref())?;
    let _permit = state.acquire_permit().await;

    let generation = state
        .engine
        .chat_generate(variant, req.messages, max_new_tokens(req.max_tokens))
        .await?;

    Ok(Json(ChatCompletionResponse {
        model_id: variant.dir_name().to_string(),
        message: ChatMessageResponse {
            role: "assistant",
            content: generation.text,
        },
        stats: ChatCompletionStats {
            tokens_generated: generation.tokens_generated,
            generation_time_ms: generation.generation_time_ms,
        },
    }))
}

pub async fn complete_stream(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::bad_request(
            "Chat request must include at least one message",
        ));
    }

    let variant = parse_chat_model(req.model_id.as_deref())?;
    let max_tokens = max_new_tokens(req.max_tokens);
    let messages = req.messages;
    let model_id = variant.dir_name().to_string();
    let timeout = Duration::from_secs(state.request_timeout_secs);

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<ChatStreamEvent>();
    let engine = state.engine.clone();
    let semaphore = state.request_semaphore.clone();

    tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                let _ = event_tx.send(ChatStreamEvent {
                    event: "error",
                    model_id: None,
                    delta: None,
                    message: None,
                    stats: None,
                    error: Some("Server is shutting down".to_string()),
                });
                return;
            }
        };

        let _ = event_tx.send(ChatStreamEvent {
            event: "start",
            model_id: Some(model_id.clone()),
            delta: None,
            message: None,
            stats: None,
            error: None,
        });

        let delta_tx = event_tx.clone();
        let result = tokio::time::timeout(timeout, async {
            engine
                .chat_generate_streaming(variant, messages, max_tokens, move |delta| {
                    let _ = delta_tx.send(ChatStreamEvent {
                        event: "delta",
                        model_id: None,
                        delta: Some(delta),
                        message: None,
                        stats: None,
                        error: None,
                    });
                })
                .await
        })
        .await;

        match result {
            Ok(Ok(generation)) => {
                let _ = event_tx.send(ChatStreamEvent {
                    event: "done",
                    model_id: Some(model_id),
                    delta: None,
                    message: Some(generation.text),
                    stats: Some(ChatCompletionStats {
                        tokens_generated: generation.tokens_generated,
                        generation_time_ms: generation.generation_time_ms,
                    }),
                    error: None,
                });
            }
            Ok(Err(err)) => {
                let _ = event_tx.send(ChatStreamEvent {
                    event: "error",
                    model_id: None,
                    delta: None,
                    message: None,
                    stats: None,
                    error: Some(err.to_string()),
                });
            }
            Err(_) => {
                let _ = event_tx.send(ChatStreamEvent {
                    event: "error",
                    model_id: None,
                    delta: None,
                    message: None,
                    stats: None,
                    error: Some("Chat request timed out".to_string()),
                });
            }
        }
    });

    let stream = async_stream::stream! {
        while let Some(event) = event_rx.recv().await {
            let payload = serde_json::to_string(&event).unwrap_or_default();
            let is_terminal = event.event == "done" || event.event == "error";
            yield Ok(Event::default().data(payload));
            if is_terminal {
                break;
            }
        }
    };

    Ok(Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default()))
}
