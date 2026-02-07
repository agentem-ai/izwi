//! OpenAI-compatible chat completions endpoints.

use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::models::qwen3_chat::ChatMessage;
use izwi_core::{parse_chat_model_variant, ModelVariant};

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct OpenAiChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChoice>,
    usage: OpenAiUsage,
}

#[derive(Debug, Serialize)]
struct OpenAiChoice {
    index: usize,
    message: OpenAiAssistantMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct OpenAiAssistantMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenAiUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct OpenAiChatChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChunkChoice>,
}

#[derive(Debug, Serialize)]
struct OpenAiChunkChoice {
    index: usize,
    delta: OpenAiDelta,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct OpenAiDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

fn max_new_tokens(value: Option<usize>) -> usize {
    value.unwrap_or(1536).clamp(1, 4096)
}

fn parse_chat_model(model_id: &str) -> Result<ModelVariant, ApiError> {
    parse_chat_model_variant(Some(model_id)).map_err(|err| ApiError::bad_request(err.to_string()))
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::bad_request(
            "Chat request must include at least one message",
        ));
    }

    if req.stream.unwrap_or(false) {
        let stream_response = complete_stream(state, req).await?;
        return Ok(stream_response.into_response());
    }

    let variant = parse_chat_model(&req.model)?;
    let _permit = state.acquire_permit().await;

    let generation = state
        .engine
        .chat_generate(variant, req.messages, max_new_tokens(req.max_tokens))
        .await?;

    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = now_unix_secs();
    let completion_tokens = generation.tokens_generated;
    let prompt_tokens = 0usize;

    let response = OpenAiChatCompletionResponse {
        id: completion_id,
        object: "chat.completion",
        created,
        model: variant.dir_name().to_string(),
        choices: vec![OpenAiChoice {
            index: 0,
            message: OpenAiAssistantMessage {
                role: "assistant",
                content: generation.text,
            },
            finish_reason: "stop",
        }],
        usage: OpenAiUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    Ok(Json(response).into_response())
}

async fn complete_stream(
    state: AppState,
    req: ChatCompletionRequest,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let variant = parse_chat_model(&req.model)?;
    let max_tokens = max_new_tokens(req.max_tokens);
    let messages = req.messages;
    let model_id = variant.dir_name().to_string();
    let timeout = Duration::from_secs(state.request_timeout_secs);

    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = now_unix_secs();

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<String>();
    let engine = state.engine.clone();
    let semaphore = state.request_semaphore.clone();
    let completion_id_for_task = completion_id.clone();
    let model_id_for_task = model_id.clone();

    tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                let _ = event_tx.send(
                    serde_json::json!({
                        "error": {
                            "message": "Server is shutting down",
                            "type": "server_error"
                        }
                    })
                    .to_string(),
                );
                let _ = event_tx.send("[DONE]".to_string());
                return;
            }
        };

        let start_chunk = OpenAiChatChunk {
            id: completion_id_for_task.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_id_for_task.clone(),
            choices: vec![OpenAiChunkChoice {
                index: 0,
                delta: OpenAiDelta {
                    role: Some("assistant"),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        let _ = event_tx.send(serde_json::to_string(&start_chunk).unwrap_or_default());

        let delta_tx = event_tx.clone();
        let result = tokio::time::timeout(timeout, async {
            engine
                .chat_generate_streaming(variant, messages, max_tokens, move |delta| {
                    let chunk = OpenAiChatChunk {
                        id: completion_id_for_task.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_id_for_task.clone(),
                        choices: vec![OpenAiChunkChoice {
                            index: 0,
                            delta: OpenAiDelta {
                                role: None,
                                content: Some(delta),
                            },
                            finish_reason: None,
                        }],
                    };
                    let _ = delta_tx.send(serde_json::to_string(&chunk).unwrap_or_default());
                })
                .await
        })
        .await;

        match result {
            Ok(Ok(_generation)) => {
                let final_chunk = OpenAiChatChunk {
                    id: completion_id,
                    object: "chat.completion.chunk",
                    created,
                    model: model_id,
                    choices: vec![OpenAiChunkChoice {
                        index: 0,
                        delta: OpenAiDelta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some("stop"),
                    }],
                };
                let _ = event_tx.send(serde_json::to_string(&final_chunk).unwrap_or_default());
            }
            Ok(Err(err)) => {
                let _ = event_tx.send(
                    serde_json::json!({
                        "error": {
                            "message": err.to_string(),
                            "type": "server_error"
                        }
                    })
                    .to_string(),
                );
            }
            Err(_) => {
                let _ = event_tx.send(
                    serde_json::json!({
                        "error": {
                            "message": "Chat request timed out",
                            "type": "timeout_error"
                        }
                    })
                    .to_string(),
                );
            }
        }

        let _ = event_tx.send("[DONE]".to_string());
    });

    let stream = async_stream::stream! {
        while let Some(event) = event_rx.recv().await {
            yield Ok(Event::default().data(event.clone()));
            if event == "[DONE]" {
                break;
            }
        }
    };

    Ok(Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default()))
}
