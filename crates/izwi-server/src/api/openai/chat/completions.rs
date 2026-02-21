//! OpenAI-compatible chat completions endpoints.

use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Extension, State},
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::api::request_context::RequestContext;
use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::{parse_chat_model_variant, ModelVariant};
use izwi_core::{ChatMessage, ChatRole};

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OpenAiInboundMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionStreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiInboundMessage {
    pub role: String,
    pub content: OpenAiInboundContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum OpenAiInboundContent {
    Text(String),
    Parts(Vec<OpenAiInboundContentPart>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiInboundContentPart {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub input_text: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAiChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChoice>,
    usage: OpenAiUsage,
    izwi_generation_time_ms: f64,
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

#[derive(Debug, Clone, Serialize)]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<OpenAiUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    izwi_generation_time_ms: Option<f64>,
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

fn max_new_tokens(
    variant: ModelVariant,
    max_completion_tokens: Option<usize>,
    max_tokens: Option<usize>,
) -> usize {
    let requested = max_completion_tokens.or(max_tokens);

    let default = match variant {
        ModelVariant::Gemma34BIt => 4096,
        ModelVariant::Gemma31BIt => 4096,
        _ => 1536,
    };

    requested.unwrap_or(default).clamp(1, 4096)
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

fn to_core_messages(messages: Vec<OpenAiInboundMessage>) -> Result<Vec<ChatMessage>, ApiError> {
    messages
        .into_iter()
        .map(|message| {
            let role = parse_role(&message.role)?;
            let content = flatten_content(message.content);
            if content.trim().is_empty() {
                return Err(ApiError::bad_request(
                    "Chat message content cannot be empty",
                ));
            }
            Ok(ChatMessage { role, content })
        })
        .collect()
}

fn parse_role(raw: &str) -> Result<ChatRole, ApiError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "system" => Ok(ChatRole::System),
        "user" => Ok(ChatRole::User),
        "assistant" => Ok(ChatRole::Assistant),
        other => Err(ApiError::bad_request(format!(
            "Unsupported chat message role: {}",
            other
        ))),
    }
}

fn flatten_content(content: OpenAiInboundContent) -> String {
    match content {
        OpenAiInboundContent::Text(text) => text,
        OpenAiInboundContent::Parts(parts) => parts
            .into_iter()
            .filter_map(|part| {
                if part.kind.as_deref() == Some("text") {
                    part.text.or(part.input_text)
                } else {
                    part.text.or(part.input_text)
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

pub async fn completions(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::bad_request(
            "This server currently supports only `n=1` for chat completions",
        ));
    }

    let messages = to_core_messages(req.messages.clone())?;
    if messages.is_empty() {
        return Err(ApiError::bad_request(
            "Chat request must include at least one message",
        ));
    }

    if req.stream.unwrap_or(false) {
        let stream_response = complete_stream(state, req, messages, ctx.correlation_id).await?;
        return Ok(stream_response.into_response());
    }

    let variant = parse_chat_model(&req.model)?;
    let _permit = state.acquire_permit().await;

    let generation = state
        .runtime
        .chat_generate_with_correlation(
            variant,
            messages,
            max_new_tokens(variant, req.max_completion_tokens, req.max_tokens),
            Some(&ctx.correlation_id),
        )
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
        izwi_generation_time_ms: generation.generation_time_ms,
    };

    Ok(Json(response).into_response())
}

async fn complete_stream(
    state: AppState,
    req: ChatCompletionRequest,
    messages: Vec<ChatMessage>,
    correlation_id: String,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let variant = parse_chat_model(&req.model)?;
    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|opts| opts.include_usage)
        .unwrap_or(false);
    let max_tokens = max_new_tokens(variant, req.max_completion_tokens, req.max_tokens);
    let model_id = variant.dir_name().to_string();
    let timeout = Duration::from_secs(state.request_timeout_secs);

    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = now_unix_secs();

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<String>();
    let engine = state.runtime.clone();
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
            usage: None,
            izwi_generation_time_ms: None,
        };
        let _ = event_tx.send(serde_json::to_string(&start_chunk).unwrap_or_default());

        let delta_tx = event_tx.clone();
        let result = tokio::time::timeout(timeout, async {
            engine
                .chat_generate_streaming_with_correlation(
                    variant,
                    messages,
                    max_tokens,
                    Some(correlation_id.as_str()),
                    move |delta| {
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
                            usage: None,
                            izwi_generation_time_ms: None,
                        };
                        let _ = delta_tx.send(serde_json::to_string(&chunk).unwrap_or_default());
                    },
                )
                .await
        })
        .await;

        match result {
            Ok(Ok(generation)) => {
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
                    usage: include_usage.then_some(OpenAiUsage {
                        prompt_tokens: 0,
                        completion_tokens: generation.tokens_generated,
                        total_tokens: generation.tokens_generated,
                    }),
                    izwi_generation_time_ms: Some(generation.generation_time_ms),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flattens_text_parts_content() {
        let flattened = flatten_content(OpenAiInboundContent::Parts(vec![
            OpenAiInboundContentPart {
                kind: Some("text".to_string()),
                text: Some("hello".to_string()),
                input_text: None,
            },
            OpenAiInboundContentPart {
                kind: Some("input_text".to_string()),
                text: None,
                input_text: Some("world".to_string()),
            },
        ]));

        assert_eq!(flattened, "hello\nworld");
    }
}
