use std::convert::Infallible;
use std::time::Duration;

use axum::{
    body::Body,
    extract::{Extension, Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::api::request_context::RequestContext;
use crate::chat_store::{ChatThreadMessage, ChatThreadSummary};
use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::models::chat_types::{ChatMessage, ChatRole};
use izwi_core::{parse_chat_model_variant, ModelVariant};

#[derive(Debug, Serialize)]
pub struct ChatThreadListResponse {
    pub threads: Vec<ChatThreadSummary>,
}

#[derive(Debug, Deserialize)]
pub struct CreateChatThreadRequest {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatThreadDetailResponse {
    pub thread: ChatThreadSummary,
    pub messages: Vec<ChatThreadMessage>,
}

#[derive(Debug, Serialize)]
pub struct DeleteChatThreadResponse {
    pub id: String,
    pub deleted: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct CreateThreadMessageRequest {
    pub model: String,
    pub content: String,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatGenerationStats {
    pub tokens_generated: usize,
    pub generation_time_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateThreadMessageResponse {
    pub thread_id: String,
    pub model_id: String,
    pub user_message: ChatThreadMessage,
    pub assistant_message: ChatThreadMessage,
    pub stats: ChatGenerationStats,
}

#[derive(Debug, Serialize)]
struct ThreadStreamStartEvent {
    event: &'static str,
    thread_id: String,
    model_id: String,
    user_message: ChatThreadMessage,
}

#[derive(Debug, Serialize)]
struct ThreadStreamDeltaEvent {
    event: &'static str,
    delta: String,
}

#[derive(Debug, Serialize)]
struct ThreadStreamDoneEvent {
    event: &'static str,
    thread_id: String,
    model_id: String,
    assistant_message: ChatThreadMessage,
    stats: ChatGenerationStats,
}

#[derive(Debug, Serialize)]
struct ThreadStreamErrorEvent {
    event: &'static str,
    error: String,
}

pub async fn list_threads(
    State(state): State<AppState>,
) -> Result<Json<ChatThreadListResponse>, ApiError> {
    let threads = state
        .chat_store
        .list_threads()
        .await
        .map_err(map_store_error)?;

    Ok(Json(ChatThreadListResponse { threads }))
}

pub async fn create_thread(
    State(state): State<AppState>,
    Json(req): Json<CreateChatThreadRequest>,
) -> Result<Json<ChatThreadSummary>, ApiError> {
    let thread = state
        .chat_store
        .create_thread(req.title, req.model_id)
        .await
        .map_err(map_store_error)?;

    Ok(Json(thread))
}

pub async fn get_thread(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
) -> Result<Json<ChatThreadDetailResponse>, ApiError> {
    let thread = get_thread_or_not_found(&state, &thread_id).await?;
    let messages = state
        .chat_store
        .list_messages(thread_id)
        .await
        .map_err(map_store_error)?;

    Ok(Json(ChatThreadDetailResponse { thread, messages }))
}

pub async fn list_thread_messages(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
) -> Result<Json<Vec<ChatThreadMessage>>, ApiError> {
    get_thread_or_not_found(&state, &thread_id).await?;
    let messages = state
        .chat_store
        .list_messages(thread_id)
        .await
        .map_err(map_store_error)?;

    Ok(Json(messages))
}

pub async fn delete_thread(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
) -> Result<Json<DeleteChatThreadResponse>, ApiError> {
    let deleted = state
        .chat_store
        .delete_thread(thread_id.clone())
        .await
        .map_err(map_store_error)?;

    if !deleted {
        return Err(ApiError::not_found("Thread not found"));
    }

    Ok(Json(DeleteChatThreadResponse {
        id: thread_id,
        deleted: true,
    }))
}

pub async fn create_thread_message(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
    Extension(ctx): Extension<RequestContext>,
    Json(req): Json<CreateThreadMessageRequest>,
) -> Result<Response, ApiError> {
    let user_content = req.content.trim().to_string();
    if user_content.is_empty() {
        return Err(ApiError::bad_request("Message content cannot be empty"));
    }

    get_thread_or_not_found(&state, &thread_id).await?;
    let existing_messages = state
        .chat_store
        .list_messages(thread_id.clone())
        .await
        .map_err(map_store_error)?;

    let model_variant = parse_chat_model(&req.model)?;
    let model_id = model_variant.dir_name().to_string();

    let user_message = state
        .chat_store
        .append_message(
            thread_id.clone(),
            "user".to_string(),
            user_content.clone(),
            Some(model_id.clone()),
            None,
            None,
        )
        .await
        .map_err(map_store_or_not_found)?;

    let runtime_messages = build_runtime_messages(
        &existing_messages,
        &user_content,
        req.system_prompt.as_deref(),
    )?;

    if req.stream.unwrap_or(false) {
        return create_streaming_thread_message(
            state,
            req,
            model_variant,
            model_id,
            thread_id,
            user_message,
            runtime_messages,
            ctx.correlation_id,
        )
        .await;
    }

    let _permit = state.acquire_permit().await;
    let generation = state
        .runtime
        .chat_generate_with_correlation(
            model_variant,
            runtime_messages,
            max_new_tokens(model_variant, req.max_completion_tokens, req.max_tokens),
            Some(&ctx.correlation_id),
        )
        .await?;

    let assistant_message = state
        .chat_store
        .append_message(
            thread_id.clone(),
            "assistant".to_string(),
            generation.text.clone(),
            Some(model_id.clone()),
            Some(generation.tokens_generated),
            Some(generation.generation_time_ms),
        )
        .await
        .map_err(map_store_or_not_found)?;

    let response = CreateThreadMessageResponse {
        thread_id,
        model_id,
        user_message,
        assistant_message,
        stats: ChatGenerationStats {
            tokens_generated: generation.tokens_generated,
            generation_time_ms: generation.generation_time_ms,
        },
    };

    Ok(Json(response).into_response())
}

async fn create_streaming_thread_message(
    state: AppState,
    req: CreateThreadMessageRequest,
    model_variant: ModelVariant,
    model_id: String,
    thread_id: String,
    user_message: ChatThreadMessage,
    runtime_messages: Vec<ChatMessage>,
    correlation_id: String,
) -> Result<Response, ApiError> {
    let timeout = Duration::from_secs(state.request_timeout_secs);
    let max_tokens = max_new_tokens(model_variant, req.max_completion_tokens, req.max_tokens);
    let semaphore = state.request_semaphore.clone();
    let runtime = state.runtime.clone();
    let chat_store = state.chat_store.clone();

    let thread_id_for_task = thread_id.clone();
    let model_id_for_task = model_id.clone();
    let user_message_for_start = user_message.clone();

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<String>();

    tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                let _ = event_tx.send(
                    serde_json::to_string(&ThreadStreamErrorEvent {
                        event: "error",
                        error: "Server is shutting down".to_string(),
                    })
                    .unwrap_or_default(),
                );
                let _ = event_tx.send("[DONE]".to_string());
                return;
            }
        };

        let _ = event_tx.send(
            serde_json::to_string(&ThreadStreamStartEvent {
                event: "start",
                thread_id: thread_id_for_task.clone(),
                model_id: model_id_for_task.clone(),
                user_message: user_message_for_start,
            })
            .unwrap_or_default(),
        );

        let full_text = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let full_text_for_delta = full_text.clone();
        let delta_tx = event_tx.clone();

        let generation_result = tokio::time::timeout(timeout, async {
            runtime
                .chat_generate_streaming_with_correlation(
                    model_variant,
                    runtime_messages,
                    max_tokens,
                    Some(correlation_id.as_str()),
                    move |delta| {
                        if let Ok(mut output_text) = full_text_for_delta.lock() {
                            output_text.push_str(&delta);
                        }
                        let payload = ThreadStreamDeltaEvent {
                            event: "delta",
                            delta,
                        };
                        let _ = delta_tx.send(serde_json::to_string(&payload).unwrap_or_default());
                    },
                )
                .await
        })
        .await;

        match generation_result {
            Ok(Ok(generation)) => {
                let assistant_text = full_text.lock().map(|s| s.clone()).unwrap_or_default();
                match chat_store
                    .append_message(
                        thread_id_for_task.clone(),
                        "assistant".to_string(),
                        assistant_text,
                        Some(model_id_for_task.clone()),
                        Some(generation.tokens_generated),
                        Some(generation.generation_time_ms),
                    )
                    .await
                {
                    Ok(assistant_message) => {
                        let done_event = ThreadStreamDoneEvent {
                            event: "done",
                            thread_id: thread_id_for_task,
                            model_id: model_id_for_task,
                            assistant_message,
                            stats: ChatGenerationStats {
                                tokens_generated: generation.tokens_generated,
                                generation_time_ms: generation.generation_time_ms,
                            },
                        };
                        let _ =
                            event_tx.send(serde_json::to_string(&done_event).unwrap_or_default());
                    }
                    Err(err) => {
                        let _ = event_tx.send(
                            serde_json::to_string(&ThreadStreamErrorEvent {
                                event: "error",
                                error: format!("Failed to persist assistant message: {err}"),
                            })
                            .unwrap_or_default(),
                        );
                    }
                }
            }
            Ok(Err(err)) => {
                let _ = event_tx.send(
                    serde_json::to_string(&ThreadStreamErrorEvent {
                        event: "error",
                        error: err.to_string(),
                    })
                    .unwrap_or_default(),
                );
            }
            Err(_) => {
                let _ = event_tx.send(
                    serde_json::to_string(&ThreadStreamErrorEvent {
                        event: "error",
                        error: "Chat request timed out".to_string(),
                    })
                    .unwrap_or_default(),
                );
            }
        }

        let _ = event_tx.send("[DONE]".to_string());
    });

    let stream = async_stream::stream! {
        while let Some(payload) = event_rx.recv().await {
            yield Ok::<_, Infallible>(format!("data: {payload}\n\n"));
            if payload == "[DONE]" {
                break;
            }
        }
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap())
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

fn build_runtime_messages(
    existing: &[ChatThreadMessage],
    new_user_content: &str,
    system_prompt: Option<&str>,
) -> Result<Vec<ChatMessage>, ApiError> {
    let mut messages = Vec::new();

    if let Some(prompt) = system_prompt
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
    {
        messages.push(ChatMessage {
            role: ChatRole::System,
            content: prompt.to_string(),
        });
    }

    for message in existing {
        let role = parse_stored_role(&message.role)?;
        messages.push(ChatMessage {
            role,
            content: message.content.clone(),
        });
    }

    messages.push(ChatMessage {
        role: ChatRole::User,
        content: new_user_content.to_string(),
    });

    Ok(messages)
}

fn parse_stored_role(role: &str) -> Result<ChatRole, ApiError> {
    match role {
        "system" => Ok(ChatRole::System),
        "user" => Ok(ChatRole::User),
        "assistant" => Ok(ChatRole::Assistant),
        other => Err(ApiError::internal(format!(
            "Invalid stored chat role: {other}"
        ))),
    }
}

async fn get_thread_or_not_found(
    state: &AppState,
    thread_id: &str,
) -> Result<ChatThreadSummary, ApiError> {
    let thread = state
        .chat_store
        .get_thread(thread_id.to_string())
        .await
        .map_err(map_store_error)?;

    thread.ok_or_else(|| ApiError::not_found("Thread not found"))
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Chat storage error: {err}"))
}

fn map_store_or_not_found(err: anyhow::Error) -> ApiError {
    let error_text = err.to_string();
    if error_text.contains("Thread not found") {
        ApiError::not_found("Thread not found")
    } else {
        map_store_error(err)
    }
}
