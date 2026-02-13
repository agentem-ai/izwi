use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::Response,
    Json,
};
use tokio::sync::mpsc;

use crate::error::ApiError;
use crate::state::{AppState, StoredResponseInputItem, StoredResponseRecord};
use izwi_core::models::chat_types::{ChatMessage, ChatRole};
use izwi_core::parse_chat_model_variant;

use super::dto::{
    ResponseDeletedObject, ResponseError, ResponseInput, ResponseInputContent,
    ResponseInputItemContent, ResponseInputItemObject, ResponseInputItemsList, ResponseObject,
    ResponseOutputContent, ResponseOutputItem, ResponseStreamCompletedPayload,
    ResponseStreamCreatedPayload, ResponseStreamDeltaPayload, ResponseStreamEnvelope,
    ResponseUsage, ResponsesCreateRequest,
};

pub async fn create_response(
    State(state): State<AppState>,
    Json(req): Json<ResponsesCreateRequest>,
) -> Result<Response<Body>, ApiError> {
    let model_variant = parse_chat_model_variant(Some(&req.model))
        .map_err(|err| ApiError::bad_request(err.to_string()))?;

    let (messages, input_items) =
        build_input_messages(req.instructions.as_deref(), req.input.clone())?;
    if messages.is_empty() {
        return Err(ApiError::bad_request(
            "Responses request requires non-empty `input` or `instructions`",
        ));
    }

    if req.stream.unwrap_or(false) {
        return create_streaming_response(state, req, model_variant, messages, input_items).await;
    }

    let _permit = state.acquire_permit().await;

    let output = state
        .engine
        .chat_generate(
            model_variant,
            messages,
            req.max_output_tokens.unwrap_or(1536).clamp(1, 4096),
        )
        .await?;

    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let created_at = now_unix_secs();

    let usage = ResponseUsage {
        input_tokens: 0,
        output_tokens: output.tokens_generated,
        total_tokens: output.tokens_generated,
    };

    let response = ResponseObject {
        id: response_id.clone(),
        object: "response",
        created_at,
        status: "completed".to_string(),
        model: model_variant.dir_name().to_string(),
        output: vec![assistant_output_item(output.text.clone())],
        usage: usage.clone(),
        error: None,
        metadata: req.metadata.clone(),
    };

    persist_response(
        &state,
        StoredResponseRecord {
            id: response_id,
            created_at,
            status: "completed".to_string(),
            model: model_variant.dir_name().to_string(),
            input_items,
            output_text: Some(output.text),
            output_tokens: usage.output_tokens,
            error: None,
            metadata: req.metadata,
        },
        req.store,
    )
    .await;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::to_vec(&response).unwrap_or_default(),
        ))
        .unwrap())
}

pub async fn get_response(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
) -> Result<Json<ResponseObject>, ApiError> {
    let store = state.response_store.read().await;
    let record = store
        .get(&response_id)
        .cloned()
        .ok_or_else(|| ApiError::not_found("Response not found"))?;

    Ok(Json(record_to_response(record)))
}

pub async fn delete_response(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
) -> Result<Json<ResponseDeletedObject>, ApiError> {
    let mut store = state.response_store.write().await;
    if store.remove(&response_id).is_none() {
        return Err(ApiError::not_found("Response not found"));
    }

    Ok(Json(ResponseDeletedObject {
        id: response_id,
        object: "response.deleted",
        deleted: true,
    }))
}

pub async fn cancel_response(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
) -> Result<Json<ResponseObject>, ApiError> {
    let mut store = state.response_store.write().await;
    let record = store
        .get_mut(&response_id)
        .ok_or_else(|| ApiError::not_found("Response not found"))?;

    if record.status != "completed" {
        record.status = "cancelled".to_string();
        record.error = Some("Response was cancelled".to_string());
    }

    Ok(Json(record_to_response(record.clone())))
}

pub async fn list_response_input_items(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
) -> Result<Json<ResponseInputItemsList>, ApiError> {
    let store = state.response_store.read().await;
    let record = store
        .get(&response_id)
        .cloned()
        .ok_or_else(|| ApiError::not_found("Response not found"))?;

    let data = record
        .input_items
        .into_iter()
        .enumerate()
        .map(|(idx, item)| ResponseInputItemObject {
            id: format!("initem_{}_{idx}", response_id),
            item_type: "message",
            role: item.role,
            content: vec![ResponseInputItemContent {
                content_type: "input_text",
                text: item.content,
            }],
        })
        .collect();

    Ok(Json(ResponseInputItemsList {
        object: "list",
        data,
    }))
}

async fn create_streaming_response(
    state: AppState,
    req: ResponsesCreateRequest,
    model_variant: izwi_core::ModelVariant,
    messages: Vec<ChatMessage>,
    input_items: Vec<StoredResponseInputItem>,
) -> Result<Response<Body>, ApiError> {
    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let created_at = now_unix_secs();
    let metadata = req.metadata.clone();
    let response_id_for_task = response_id.clone();
    let model_name = model_variant.dir_name().to_string();

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<String>();
    let semaphore = state.request_semaphore.clone();
    let engine = state.engine.clone();
    let timeout = Duration::from_secs(state.request_timeout_secs);
    let store_state = state.clone();

    tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                let _ = event_tx.send(
                    serde_json::json!({
                        "type": "response.failed",
                        "response_id": response_id_for_task,
                        "error": {"message": "Server is shutting down"}
                    })
                    .to_string(),
                );
                let _ = event_tx.send("[DONE]".to_string());
                return;
            }
        };

        let created_response = ResponseObject {
            id: response_id_for_task.clone(),
            object: "response",
            created_at,
            status: "in_progress".to_string(),
            model: model_name.clone(),
            output: Vec::new(),
            usage: ResponseUsage {
                input_tokens: 0,
                output_tokens: 0,
                total_tokens: 0,
            },
            error: None,
            metadata: metadata.clone(),
        };

        let created_event = ResponseStreamEnvelope {
            event_type: "response.created",
            payload: ResponseStreamCreatedPayload {
                response: created_response,
            },
        };
        let _ = event_tx.send(serde_json::to_string(&created_event).unwrap_or_default());

        let delta_tx = event_tx.clone();
        let full_text = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let full_text_for_cb = full_text.clone();
        let response_id_for_delta = response_id_for_task.clone();

        let result = tokio::time::timeout(timeout, async {
            engine
                .chat_generate_streaming(
                    model_variant,
                    messages,
                    req.max_output_tokens.unwrap_or(1536).clamp(1, 4096),
                    move |delta| {
                        if let Ok(mut text) = full_text_for_cb.lock() {
                            text.push_str(&delta);
                        }

                        let delta_event = ResponseStreamEnvelope {
                            event_type: "response.output_text.delta",
                            payload: ResponseStreamDeltaPayload {
                                response_id: response_id_for_delta.clone(),
                                delta,
                            },
                        };

                        let _ =
                            delta_tx.send(serde_json::to_string(&delta_event).unwrap_or_default());
                    },
                )
                .await
        })
        .await;

        match result {
            Ok(Ok(generation)) => {
                let output_text = full_text.lock().map(|s| s.clone()).unwrap_or_default();
                let completed = ResponseObject {
                    id: response_id_for_task.clone(),
                    object: "response",
                    created_at,
                    status: "completed".to_string(),
                    model: model_name.clone(),
                    output: vec![assistant_output_item(output_text.clone())],
                    usage: ResponseUsage {
                        input_tokens: 0,
                        output_tokens: generation.tokens_generated,
                        total_tokens: generation.tokens_generated,
                    },
                    error: None,
                    metadata: metadata.clone(),
                };

                persist_response(
                    &store_state,
                    StoredResponseRecord {
                        id: response_id_for_task.clone(),
                        created_at,
                        status: "completed".to_string(),
                        model: model_name,
                        input_items,
                        output_text: Some(output_text),
                        output_tokens: generation.tokens_generated,
                        error: None,
                        metadata,
                    },
                    req.store,
                )
                .await;

                let completed_event = ResponseStreamEnvelope {
                    event_type: "response.completed",
                    payload: ResponseStreamCompletedPayload {
                        response: completed,
                    },
                };
                let _ = event_tx.send(serde_json::to_string(&completed_event).unwrap_or_default());
            }
            Ok(Err(err)) => {
                let _ = event_tx.send(
                    serde_json::json!({
                        "type": "response.failed",
                        "response_id": response_id_for_task,
                        "error": {"message": err.to_string()}
                    })
                    .to_string(),
                );
            }
            Err(_) => {
                let _ = event_tx.send(
                    serde_json::json!({
                        "type": "response.failed",
                        "response_id": response_id_for_task,
                        "error": {"message": "Response request timed out"}
                    })
                    .to_string(),
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

fn build_input_messages(
    instructions: Option<&str>,
    input: Option<ResponseInput>,
) -> Result<(Vec<ChatMessage>, Vec<StoredResponseInputItem>), ApiError> {
    let mut messages = Vec::new();
    let mut stored = Vec::new();

    if let Some(instructions) = instructions {
        if !instructions.trim().is_empty() {
            messages.push(ChatMessage {
                role: ChatRole::System,
                content: instructions.to_string(),
            });
            stored.push(StoredResponseInputItem {
                role: "system".to_string(),
                content: instructions.to_string(),
            });
        }
    }

    let input_items = match input {
        None => Vec::new(),
        Some(ResponseInput::Text(text)) => vec![("user".to_string(), text)],
        Some(ResponseInput::One(item)) => vec![normalize_input_item(item)?],
        Some(ResponseInput::Many(items)) => items
            .into_iter()
            .map(normalize_input_item)
            .collect::<Result<Vec<_>, _>>()?,
    };

    for (role, content) in input_items {
        let chat_role = match role.as_str() {
            "system" | "developer" => ChatRole::System,
            "assistant" => ChatRole::Assistant,
            "user" => ChatRole::User,
            other => {
                return Err(ApiError::bad_request(format!(
                    "Unsupported response input role: {}",
                    other
                )))
            }
        };

        messages.push(ChatMessage {
            role: chat_role,
            content: content.clone(),
        });

        stored.push(StoredResponseInputItem { role, content });
    }

    Ok((messages, stored))
}

fn normalize_input_item(item: super::dto::ResponseInputItem) -> Result<(String, String), ApiError> {
    let role = item
        .role
        .unwrap_or_else(|| "user".to_string())
        .trim()
        .to_ascii_lowercase();
    let content = flatten_content(item.content);

    if content.trim().is_empty() {
        return Err(ApiError::bad_request(
            "Response input item content cannot be empty",
        ));
    }

    Ok((role, content))
}

fn flatten_content(content: ResponseInputContent) -> String {
    match content {
        ResponseInputContent::Text(text) => text,
        ResponseInputContent::Parts(parts) => parts
            .into_iter()
            .filter_map(|part| {
                if part.kind.as_deref() == Some("input_text") {
                    part.input_text.or(part.text)
                } else {
                    part.text.or(part.input_text)
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn assistant_output_item(text: String) -> ResponseOutputItem {
    ResponseOutputItem {
        id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
        item_type: "message",
        role: "assistant",
        content: vec![ResponseOutputContent {
            content_type: "output_text",
            text,
        }],
    }
}

fn record_to_response(record: StoredResponseRecord) -> ResponseObject {
    ResponseObject {
        id: record.id,
        object: "response",
        created_at: record.created_at,
        status: record.status.clone(),
        model: record.model,
        output: record
            .output_text
            .map(assistant_output_item)
            .into_iter()
            .collect(),
        usage: ResponseUsage {
            input_tokens: 0,
            output_tokens: record.output_tokens,
            total_tokens: record.output_tokens,
        },
        error: record.error.map(|message| ResponseError {
            message,
            code: "response_error",
        }),
        metadata: record.metadata,
    }
}

async fn persist_response(state: &AppState, record: StoredResponseRecord, store: Option<bool>) {
    if store == Some(false) {
        return;
    }

    let mut response_store = state.response_store.write().await;
    response_store.insert(record.id.clone(), record);
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::super::dto::{ResponseInputContentPart, ResponseInputItem};
    use super::*;

    #[test]
    fn builds_messages_from_text_and_instructions() {
        let (messages, stored) = build_input_messages(
            Some("Be concise."),
            Some(ResponseInput::Text("Hello".to_string())),
        )
        .expect("build input messages");

        assert_eq!(messages.len(), 2);
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].role, "system");
        assert_eq!(stored[1].role, "user");
    }

    #[test]
    fn flattens_part_content() {
        let flattened = flatten_content(ResponseInputContent::Parts(vec![
            ResponseInputContentPart {
                kind: Some("input_text".to_string()),
                text: None,
                input_text: Some("part1".to_string()),
            },
            ResponseInputContentPart {
                kind: Some("text".to_string()),
                text: Some("part2".to_string()),
                input_text: None,
            },
        ]));
        assert_eq!(flattened, "part1\npart2");
    }

    #[test]
    fn normalizes_input_item_default_role() {
        let item = ResponseInputItem {
            role: None,
            content: ResponseInputContent::Text("hello".to_string()),
        };
        let (role, text) = normalize_input_item(item).expect("normalize");
        assert_eq!(role, "user");
        assert_eq!(text, "hello");
    }
}
