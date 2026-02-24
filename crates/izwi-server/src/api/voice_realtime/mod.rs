//! Realtime voice websocket endpoint for `/voice`.
//!
//! Frontend responsibilities:
//! - microphone capture
//! - simple local VAD (speech start/stop)
//! - audio playback
//!
//! Backend responsibilities:
//! - ASR -> agent -> TTS orchestration
//! - streaming assistant audio/text events
//! - interruption / barge-in cancellation

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Extension, State,
    },
    response::Response,
    routing::get,
    Router,
};
use base64::Engine;
use futures::{SinkExt, StreamExt};
use izwi_agent::{
    planner::{PlanningMode, SimplePlanner},
    AgentDefinition, AgentEngine, AgentSession, AgentTurnOptions, MemoryMessage, MemoryMessageMeta,
    MemoryMessageRole, MemoryStore, ModelBackend, ModelOutput, ModelRequest, NoopTool, TimeTool,
    ToolRegistry, TurnInput,
};
use izwi_core::{
    audio::{AudioEncoder, AudioFormat},
    parse_chat_model_variant, parse_tts_model_variant, ChatMessage, ChatRole, GenerationConfig,
    GenerationRequest,
};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::api::request_context::RequestContext;
use crate::chat_store::ChatStore;
use crate::state::{AppState, StoredAgentSessionRecord};

const DEFAULT_AGENT_ID: &str = "voice-agent";
const DEFAULT_AGENT_NAME: &str = "Voice Agent";
const DEFAULT_AGENT_SYSTEM_PROMPT: &str =
    "You are a helpful voice assistant. Reply with concise spoken-friendly language. Avoid markdown. Keep responses brief unless asked for details.";
const DEFAULT_CHAT_MODEL: &str = "Qwen3-0.6B-4bit";
const MAX_UTTERANCE_BYTES: usize = 16 * 1024 * 1024;

pub fn router() -> Router<AppState> {
    Router::new().route("/voice/realtime/ws", get(ws_upgrade))
}

async fn ws_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
) -> Response {
    let correlation_id = ctx.correlation_id;
    ws.on_upgrade(move |socket| handle_socket(socket, state, correlation_id))
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientEvent {
    SessionStart {
        #[serde(default)]
        system_prompt: Option<String>,
    },
    InputAudioStart {
        utterance_id: String,
        utterance_seq: u64,
    },
    InputAudioCommit {
        utterance_id: String,
        utterance_seq: u64,
        #[serde(default)]
        mime_type: Option<String>,
        asr_model_id: String,
        text_model_id: String,
        tts_model_id: String,
        #[serde(default)]
        speaker: Option<String>,
        #[serde(default)]
        asr_language: Option<String>,
        #[serde(default)]
        max_output_tokens: Option<usize>,
    },
    Interrupt {
        #[serde(default)]
        reason: Option<String>,
    },
    Ping {
        #[serde(default)]
        timestamp_ms: Option<u64>,
    },
}

#[derive(Debug, Clone)]
struct PendingAudioCommit {
    utterance_id: String,
    utterance_seq: u64,
    mime_type: Option<String>,
    asr_model_id: String,
    text_model_id: String,
    tts_model_id: String,
    speaker: Option<String>,
    asr_language: Option<String>,
    max_output_tokens: usize,
}

#[derive(Debug)]
struct ActiveTurn {
    utterance_id: String,
    utterance_seq: u64,
    task: tokio::task::JoinHandle<()>,
}

struct ConnectionState {
    system_prompt: String,
    agent_session_id: Option<String>,
    agent_session_system_prompt: Option<String>,
    pending_audio_commit: Option<PendingAudioCommit>,
    active_turn: Option<ActiveTurn>,
    started: bool,
}

impl Default for ConnectionState {
    fn default() -> Self {
        Self {
            system_prompt: DEFAULT_AGENT_SYSTEM_PROMPT.to_string(),
            agent_session_id: None,
            agent_session_system_prompt: None,
            pending_audio_commit: None,
            active_turn: None,
            started: false,
        }
    }
}

async fn handle_socket(socket: WebSocket, state: AppState, correlation_id: String) {
    let (mut ws_tx, mut ws_rx) = socket.split();
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Message>();

    let writer = tokio::spawn(async move {
        while let Some(message) = out_rx.recv().await {
            if ws_tx.send(message).await.is_err() {
                break;
            }
        }
    });

    let mut conn = ConnectionState::default();
    send_json(
        &out_tx,
        json!({
            "type": "connected",
            "protocol": "voice_realtime_v1",
            "server_time_ms": now_unix_millis(),
        }),
    );

    while let Some(result) = ws_rx.next().await {
        let message = match result {
            Ok(message) => message,
            Err(err) => {
                warn!("voice realtime websocket receive error: {err}");
                break;
            }
        };

        match message {
            Message::Text(text) => {
                if let Err(err) = handle_text_message(
                    &state,
                    &correlation_id,
                    &out_tx,
                    &mut conn,
                    text.as_str(),
                )
                .await
                {
                    send_error(&out_tx, None, None, err);
                }
            }
            Message::Binary(data) => {
                if let Err(err) = handle_binary_message(
                    &state,
                    &correlation_id,
                    &out_tx,
                    &mut conn,
                    data.to_vec(),
                )
                .await
                {
                    send_error(&out_tx, None, None, err);
                }
            }
            Message::Close(_) => break,
            Message::Ping(payload) => {
                let _ = out_tx.send(Message::Pong(payload));
            }
            Message::Pong(_) => {}
        }
    }

    interrupt_active_turn(&out_tx, &mut conn.active_turn, "socket_closed");
    drop(out_tx);
    let _ = writer.await;
}

async fn handle_text_message(
    state: &AppState,
    correlation_id: &str,
    out_tx: &mpsc::UnboundedSender<Message>,
    conn: &mut ConnectionState,
    text: &str,
) -> Result<(), String> {
    let event: ClientEvent =
        serde_json::from_str(text).map_err(|err| format!("Invalid websocket payload: {err}"))?;

    match event {
        ClientEvent::SessionStart { system_prompt } => {
            if let Some(prompt) = system_prompt
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
            {
                if conn.agent_session_system_prompt.as_deref() != Some(prompt.as_str()) {
                    conn.agent_session_id = None;
                }
                conn.system_prompt = prompt;
            }
            conn.started = true;
            send_json(
                out_tx,
                json!({
                    "type": "session_ready",
                    "protocol": "voice_realtime_v1",
                }),
            );
        }
        ClientEvent::InputAudioStart {
            utterance_id,
            utterance_seq,
        } => {
            if !conn.started {
                return Err("Session not started. Send `session_start` first.".to_string());
            }

            debug!("voice ws input_audio_start: {utterance_id}/{utterance_seq}");
            // Barge-in: stop any in-flight turn before the new utterance is committed.
            interrupt_active_turn(out_tx, &mut conn.active_turn, "barge_in");
            send_json(
                out_tx,
                json!({
                    "type": "listening",
                    "utterance_id": utterance_id,
                    "utterance_seq": utterance_seq,
                }),
            );
        }
        ClientEvent::InputAudioCommit {
            utterance_id,
            utterance_seq,
            mime_type,
            asr_model_id,
            text_model_id,
            tts_model_id,
            speaker,
            asr_language,
            max_output_tokens,
        } => {
            if !conn.started {
                return Err("Session not started. Send `session_start` first.".to_string());
            }

            if conn.pending_audio_commit.is_some() {
                return Err(
                    "Server is already waiting for binary audio payload for a prior commit."
                        .to_string(),
                );
            }

            if asr_model_id.trim().is_empty()
                || text_model_id.trim().is_empty()
                || tts_model_id.trim().is_empty()
            {
                return Err(
                    "Missing required model ids (`asr_model_id`, `text_model_id`, `tts_model_id`)."
                        .to_string(),
                );
            }

            // Validate model ids early so protocol errors return before the binary upload.
            let _ = resolve_chat_model_id(Some(text_model_id.trim()))?;
            parse_tts_model_variant(tts_model_id.trim())
                .map_err(|err| format!("Unsupported TTS model: {err}"))?;

            conn.pending_audio_commit = Some(PendingAudioCommit {
                utterance_id,
                utterance_seq,
                mime_type,
                asr_model_id: asr_model_id.trim().to_string(),
                text_model_id: text_model_id.trim().to_string(),
                tts_model_id: tts_model_id.trim().to_string(),
                speaker: speaker.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()),
                asr_language: asr_language
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty()),
                max_output_tokens: max_output_tokens.unwrap_or(1536).clamp(1, 4096),
            });

            send_json(
                out_tx,
                json!({
                    "type": "awaiting_audio_binary",
                    "utterance_id": conn.pending_audio_commit.as_ref().map(|p| &p.utterance_id),
                    "utterance_seq": conn.pending_audio_commit.as_ref().map(|p| p.utterance_seq),
                }),
            );
        }
        ClientEvent::Interrupt { reason } => {
            let reason = reason.unwrap_or_else(|| "client_interrupt".to_string());
            interrupt_active_turn(out_tx, &mut conn.active_turn, &reason);
        }
        ClientEvent::Ping { timestamp_ms } => {
            send_json(
                out_tx,
                json!({
                    "type": "pong",
                    "timestamp_ms": timestamp_ms,
                    "server_time_ms": now_unix_millis(),
                }),
            );
        }
    }

    // Silence unused parameters in some branches (kept for future per-message needs).
    let _ = (state, correlation_id);
    Ok(())
}

async fn handle_binary_message(
    state: &AppState,
    correlation_id: &str,
    out_tx: &mpsc::UnboundedSender<Message>,
    conn: &mut ConnectionState,
    audio_bytes: Vec<u8>,
) -> Result<(), String> {
    let Some(commit) = conn.pending_audio_commit.take() else {
        return Err("Unexpected binary message (no pending `input_audio_commit`).".to_string());
    };

    if audio_bytes.is_empty() {
        return Err("Received empty binary audio payload.".to_string());
    }
    if audio_bytes.len() > MAX_UTTERANCE_BYTES {
        return Err(format!(
            "Audio payload too large ({} bytes). Max allowed is {} bytes.",
            audio_bytes.len(),
            MAX_UTTERANCE_BYTES
        ));
    }

    debug!(
        "voice ws committed audio payload: {} bytes (mime={:?}) for {}/{}",
        audio_bytes.len(),
        commit.mime_type,
        commit.utterance_id,
        commit.utterance_seq
    );

    interrupt_active_turn(out_tx, &mut conn.active_turn, "preempted_by_new_turn");

    let agent_session_id = ensure_agent_session(
        state,
        &mut conn.agent_session_id,
        &mut conn.agent_session_system_prompt,
        &conn.system_prompt,
        &commit.text_model_id,
    )
    .await?;

    let task = spawn_turn_task(
        state.clone(),
        correlation_id.to_string(),
        out_tx.clone(),
        commit.clone(),
        audio_bytes,
        agent_session_id,
    );

    conn.active_turn = Some(ActiveTurn {
        utterance_id: commit.utterance_id,
        utterance_seq: commit.utterance_seq,
        task,
    });

    Ok(())
}

fn spawn_turn_task(
    state: AppState,
    correlation_id: String,
    out_tx: mpsc::UnboundedSender<Message>,
    commit: PendingAudioCommit,
    audio_bytes: Vec<u8>,
    agent_session_id: String,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let timeout_secs = state.request_timeout_secs.max(1);
        let timeout = Duration::from_secs(timeout_secs);

        let turn_future = async {
            let _permit = state
                .request_semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| "Server is shutting down".to_string())?;

            send_json(
                &out_tx,
                json!({
                    "type": "turn_processing",
                    "utterance_id": commit.utterance_id,
                    "utterance_seq": commit.utterance_seq,
                }),
            );

            let audio_base64 = base64::engine::general_purpose::STANDARD.encode(audio_bytes);

            send_json(
                &out_tx,
                json!({
                    "type": "user_transcript_start",
                    "utterance_id": commit.utterance_id,
                    "utterance_seq": commit.utterance_seq,
                }),
            );

            let transcript = {
                let tx = out_tx.clone();
                let utt_id = commit.utterance_id.clone();
                let utt_seq = commit.utterance_seq;
                state.runtime
                    .asr_transcribe_streaming_with_correlation(
                        &audio_base64,
                        Some(&commit.asr_model_id),
                        commit.asr_language.as_deref(),
                        Some(&correlation_id),
                        move |delta| {
                            if delta.is_empty() {
                                return;
                            }
                            send_json(
                                &tx,
                                json!({
                                    "type": "user_transcript_delta",
                                    "utterance_id": utt_id,
                                    "utterance_seq": utt_seq,
                                    "delta": delta,
                                }),
                            );
                        },
                    )
                    .await
                    .map_err(|err| format!("ASR failed: {err}"))?
            };

            let user_text = transcript.text.trim().to_string();
            send_json(
                &out_tx,
                json!({
                    "type": "user_transcript_final",
                    "utterance_id": commit.utterance_id,
                    "utterance_seq": commit.utterance_seq,
                    "text": user_text,
                    "language": transcript.language,
                    "audio_duration_secs": transcript.duration_secs,
                }),
            );

            if user_text.is_empty() {
                send_json(
                    &out_tx,
                    json!({
                        "type": "turn_done",
                        "utterance_id": commit.utterance_id,
                        "utterance_seq": commit.utterance_seq,
                        "status": "no_input",
                    }),
                );
                return Ok::<(), String>(());
            }

            send_json(
                &out_tx,
                json!({
                    "type": "assistant_text_start",
                    "utterance_id": commit.utterance_id,
                    "utterance_seq": commit.utterance_seq,
                }),
            );

            let assistant_raw = run_agent_turn(
                &state,
                &agent_session_id,
                &user_text,
                &commit.text_model_id,
                commit.max_output_tokens,
                &correlation_id,
            )
            .await?;
            let assistant_text = strip_think_tags(&assistant_raw);

            send_json(
                &out_tx,
                json!({
                    "type": "assistant_text_final",
                    "utterance_id": commit.utterance_id,
                    "utterance_seq": commit.utterance_seq,
                    "text": assistant_text,
                    "raw_text": assistant_raw,
                }),
            );

            if assistant_text.is_empty() {
                send_json(
                    &out_tx,
                    json!({
                        "type": "turn_done",
                        "utterance_id": commit.utterance_id,
                        "utterance_seq": commit.utterance_seq,
                        "status": "ok",
                    }),
                );
                return Ok(());
            }

            stream_tts_to_socket(
                &state,
                &out_tx,
                &correlation_id,
                &commit,
                assistant_text.as_str(),
            )
            .await?;

            send_json(
                &out_tx,
                json!({
                    "type": "turn_done",
                    "utterance_id": commit.utterance_id,
                    "utterance_seq": commit.utterance_seq,
                    "status": "ok",
                }),
            );

            Ok(())
        };

        match tokio::time::timeout(timeout, turn_future).await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                send_error(
                    &out_tx,
                    Some(commit.utterance_id.clone()),
                    Some(commit.utterance_seq),
                    err,
                );
                send_json(
                    &out_tx,
                    json!({
                        "type": "turn_done",
                        "utterance_id": commit.utterance_id,
                        "utterance_seq": commit.utterance_seq,
                        "status": "error",
                    }),
                );
            }
            Err(_) => {
                send_error(
                    &out_tx,
                    Some(commit.utterance_id.clone()),
                    Some(commit.utterance_seq),
                    format!("Turn timed out after {timeout_secs} seconds"),
                );
                send_json(
                    &out_tx,
                    json!({
                        "type": "turn_done",
                        "utterance_id": commit.utterance_id,
                        "utterance_seq": commit.utterance_seq,
                        "status": "timeout",
                    }),
                );
            }
        }
    })
}

async fn stream_tts_to_socket(
    state: &AppState,
    out_tx: &mpsc::UnboundedSender<Message>,
    correlation_id: &str,
    commit: &PendingAudioCommit,
    text: &str,
) -> Result<(), String> {
    let tts_variant = parse_tts_model_variant(&commit.tts_model_id)
        .map_err(|err| format!("Unsupported TTS model: {err}"))?;
    state
        .runtime
        .load_model(tts_variant)
        .await
        .map_err(|err| format!("Failed to load TTS model: {err}"))?;

    let sample_rate = state.runtime.sample_rate().await;
    let encoder = AudioEncoder::new(sample_rate, 1);

    send_json(
        out_tx,
        json!({
            "type": "assistant_audio_start",
            "utterance_id": commit.utterance_id,
            "utterance_seq": commit.utterance_seq,
            "sample_rate": sample_rate,
            "audio_format": "pcm_i16",
        }),
    );

    let mut gen_config = GenerationConfig::default();
    gen_config.streaming = true;
    gen_config.options.max_tokens = 0;
    gen_config.options.speaker = commit.speaker.clone();
    gen_config.options.voice = commit.speaker.clone();

    let gen_request = GenerationRequest {
        id: uuid::Uuid::new_v4().to_string(),
        correlation_id: Some(correlation_id.to_string()),
        text: text.to_string(),
        config: gen_config,
        language: None,
        reference_audio: None,
        reference_text: None,
        voice_description: None,
    };

    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<izwi_core::AudioChunk>(32);
    let runtime = state.runtime.clone();
    let generation_task =
        tokio::spawn(async move { runtime.generate_streaming(gen_request, chunk_tx).await });

    while let Some(chunk) = chunk_rx.recv().await {
        if chunk.samples.is_empty() && !chunk.is_final {
            continue;
        }

        let encoded = encoder
            .encode(&chunk.samples, AudioFormat::RawI16)
            .map_err(|err| format!("Failed to encode streamed TTS chunk: {err}"))?;

        send_json(
            out_tx,
            json!({
                "type": "assistant_audio_chunk",
                "utterance_id": commit.utterance_id,
                "utterance_seq": commit.utterance_seq,
                "sequence": chunk.sequence,
                "audio_base64": base64::engine::general_purpose::STANDARD.encode(encoded),
                "sample_count": chunk.samples.len(),
                "is_final": chunk.is_final,
            }),
        );
    }

    match generation_task.await {
        Ok(Ok(())) => {
            send_json(
                out_tx,
                json!({
                    "type": "assistant_audio_done",
                    "utterance_id": commit.utterance_id,
                    "utterance_seq": commit.utterance_seq,
                }),
            );
            Ok(())
        }
        Ok(Err(err)) => Err(format!("TTS failed: {err}")),
        Err(err) => Err(format!("TTS streaming task failed: {err}")),
    }
}

fn interrupt_active_turn(
    out_tx: &mpsc::UnboundedSender<Message>,
    active_turn: &mut Option<ActiveTurn>,
    reason: &str,
) {
    if let Some(turn) = active_turn.take() {
        if turn.task.is_finished() {
            return;
        }
        turn.task.abort();
        send_json(
            out_tx,
            json!({
                "type": "turn_done",
                "utterance_id": turn.utterance_id,
                "utterance_seq": turn.utterance_seq,
                "status": "interrupted",
                "reason": reason,
            }),
        );
    }
}

async fn ensure_agent_session(
    state: &AppState,
    agent_session_id: &mut Option<String>,
    agent_session_system_prompt: &mut Option<String>,
    system_prompt: &str,
    text_model_id: &str,
) -> Result<String, String> {
    if let Some(existing_id) = agent_session_id.as_ref() {
        if agent_session_system_prompt.as_deref() == Some(system_prompt) {
            return Ok(existing_id.clone());
        }
    }

    let model_id = resolve_chat_model_id(Some(text_model_id))?;
    let thread = state
        .chat_store
        .create_thread(Some("Voice Session".to_string()), Some(model_id.clone()))
        .await
        .map_err(|err| format!("Chat storage error: {err}"))?;

    let now = now_unix_millis();
    let session_id = format!("agent_sess_{}", uuid::Uuid::new_v4().simple());
    let record = StoredAgentSessionRecord {
        id: session_id.clone(),
        agent_id: DEFAULT_AGENT_ID.to_string(),
        thread_id: thread.id,
        model_id,
        system_prompt: system_prompt.to_string(),
        planning_mode: PlanningMode::Auto,
        created_at: now,
        updated_at: now,
    };

    state
        .agent_session_store
        .write()
        .await
        .insert(session_id.clone(), record);

    *agent_session_id = Some(session_id.clone());
    *agent_session_system_prompt = Some(system_prompt.to_string());
    Ok(session_id)
}

async fn run_agent_turn(
    state: &AppState,
    session_id: &str,
    input: &str,
    model_id: &str,
    max_output_tokens: usize,
    correlation_id: &str,
) -> Result<String, String> {
    let session_record = {
        let store = state.agent_session_store.read().await;
        store.get(session_id)
            .cloned()
            .ok_or_else(|| "Agent session not found".to_string())?
    };

    let resolved_model_id = resolve_chat_model_id(Some(model_id))?;

    let agent = AgentDefinition {
        id: session_record.agent_id.clone(),
        name: DEFAULT_AGENT_NAME.to_string(),
        system_prompt: session_record.system_prompt.clone(),
        default_model: session_record.model_id.clone(),
        capabilities: Default::default(),
        planning_mode: session_record.planning_mode,
    };
    let session = AgentSession {
        id: session_record.id.clone(),
        agent_id: session_record.agent_id.clone(),
        thread_id: session_record.thread_id.clone(),
        created_at: session_record.created_at,
        updated_at: session_record.updated_at,
    };

    let memory = ChatStoreMemory::new(state.chat_store.clone());
    let backend = IzwiRuntimeBackend {
        runtime: state.runtime.clone(),
        correlation_id: correlation_id.to_string(),
    };
    let planner = SimplePlanner;
    let mut tools = ToolRegistry::new();
    tools.register(NoopTool);
    tools.register(TimeTool);

    let result = AgentEngine
        .run_turn(
            &agent,
            &session,
            TurnInput {
                text: input.to_string(),
            },
            Some(resolved_model_id.clone()),
            &memory,
            &backend,
            &planner,
            &tools,
            AgentTurnOptions {
                max_output_tokens: max_output_tokens.clamp(1, 4096),
                max_tool_calls: 1,
            },
        )
        .await
        .map_err(|err| match err {
            izwi_agent::AgentError::InvalidInput(msg) => msg,
            other => other.to_string(),
        })?;

    {
        let mut store = state.agent_session_store.write().await;
        if let Some(record) = store.get_mut(session_id) {
            record.updated_at = now_unix_millis();
            record.model_id = resolved_model_id;
        }
    }

    Ok(result.assistant_text)
}

fn resolve_chat_model_id(raw: Option<&str>) -> Result<String, String> {
    let requested = raw
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_CHAT_MODEL);
    let variant =
        parse_chat_model_variant(Some(requested)).map_err(|err| format!("Invalid chat model: {err}"))?;
    Ok(variant.dir_name().to_string())
}

fn send_json(out_tx: &mpsc::UnboundedSender<Message>, value: serde_json::Value) -> bool {
    match serde_json::to_string(&value) {
        Ok(text) => out_tx.send(Message::Text(text.into())).is_ok(),
        Err(err) => {
            warn!("failed to serialize voice ws event: {err}");
            false
        }
    }
}

fn send_error(
    out_tx: &mpsc::UnboundedSender<Message>,
    utterance_id: Option<String>,
    utterance_seq: Option<u64>,
    message: impl Into<String>,
) {
    let message = message.into();
    let _ = send_json(
        out_tx,
        json!({
            "type": "error",
            "utterance_id": utterance_id,
            "utterance_seq": utterance_seq,
            "message": message,
        }),
    );
}

fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn strip_think_tags(input: &str) -> String {
    let open_tag = "<think>";
    let close_tag = "</think>";
    let mut out = input.to_string();

    loop {
        let Some(start) = out.find(open_tag) else {
            break;
        };
        if let Some(end_rel) = out[start + open_tag.len()..].find(close_tag) {
            let end = start + open_tag.len() + end_rel;
            let mut next = String::with_capacity(out.len());
            next.push_str(&out[..start]);
            next.push_str(&out[end + close_tag.len()..]);
            out = next;
        } else {
            out.truncate(start);
            break;
        }
    }

    out.trim().to_string()
}

struct ChatStoreMemory {
    chat_store: Arc<ChatStore>,
}

impl ChatStoreMemory {
    fn new(chat_store: Arc<ChatStore>) -> Self {
        Self { chat_store }
    }
}

#[async_trait::async_trait]
impl MemoryStore for ChatStoreMemory {
    async fn load_messages(&self, thread_id: &str) -> izwi_agent::Result<Vec<MemoryMessage>> {
        let records = self
            .chat_store
            .list_messages(thread_id.to_string())
            .await
            .map_err(|err| izwi_agent::AgentError::Memory(err.to_string()))?;

        let mut out = Vec::with_capacity(records.len());
        for record in records {
            let role = match record.role.as_str() {
                "system" => MemoryMessageRole::System,
                "user" => MemoryMessageRole::User,
                "assistant" => MemoryMessageRole::Assistant,
                other => {
                    return Err(izwi_agent::AgentError::Memory(format!(
                        "Invalid stored chat role: {other}"
                    )))
                }
            };
            out.push(MemoryMessage {
                role,
                content: record.content,
            });
        }

        Ok(out)
    }

    async fn append_message(
        &self,
        thread_id: &str,
        role: MemoryMessageRole,
        content: String,
        meta: MemoryMessageMeta,
    ) -> izwi_agent::Result<()> {
        self.chat_store
            .append_message(
                thread_id.to_string(),
                role.as_str().to_string(),
                content,
                meta.model_id,
                meta.tokens_generated,
                meta.generation_time_ms,
            )
            .await
            .map_err(|err| izwi_agent::AgentError::Memory(err.to_string()))?;
        Ok(())
    }
}

struct IzwiRuntimeBackend {
    runtime: Arc<izwi_core::RuntimeService>,
    correlation_id: String,
}

#[async_trait::async_trait]
impl ModelBackend for IzwiRuntimeBackend {
    async fn generate(&self, request: ModelRequest) -> izwi_agent::Result<ModelOutput> {
        let variant = parse_chat_model_variant(Some(&request.model_id))
            .map_err(|err| izwi_agent::AgentError::Model(err.to_string()))?;

        let mut runtime_messages = Vec::with_capacity(request.messages.len());
        for message in request.messages {
            let role = match message.role {
                MemoryMessageRole::System => ChatRole::System,
                MemoryMessageRole::User => ChatRole::User,
                MemoryMessageRole::Assistant => ChatRole::Assistant,
            };
            runtime_messages.push(ChatMessage {
                role,
                content: message.content,
            });
        }

        let generation = self
            .runtime
            .chat_generate_with_correlation(
                variant,
                runtime_messages,
                request.max_output_tokens.clamp(1, 4096),
                Some(&self.correlation_id),
            )
            .await
            .map_err(|err| izwi_agent::AgentError::Model(err.to_string()))?;

        Ok(ModelOutput {
            text: generation.text,
            tokens_generated: generation.tokens_generated,
            generation_time_ms: generation.generation_time_ms,
        })
    }
}
