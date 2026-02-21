use crate::error::{Error, Result};
use crate::models::shared::chat::ChatMessage;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::state::ActiveChatDecode;
use super::{ExecutorOutput, NativeExecutor};

impl NativeExecutor {
    fn chat_messages(request: &EngineCoreRequest) -> Result<&[ChatMessage]> {
        request
            .chat_messages
            .as_deref()
            .ok_or_else(|| Error::InvalidInput("Chat request missing messages".to_string()))
    }

    pub(super) fn chat_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ExecutorOutput> {
        let variant = Self::resolve_variant(request)?;
        let messages = Self::chat_messages(request)?;
        let max_new_tokens = request.params.max_tokens.max(1);
        let stream_tx = Self::stream_sender(request);

        let model = self.with_registry(|registry| {
            registry
                .try_get_chat(variant)
                .ok_or_else(|| Error::ModelNotFound(format!("Chat model {variant} is not loaded")))
        })?;

        // Fallback path for chat backends that do not expose incremental decode state.
        if !model.supports_incremental_decode() {
            let output = Self::run_blocking(|| {
                if let Some(tx) = stream_tx.as_ref() {
                    let mut sequence = 0usize;
                    let mut stream_err: Option<Error> = None;
                    let mut emit = |delta: &str| {
                        if stream_err.is_none() {
                            if let Err(err) =
                                Self::stream_text(tx, &request.id, &mut sequence, delta.to_string())
                            {
                                stream_err = Some(err);
                            }
                        }
                    };
                    let output =
                        model.generate_with_callback(messages, max_new_tokens, &mut emit)?;
                    if let Some(err) = stream_err {
                        return Err(err);
                    }
                    Self::stream_final_marker(tx, &request.id, &mut sequence)?;
                    Ok(output)
                } else {
                    model.generate(messages, max_new_tokens)
                }
            })?;

            return Ok(ExecutorOutput {
                request_id: request.id.clone(),
                audio: Some(AudioOutput::empty(24_000)),
                text: Some(output.text),
                tokens_processed: request.num_prompt_tokens(),
                tokens_generated: output.tokens_generated.max(1),
                finished: true,
                error: None,
            });
        }

        let mut active_state = {
            let mut guard = self.chat_decode_states.lock().map_err(|_| {
                Error::InferenceError("Chat decode state mutex poisoned".to_string())
            })?;
            if scheduled.is_prefill {
                // Prefill scheduling can happen after preemption; reset stale state.
                guard.remove(&request.id)
            } else {
                guard.remove(&request.id)
            }
        };

        if active_state
            .as_ref()
            .map(|state| state.variant != variant)
            .unwrap_or(false)
        {
            active_state = None;
        }

        let mut active_state = if let Some(state) = active_state {
            state
        } else {
            let decode_state =
                Self::run_blocking(|| model.start_decode_state(messages, max_new_tokens))?;
            ActiveChatDecode {
                variant,
                state: decode_state,
                prompt_accounted: false,
                last_tokens_generated: 0,
                stream_sequence: 0,
            }
        };

        let step = Self::run_blocking(|| model.decode_step(&mut active_state.state))?;
        let step_tokens_generated = step
            .tokens_generated
            .saturating_sub(active_state.last_tokens_generated);
        active_state.last_tokens_generated = step.tokens_generated;

        let mut tokens_processed = scheduled.num_tokens.max(1);
        if !active_state.prompt_accounted {
            active_state.prompt_accounted = true;
            tokens_processed = tokens_processed.saturating_add(request.num_prompt_tokens());
        }

        if let Some(tx) = stream_tx.as_ref() {
            if !step.delta.is_empty() {
                Self::stream_text(
                    tx,
                    &request.id,
                    &mut active_state.stream_sequence,
                    step.delta.clone(),
                )?;
            }
            if step.finished {
                Self::stream_final_marker(tx, &request.id, &mut active_state.stream_sequence)?;
            }
        }

        if !step.finished {
            let mut guard = self.chat_decode_states.lock().map_err(|_| {
                Error::InferenceError("Chat decode state mutex poisoned".to_string())
            })?;
            guard.insert(request.id.clone(), active_state);
        }

        Ok(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput::empty(24_000)),
            text: Some(step.text),
            tokens_processed,
            tokens_generated: step_tokens_generated,
            finished: step.finished,
            error: None,
        })
    }
}
