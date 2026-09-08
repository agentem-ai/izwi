//! Rollback-safe staged Fish S2 TTS generation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::backends::kv::KvWriteBatchCompletion;
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

use super::{
    append_generated_frame, elapsed_ms, generated_frame_prompt, sample_semantic_token,
    FishS2ConditioningPrompt, FishS2GenerationOutput, FishS2GenerationParams, FishS2Reference,
    FishS2Sampler, FishS2SemanticSampler, FishS2SlowOutput, FishS2TtsGenerationDiagnostics,
    FishS2TtsModel, FishS2VqCodes, RAS_WIN_SIZE,
};

static NEXT_FISH_S2_STATE_ID: AtomicU64 = AtomicU64::new(1);

/// Host ownership and accelerator workspace required before reference preparation.
/// Heap bytes exclude allocator metadata, consistently with other resource charges.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FishS2PreparationMemory {
    pub host_transient_bytes: u64,
    pub retained_artifact_bytes: u64,
    pub accelerator_workspace_bytes: u64,
}

impl FishS2PreparationMemory {
    pub(crate) fn host_bytes(self) -> Result<u64> {
        checked_bytes_sum(&[self.host_transient_bytes, self.retained_artifact_bytes])
    }
}

fn checked_bytes_sum(values: &[u64]) -> Result<u64> {
    values
        .iter()
        .try_fold(0u64, |sum, value| sum.checked_add(*value))
        .ok_or_else(|| Error::Overloaded("Fish S2 preparation byte count overflow".into()))
}

fn allocation_bytes(count: usize, element_size: usize) -> Result<u64> {
    count
        .checked_mul(element_size)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| Error::Overloaded("Fish S2 preparation allocation overflow".into()))
}

fn artifact_bound(num_codebooks: usize, tokens: usize) -> Result<u64> {
    let rows = num_codebooks
        .checked_add(1)
        .ok_or_else(|| Error::Overloaded("Fish S2 artifact rows overflow".into()))?;
    checked_bytes_sum(&[
        allocation_bytes(rows, std::mem::size_of::<Vec<u32>>())?,
        allocation_bytes(
            rows.checked_mul(tokens)
                .ok_or_else(|| Error::Overloaded("Fish S2 artifact shape overflow".into()))?,
            4,
        )?,
        allocation_bytes(tokens, std::mem::size_of::<bool>())?,
        // Include the Arc allocation's object and strong/weak counters once.
        allocation_bytes(
            1,
            std::mem::size_of::<FishS2PreparedArtifact>() + 2 * std::mem::size_of::<usize>(),
        )?,
    ])
}

#[derive(Clone)]
pub(crate) struct FishS2PreparedArtifact {
    model_identity: u64,
    prompt: FishS2ConditioningPrompt,
    reference_encode_ms: f32,
    prompt_build_ms: f32,
}

impl std::fmt::Debug for FishS2PreparedArtifact {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FishS2PreparedArtifact")
            .field("model_identity", &self.model_identity)
            .field("prompt_tokens", &self.prompt.prompt_length)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FishS2RetainedStep {
    Prefill {
        consumed: usize,
        position: usize,
        complete: bool,
    },
    Frame {
        frames_generated: usize,
    },
    Finished {
        frames_generated: usize,
    },
}

pub(crate) struct FishS2RetainedState {
    state_id: u64,
    model_identity: u64,
    artifact: Arc<FishS2PreparedArtifact>,
    params: FishS2GenerationParams,
    slow_cache: PhysicalPagedKvCache,
    slow_position: usize,
    slow_output: Option<FishS2SlowOutput>,
    semantic_sampler: FishS2SemanticSampler,
    fast_sampler: FishS2Sampler,
    generated_codebooks: Vec<Vec<u32>>,
    recent_semantic_tokens: Vec<u32>,
    max_frames: usize,
    stop_reason: String,
    finished: bool,
    active_quantum: Option<u64>,
    next_quantum: u64,
    staged_step: Option<FishS2RetainedStep>,
    completions_drained: bool,
}

pub(crate) struct FishS2RetainedCheckpoint {
    state_id: u64,
    quantum: u64,
    payload: Option<FishS2RetainedCheckpointPayload>,
}

struct FishS2RetainedCheckpointPayload {
    slow_cache: Option<PhysicalPagedKvCache>,
    slow_position: usize,
    slow_output: Option<FishS2SlowOutput>,
    semantic_sampler: FishS2SemanticSampler,
    fast_sampler: FishS2Sampler,
    generated_codebooks: Vec<Vec<u32>>,
    recent_semantic_tokens: Vec<u32>,
    stop_reason: String,
    finished: bool,
    staged_step: Option<FishS2RetainedStep>,
    completions_drained: bool,
}

impl FishS2TtsModel {
    pub(crate) fn preparation_memory(
        &self,
        text: &str,
        reference: &FishS2Reference,
        context_limit: usize,
    ) -> Result<FishS2PreparationMemory> {
        validate_preparation_inputs(text, reference)?;
        preparation_memory_for_geometry(
            &self.config,
            text.len(),
            reference.text.capacity(),
            reference.audio_samples.len(),
            reference.audio_samples.capacity(),
            reference.sample_rate,
            context_limit,
        )
    }

    pub(crate) fn prepare_retained_artifact(
        &self,
        text: &str,
        reference: FishS2Reference,
    ) -> Result<Arc<FishS2PreparedArtifact>> {
        self.prepare_retained_artifact_with_cancel(
            text,
            reference,
            self.config.max_seq_len,
            &|| Ok(()),
        )
    }

    pub(crate) fn prepare_retained_artifact_with_cancel(
        &self,
        text: &str,
        reference: FishS2Reference,
        context_limit: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Arc<FishS2PreparedArtifact>> {
        check()?;
        self.preparation_memory(text, &reference, context_limit)?;
        let runtime = self.native_runtime()?;
        let started = Instant::now();
        let mut reference_encode_ms = 0.0;
        let (reference_codes, cache_hit) = self.reference_cache.get_or_encode(
            &reference.audio_samples,
            reference.sample_rate,
            check,
            || {
                let encode_started = Instant::now();
                let codes = runtime.dac.encode_reference_audio_with_cancel(
                    &reference.audio_samples,
                    reference.sample_rate,
                    check,
                );
                reference_encode_ms = elapsed_ms(encode_started);
                codes
            },
        )?;
        tracing::debug!(
            reference_cache_hit = cache_hit,
            reference_lookup_total_ms = elapsed_ms(started),
            reference_encode_ms,
            "Fish reference code preparation"
        );
        let started = Instant::now();
        let prompt = runtime.tokenizer.build_reference_voice_prompt_bounded(
            &self.config,
            reference.text.trim(),
            reference_codes,
            text.trim(),
            context_limit,
        )?;
        let prompt_build_ms = elapsed_ms(started);
        check()?;
        if prompt.prompt_length >= self.config.max_seq_len {
            return Err(Error::InvalidInput(format!(
                "Fish S2 prompt length {} exceeds max_seq_len {}",
                prompt.prompt_length, self.config.max_seq_len
            )));
        }
        Ok(Arc::new(FishS2PreparedArtifact {
            model_identity: self.model_identity,
            prompt,
            reference_encode_ms,
            prompt_build_ms,
        }))
    }

    pub(crate) fn new_retained_state(
        &self,
        artifact: Arc<FishS2PreparedArtifact>,
        params: FishS2GenerationParams,
        slow_cache: PhysicalPagedKvCache,
        max_sequence_tokens: usize,
    ) -> Result<FishS2RetainedState> {
        params.validate()?;
        if artifact.model_identity != self.model_identity {
            return Err(Error::InvalidInput(
                "Fish S2 prepared artifact belongs to another model load".into(),
            ));
        }
        if slow_cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Fish S2 retained caches must begin empty".into(),
            ));
        }
        let max_frames = super::effective_frame_budget(
            artifact.prompt.prompt_length,
            self.config.max_seq_len,
            max_sequence_tokens,
            params.max_frames,
        )?;
        let semantic_sampler = FishS2SemanticSampler::from_params(&params);
        let fast_sampler = FishS2Sampler::with_top_k(
            params.temperature,
            params.top_p,
            params.top_k,
            params.seed.wrapping_add(1),
        );
        Ok(FishS2RetainedState {
            state_id: next_state_id()?,
            model_identity: self.model_identity,
            artifact,
            params,
            slow_cache,
            slow_position: 0,
            slow_output: None,
            semantic_sampler,
            fast_sampler,
            generated_codebooks: vec![Vec::new(); self.config.num_codebooks],
            recent_semantic_tokens: Vec::with_capacity(RAS_WIN_SIZE),
            max_frames,
            stop_reason: "max_frames".into(),
            finished: false,
            active_quantum: None,
            next_quantum: 1,
            staged_step: None,
            completions_drained: true,
        })
    }

    pub(crate) fn new_retained_state_in_quantum(
        &self,
        artifact: Arc<FishS2PreparedArtifact>,
        params: FishS2GenerationParams,
        slow_cache: PhysicalPagedKvCache,
        max_sequence_tokens: usize,
    ) -> Result<(FishS2RetainedState, FishS2RetainedCheckpoint)> {
        // The cache contains only this quantum's pages, not the full sequence.
        let mut state =
            self.new_retained_state(artifact, params, slow_cache, max_sequence_tokens)?;
        state.active_quantum = Some(1);
        state.next_quantum = 2;
        let checkpoint = FishS2RetainedCheckpoint {
            state_id: state.state_id,
            quantum: 1,
            payload: Some(FishS2RetainedCheckpointPayload {
                slow_cache: None,
                slow_position: 0,
                slow_output: None,
                semantic_sampler: state.semantic_sampler.clone(),
                fast_sampler: state.fast_sampler.clone(),
                generated_codebooks: vec![Vec::new(); self.config.num_codebooks],
                recent_semantic_tokens: Vec::new(),
                stop_reason: "max_frames".into(),
                finished: false,
                staged_step: None,
                completions_drained: true,
            }),
        };
        Ok((state, checkpoint))
    }

    pub(crate) fn retained_prefill_step(
        &self,
        state: &mut FishS2RetainedState,
        max_tokens: usize,
    ) -> Result<FishS2RetainedStep> {
        self.validate_retained_state(state)?;
        state.require_clean_quantum()?;
        if max_tokens == 0 {
            return Err(Error::InvalidInput(
                "Fish S2 prefill quantum must be nonzero".into(),
            ));
        }
        let remaining = state.artifact.prompt.prompt_length - state.slow_position;
        let consumed = remaining.min(max_tokens);
        if consumed > 0 {
            let prompt = slice_prompt(&state.artifact.prompt, state.slow_position, consumed)?;
            let runtime = self.native_runtime()?;
            let embeds = runtime.slow.embed_prompt(&prompt)?;
            state.slow_output = Some(runtime.slow.forward_embeds(
                &embeds,
                state.slow_position,
                &mut state.slow_cache,
                false,
            )?);
            state.slow_position += consumed;
            state.completions_drained = false;
        }
        state.stage(FishS2RetainedStep::Prefill {
            consumed,
            position: state.slow_position,
            complete: state.slow_position == state.artifact.prompt.prompt_length,
        })
    }

    pub(crate) fn retained_decode_step(
        &self,
        state: &mut FishS2RetainedState,
        fast_cache: &mut PhysicalPagedKvCache,
    ) -> Result<FishS2RetainedStep> {
        self.validate_retained_state(state)?;
        state.require_clean_quantum()?;
        if state.slow_position < state.artifact.prompt.prompt_length {
            return Err(Error::InferenceError(
                "Fish S2 decode cannot run before prefill completes".into(),
            ));
        }
        if state.finished {
            return state.stage_finished();
        }
        let runtime = self.native_runtime()?;
        let slow = state.slow_output.as_ref().ok_or_else(|| {
            Error::InferenceError("Fish S2 retained state has no slow output".into())
        })?;
        let semantic_index = sample_semantic_token(
            &slow.logits,
            &runtime.semantic_allowed_mask,
            runtime
                .slow
                .eos_logit_index(runtime.tokenizer.specials().eos),
            state.frames_generated() > 0,
            &state.recent_semantic_tokens,
            &mut state.semantic_sampler,
        )?;
        let semantic = runtime.slow.token_id_from_logit(semantic_index)?;
        if semantic == runtime.tokenizer.specials().eos {
            // Complete the slow-token append authorized for this quantum even
            // when EOS terminates generation; every committed row needs its KV receipt.
            let mut values = vec![vec![0]; self.config.num_codebooks + 1];
            values[0][0] = semantic;
            let prompt = FishS2ConditioningPrompt {
                values,
                vq_mask: vec![false],
                prompt_length: 1,
            };
            let embeds = runtime.slow.embed_prompt(&prompt)?;
            state.slow_output = Some(runtime.slow.forward_embeds(
                &embeds,
                state.slow_position,
                &mut state.slow_cache,
                false,
            )?);
            state.slow_position += 1;
            state.completions_drained = false;
            state.stop_reason = "im_end".into();
            state.finished = true;
            return state.stage_finished();
        }
        // The fast clock restarts inside this bounded invocation. Only complete
        // frames cross the scheduler's retained-state commit boundary.
        let frame = runtime.fast.generate_frame(
            semantic,
            &slow.hidden_states,
            &mut state.fast_sampler,
            fast_cache,
        )?;
        append_generated_frame(&mut state.generated_codebooks, &frame)?;
        state.recent_semantic_tokens.push(semantic_index);
        if state.recent_semantic_tokens.len() > RAS_WIN_SIZE {
            state.recent_semantic_tokens.remove(0);
        }
        let frame_prompt = generated_frame_prompt(self.config.num_codebooks, &frame)?;
        let embeds = runtime.slow.embed_prompt(&frame_prompt)?;
        state.slow_output = Some(runtime.slow.forward_embeds(
            &embeds,
            state.slow_position,
            &mut state.slow_cache,
            false,
        )?);
        state.slow_position += 1;
        state.completions_drained = false;
        if state.frames_generated() >= state.max_frames {
            state.finished = true;
            return state.stage_finished();
        }
        state.stage(FishS2RetainedStep::Frame {
            frames_generated: state.frames_generated(),
        })
    }

    pub(crate) fn finalize_retained_state(
        &self,
        state: &FishS2RetainedState,
    ) -> Result<FishS2GenerationOutput> {
        self.finalize_retained_state_with_cancel(state, &|| Ok(()))
    }

    pub(crate) fn finalize_retained_state_with_cancel(
        &self,
        state: &FishS2RetainedState,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<FishS2GenerationOutput> {
        check()?;
        self.validate_retained_state(state)?;
        if !state.finished || state.active_quantum.is_some() {
            return Err(Error::InferenceError(
                "Fish S2 finalize requires a terminal committed frame boundary".into(),
            ));
        }
        let frames_generated = state.frames_generated();
        if frames_generated == 0 {
            return Err(Error::InferenceError(
                "Fish S2 generation produced no audio frames".into(),
            ));
        }
        let started = Instant::now();
        let samples = self.native_runtime()?.dac.decode_vq_codes_with_cancel(
            &FishS2VqCodes {
                codebooks: state.generated_codebooks.clone(),
            },
            check,
        )?;
        check()?;
        let dac_decode_ms = elapsed_ms(started);
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::InferenceError(
                "Fish S2 DAC produced empty or non-finite audio".into(),
            ));
        }
        let sample_rate = self.native_runtime()?.dac.config().sample_rate;
        Ok(FishS2GenerationOutput {
            samples,
            sample_rate,
            frames_generated,
            diagnostics: FishS2TtsGenerationDiagnostics {
                model_family: "fish_s2_tts",
                sample_rate,
                prompt_tokens: state.artifact.prompt.prompt_length,
                max_frames: state.max_frames,
                frames_generated,
                temperature: state.params.temperature,
                top_p: state.params.top_p,
                top_k: state.params.top_k,
                seed: state.params.seed,
                repetition_aware: state.params.repetition_aware,
                stop_reason: state.stop_reason.clone(),
                reference_encode_ms: state.artifact.reference_encode_ms,
                prompt_build_ms: state.artifact.prompt_build_ms,
                slow_prefill_ms: 0.0,
                ar_decode_ms: 0.0,
                dac_decode_ms,
                total_model_ms: 0.0,
            },
        })
    }

    fn native_runtime(&self) -> Result<&super::FishS2NativeRuntime> {
        self.runtime
            .as_ref()
            .ok_or_else(|| Error::ModelLoadError("Fish S2 native runtime is not loaded".into()))
    }

    fn validate_retained_state(&self, state: &FishS2RetainedState) -> Result<()> {
        if state.model_identity != self.model_identity
            || state.artifact.model_identity != self.model_identity
        {
            return Err(Error::InvalidInput(
                "Fish S2 retained state belongs to another model load".into(),
            ));
        }
        Ok(())
    }
}

impl FishS2PreparedArtifact {
    #[cfg(test)]
    pub(crate) fn test_prompt(rows: usize, tokens: usize) -> Arc<Self> {
        Arc::new(Self {
            model_identity: 1,
            prompt: FishS2ConditioningPrompt {
                values: vec![vec![0; tokens]; rows],
                vq_mask: vec![false; tokens],
                prompt_length: tokens,
            },
            reference_encode_ms: 0.0,
            prompt_build_ms: 0.0,
        })
    }

    pub(crate) const fn prompt_tokens(&self) -> usize {
        self.prompt.prompt_length
    }

    pub(crate) fn retained_bytes(&self) -> Result<u64> {
        let mut bytes = checked_bytes_sum(&[
            allocation_bytes(
                self.prompt.values.capacity(),
                std::mem::size_of::<Vec<u32>>(),
            )?,
            allocation_bytes(self.prompt.vq_mask.capacity(), std::mem::size_of::<bool>())?,
            allocation_bytes(
                1,
                std::mem::size_of::<Self>() + 2 * std::mem::size_of::<usize>(),
            )?,
        ])?;
        for row in &self.prompt.values {
            bytes = checked_bytes_sum(&[bytes, allocation_bytes(row.capacity(), 4)?])?;
        }
        Ok(bytes)
    }
}

impl FishS2RetainedState {
    pub(crate) fn begin_managed_quantum(
        &mut self,
        slow_cache: PhysicalPagedKvCache,
    ) -> Result<FishS2RetainedCheckpoint> {
        if self.active_quantum.is_some() || self.staged_step.is_some() || !self.completions_drained
        {
            return Err(Error::InferenceError(
                "Fish S2 retained quantum is not clean".into(),
            ));
        }
        if slow_cache.arena().id() != self.slow_cache.arena().id()
            || slow_cache.context_len() != self.slow_position
        {
            return Err(Error::InvalidInput(
                "Fish S2 managed cache authority or position changed".into(),
            ));
        }
        let quantum = self.next_quantum;
        self.next_quantum = quantum
            .checked_add(1)
            .ok_or_else(|| Error::InferenceError("Fish S2 quantum overflow".into()))?;
        self.active_quantum = Some(quantum);
        Ok(FishS2RetainedCheckpoint {
            state_id: self.state_id,
            quantum,
            payload: Some(FishS2RetainedCheckpointPayload {
                slow_cache: Some(std::mem::replace(&mut self.slow_cache, slow_cache)),
                slow_position: self.slow_position,
                slow_output: self.slow_output.clone(),
                semantic_sampler: self.semantic_sampler.clone(),
                fast_sampler: self.fast_sampler.clone(),
                generated_codebooks: self.generated_codebooks.clone(),
                recent_semantic_tokens: self.recent_semantic_tokens.clone(),
                stop_reason: self.stop_reason.clone(),
                finished: self.finished,
                staged_step: self.staged_step.clone(),
                completions_drained: self.completions_drained,
            }),
        })
    }

    pub(crate) fn commit_managed_quantum(
        &mut self,
        checkpoint: &mut FishS2RetainedCheckpoint,
    ) -> Result<()> {
        self.validate_checkpoint(checkpoint)?;
        if !self.completions_drained || self.staged_step.is_some() {
            return Err(Error::InferenceError(
                "Fish S2 completions and staged output must be drained before commit".into(),
            ));
        }
        checkpoint.payload.take();
        self.active_quantum = None;
        Ok(())
    }

    pub(crate) fn rollback_managed_quantum(
        &mut self,
        checkpoint: &mut FishS2RetainedCheckpoint,
    ) -> Result<()> {
        self.validate_checkpoint(checkpoint)?;
        let payload = checkpoint.payload.take().ok_or_else(|| {
            Error::InferenceError("Fish S2 checkpoint was already consumed".into())
        })?;
        self.slow_cache = payload.slow_cache.ok_or_else(|| {
            Error::InferenceError("initial Fish S2 state must be discarded on rollback".into())
        })?;
        self.slow_position = payload.slow_position;
        self.slow_output = payload.slow_output;
        self.semantic_sampler = payload.semantic_sampler;
        self.fast_sampler = payload.fast_sampler;
        self.generated_codebooks = payload.generated_codebooks;
        self.recent_semantic_tokens = payload.recent_semantic_tokens;
        self.stop_reason = payload.stop_reason;
        self.finished = payload.finished;
        self.staged_step = payload.staged_step;
        self.completions_drained = payload.completions_drained;
        self.active_quantum = None;
        Ok(())
    }

    pub(crate) fn take_staged_step(&mut self) -> Option<FishS2RetainedStep> {
        self.staged_step.take()
    }

    pub(crate) fn take_managed_write_completions(&mut self) -> Vec<Arc<KvWriteBatchCompletion>> {
        let completions = self.slow_cache.take_completed_writes();
        self.completions_drained = true;
        completions
    }

    pub(crate) const fn slow_position(&self) -> usize {
        self.slow_position
    }

    pub(crate) const fn finished(&self) -> bool {
        self.finished
    }

    pub(crate) fn params(&self) -> &FishS2GenerationParams {
        &self.params
    }

    pub(crate) fn frames_generated(&self) -> usize {
        self.generated_codebooks.first().map(Vec::len).unwrap_or(0)
    }

    fn require_clean_quantum(&self) -> Result<()> {
        if self.active_quantum.is_none() || self.staged_step.is_some() {
            return Err(Error::InferenceError(
                "Fish S2 step requires one clean active quantum".into(),
            ));
        }
        Ok(())
    }

    fn stage(&mut self, step: FishS2RetainedStep) -> Result<FishS2RetainedStep> {
        self.staged_step = Some(step.clone());
        Ok(step)
    }

    fn stage_finished(&mut self) -> Result<FishS2RetainedStep> {
        self.stage(FishS2RetainedStep::Finished {
            frames_generated: self.frames_generated(),
        })
    }

    fn validate_checkpoint(&self, checkpoint: &FishS2RetainedCheckpoint) -> Result<()> {
        if checkpoint.state_id != self.state_id
            || self.active_quantum != Some(checkpoint.quantum)
            || checkpoint.payload.is_none()
        {
            return Err(Error::InferenceError(
                "Fish S2 checkpoint is foreign, stale, or out of order".into(),
            ));
        }
        Ok(())
    }
}

fn slice_prompt(
    prompt: &FishS2ConditioningPrompt,
    start: usize,
    len: usize,
) -> Result<FishS2ConditioningPrompt> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| Error::InvalidInput("Fish S2 prompt slice overflow".into()))?;
    if end > prompt.prompt_length {
        return Err(Error::InvalidInput(
            "Fish S2 prompt slice exceeds prepared artifact".into(),
        ));
    }
    Ok(FishS2ConditioningPrompt {
        values: prompt
            .values
            .iter()
            .map(|row| row[start..end].to_vec())
            .collect(),
        vq_mask: prompt.vq_mask[start..end].to_vec(),
        prompt_length: len,
    })
}

fn preparation_memory_for_geometry(
    config: &super::FishS2Config,
    text_bytes: usize,
    reference_text_bytes: usize,
    input_samples: usize,
    input_capacity: usize,
    sample_rate: u32,
    context_limit: usize,
) -> Result<FishS2PreparationMemory> {
    let context = context_limit.min(config.max_seq_len);
    let max_prompt = context
        .checked_sub(1)
        .filter(|limit| *limit > 0)
        .ok_or_else(|| {
            Error::InvalidInput(
                "Fish S2 effective context requires a prompt and generation position".into(),
            )
        })?;
    let dac = super::FishS2DacConfig::current();
    let frames = dac.reference_frame_count(input_samples, sample_rate)?;
    if frames > max_prompt {
        return Err(Error::InvalidInput(
            "Fish S2 reference frames exceed effective prompt context".into(),
        ));
    }
    let prepared_samples = frames
        .checked_mul(dac.samples_per_frame()?)
        .ok_or_else(|| Error::Overloaded("Fish S2 prepared audio length overflow".into()))?;
    let rendered_bytes = text_bytes
        .checked_add(reference_text_bytes)
        .and_then(|n| n.checked_add(512))
        .ok_or_else(|| Error::Overloaded("Fish S2 rendered text length overflow".into()))?;
    // The native byte-level tokenizer's Encoding owns ids, offsets, token
    // strings and normalization/pretokenization scratch. Price these separately
    // from the dense prompt, using a conservative per-UTF8-byte envelope plus
    // fixed small-allocation allowance, as for codec workspace.
    let tokenization = allocation_bytes(rendered_bytes, 256)?;
    let code_elements = frames
        .checked_mul(config.num_codebooks)
        .ok_or_else(|| Error::Overloaded("Fish S2 code readback shape overflow".into()))?;
    let host_transient_bytes = checked_bytes_sum(&[
        allocation_bytes(input_capacity, 4)?,
        allocation_bytes(reference_text_bytes, 1)?,
        allocation_bytes(prepared_samples, 4 * 3)?,
        super::codec::fft_workspace(
            u64::try_from(input_samples)
                .map_err(|_| Error::Overloaded("Fish S2 input samples overflow".into()))?,
            sample_rate,
        )?,
        allocation_bytes(code_elements, 4 * 3)?,
        allocation_bytes(max_prompt, 4)?,
        tokenization,
        16 * 1024 * 1024,
    ])?;
    Ok(FishS2PreparationMemory {
        host_transient_bytes,
        retained_artifact_bytes: artifact_bound(config.num_codebooks, max_prompt)?,
        accelerator_workspace_bytes: super::codec::preparation_workspace_bytes(
            input_samples,
            sample_rate,
        )?,
    })
}

fn validate_preparation_inputs(text: &str, reference: &FishS2Reference) -> Result<()> {
    if text.trim().is_empty() {
        return Err(Error::InvalidInput(
            "Fish S2 TTS text input cannot be empty".into(),
        ));
    }
    if reference.text.trim().is_empty() || reference.audio_samples.is_empty() {
        return Err(Error::InvalidInput(
            "Fish S2 reference text and audio cannot be empty".into(),
        ));
    }
    Ok(())
}

fn next_state_id() -> Result<u64> {
    NEXT_FISH_S2_STATE_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::InferenceError("Fish S2 state identity space exhausted".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::fish_s2::physical::test_physical_cache;

    #[test]
    fn managed_generation_budget_uses_logical_context_not_first_chunk() {
        let model = FishS2TtsModel::for_test();
        for (prompt, context, requested, expected) in [
            (224, 8192, 512, 512),
            (160, 8192, 512, 512),
            (224, 256, 512, 32),
        ] {
            let (mut state, mut checkpoint) = model
                .new_retained_state_in_quantum(
                    FishS2PreparedArtifact::test_prompt(11, prompt),
                    FishS2GenerationParams {
                        max_frames: requested,
                        ..Default::default()
                    },
                    test_physical_cache(91, 1, 1, 1, 192),
                    context,
                )
                .unwrap();
            assert_eq!(state.max_frames, expected);
            assert_eq!(state.slow_cache.capacity_tokens(), 192);
            state.take_managed_write_completions();
            state.commit_managed_quantum(&mut checkpoint).unwrap();
            let _checkpoint = state
                .begin_managed_quantum(test_physical_cache(91, 1, 1, 1, 256))
                .unwrap();
            assert_eq!(state.slow_cache.capacity_tokens(), 256);
            assert_eq!(state.max_frames, expected);
        }
    }

    #[test]
    fn managed_generation_rejects_prompt_at_actual_context_limit() {
        let model = FishS2TtsModel::for_test();
        for context in [192, 224] {
            let error = model
                .new_retained_state_in_quantum(
                    FishS2PreparedArtifact::test_prompt(11, 224),
                    FishS2GenerationParams::default(),
                    test_physical_cache(91, 1, 1, 1, 256),
                    context,
                )
                .err()
                .expect("actual context must leave output room");
            assert!(error.to_string().contains("leaves no output room"));
        }
    }

    fn state() -> FishS2RetainedState {
        FishS2RetainedState {
            state_id: 7,
            model_identity: 9,
            artifact: Arc::new(FishS2PreparedArtifact {
                model_identity: 9,
                prompt: FishS2ConditioningPrompt {
                    values: vec![vec![1, 2, 3], vec![4, 5, 6]],
                    vq_mask: vec![false, true, true],
                    prompt_length: 3,
                },
                reference_encode_ms: 0.0,
                prompt_build_ms: 0.0,
            }),
            params: FishS2GenerationParams::default(),
            slow_cache: test_physical_cache(91, 1, 1, 1, 8),
            slow_position: 0,
            slow_output: None,
            semantic_sampler: FishS2SemanticSampler::new(0.8, 0.8, 0),
            fast_sampler: FishS2Sampler::new(0.8, 0.8, 1),
            generated_codebooks: vec![Vec::new(), Vec::new()],
            recent_semantic_tokens: Vec::new(),
            max_frames: 4,
            stop_reason: "max_frames".into(),
            finished: false,
            active_quantum: None,
            next_quantum: 1,
            staged_step: None,
            completions_drained: true,
        }
    }

    #[test]
    fn artifact_observation_counts_spare_capacity_and_arc_once() {
        let mut artifact = (*FishS2PreparedArtifact::test_prompt(11, 17)).clone();
        artifact.prompt.values[0].reserve(31);
        artifact.prompt.vq_mask.reserve(31);
        let expected = std::mem::size_of::<FishS2PreparedArtifact>()
            + 2 * std::mem::size_of::<usize>()
            + artifact.prompt.values.capacity() * std::mem::size_of::<Vec<u32>>()
            + artifact
                .prompt
                .values
                .iter()
                .map(|row| row.capacity() * 4)
                .sum::<usize>()
            + artifact.prompt.vq_mask.capacity();
        let shared = Arc::new(artifact);
        let clone = shared.clone();
        assert_eq!(shared.retained_bytes().unwrap(), expected as u64);
        assert_eq!(
            clone.retained_bytes().unwrap(),
            shared.retained_bytes().unwrap()
        );
        assert!(shared.retained_bytes().unwrap() > artifact_bound(10, 17).unwrap());
    }

    #[test]
    fn artifact_context_envelope_covers_exact_dense_allocations() {
        let config = super::super::config::current_config();
        for tokens in [1, 17, config.max_seq_len - 1] {
            let artifact = FishS2PreparedArtifact::test_prompt(config.num_codebooks + 1, tokens);
            assert_eq!(
                artifact.retained_bytes().unwrap(),
                artifact_bound(config.num_codebooks, tokens).unwrap()
            );
        }
    }

    #[test]
    fn preparation_contract_prices_native_resampled_and_capacity_bytes() {
        let config = super::super::config::current_config();
        let native =
            preparation_memory_for_geometry(&config, 32, 64, 44_100, 44_100, 44_100, 1024).unwrap();
        let spare =
            preparation_memory_for_geometry(&config, 32, 64, 44_100, 44_200, 44_100, 1024).unwrap();
        let resampled =
            preparation_memory_for_geometry(&config, 32, 64, 16_000, 16_000, 16_000, 1024).unwrap();
        assert_eq!(
            spare.host_transient_bytes - native.host_transient_bytes,
            400
        );
        assert_eq!(
            native.retained_artifact_bytes,
            resampled.retained_artifact_bytes
        );
        assert!(resampled.host_transient_bytes > 16 * 1024 * 1024);
        assert_eq!(
            native.host_bytes().unwrap(),
            native.host_transient_bytes + native.retained_artifact_bytes
        );
    }

    #[test]
    fn preparation_contract_rejects_invalid_geometry_and_overflow() {
        let config = super::super::config::current_config();
        for (samples, rate, context) in [
            (0, 44_100, 1024),
            (1, 0, 1024),
            (1, u32::MAX, 1024),
            (44_100, 44_100, 2),
            (1, 44_100, 1),
            (usize::MAX, 1, 1024),
        ] {
            assert!(preparation_memory_for_geometry(
                &config, 1, 1, samples, samples, rate, context
            )
            .is_err());
        }
        assert!(
            preparation_memory_for_geometry(&config, usize::MAX, 1, 1, 1, 44_100, 1024).is_err()
        );
        assert!(
            preparation_memory_for_geometry(&config, 1, 1, 1, usize::MAX, 44_100, 1024).is_err()
        );
        assert!(artifact_bound(usize::MAX, 1).is_err());
        assert!(artifact_bound(10, usize::MAX).is_err());
        assert!(FishS2PreparationMemory {
            host_transient_bytes: u64::MAX,
            retained_artifact_bytes: 1,
            accelerator_workspace_bytes: 0
        }
        .host_bytes()
        .is_err());
    }

    #[test]
    fn prepared_prompt_slices_preserve_codebook_rows_and_mask() {
        let state = state();
        let slice = slice_prompt(&state.artifact.prompt, 1, 2).unwrap();
        assert_eq!(slice.values, vec![vec![2, 3], vec![5, 6]]);
        assert_eq!(slice.vq_mask, vec![true, true]);
        assert_eq!(slice.prompt_length, 2);
    }

    #[test]
    fn managed_rollback_restores_frame_codes_and_sampler_state() {
        let mut state = state();
        let logits =
            candle_core::Tensor::new(&[0.0f32, 0.1, 0.2, 0.3], &candle_core::Device::Cpu).unwrap();
        let mut original = state.fast_sampler.clone();
        let expected = super::super::fast::sample_logits(&logits, &mut original).unwrap();
        let slow = test_physical_cache(91, 1, 1, 1, 8);
        let mut checkpoint = state.begin_managed_quantum(slow).unwrap();
        state.slow_position = 3;
        state.generated_codebooks = vec![vec![1], vec![2]];
        state.recent_semantic_tokens.push(42);
        let _ = super::super::fast::sample_logits(&logits, &mut state.fast_sampler).unwrap();
        state.staged_step = Some(FishS2RetainedStep::Frame {
            frames_generated: 1,
        });
        state.rollback_managed_quantum(&mut checkpoint).unwrap();
        assert_eq!(state.slow_position(), 0);
        assert_eq!(state.frames_generated(), 0);
        assert!(state.recent_semantic_tokens.is_empty());
        assert_eq!(
            super::super::fast::sample_logits(&logits, &mut state.fast_sampler).unwrap(),
            expected
        );
        assert!(state.take_staged_step().is_none());
    }

    #[test]
    fn commit_requires_staged_output_and_write_receipts_to_be_drained() {
        let mut state = state();
        let slow = test_physical_cache(91, 1, 1, 1, 8);
        let mut checkpoint = state.begin_managed_quantum(slow).unwrap();
        state.staged_step = Some(FishS2RetainedStep::Frame {
            frames_generated: 1,
        });
        assert!(state.commit_managed_quantum(&mut checkpoint).is_err());
        state.take_staged_step();
        state.take_managed_write_completions();
        state.commit_managed_quantum(&mut checkpoint).unwrap();
    }
}
