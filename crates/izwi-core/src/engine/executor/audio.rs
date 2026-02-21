use izwi_asr_toolkit::{plan_audio_chunks, AsrLongFormConfig, AudioChunk, TranscriptAssembler};
use tokio::sync::mpsc;
use tracing::debug;

use crate::error::{Error, Result};
use crate::runtime::audio_io::{base64_decode, decode_audio_bytes};

use super::super::output::StreamingOutput;
use super::NativeExecutor;

impl NativeExecutor {
    pub(super) fn env_f32(key: &str) -> Option<f32> {
        std::env::var(key)
            .ok()
            .and_then(|raw| raw.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
    }

    pub(super) fn asr_long_form_config() -> AsrLongFormConfig {
        let mut cfg = AsrLongFormConfig::default();
        if let Some(v) = Self::env_f32("IZWI_ASR_CHUNK_TARGET_SECS") {
            cfg.target_chunk_secs = v;
        }
        if let Some(v) = Self::env_f32("IZWI_ASR_CHUNK_MAX_SECS") {
            cfg.hard_max_chunk_secs = v;
        }
        if let Some(v) = Self::env_f32("IZWI_ASR_CHUNK_OVERLAP_SECS") {
            cfg.overlap_secs = v;
        }
        cfg
    }

    pub(super) fn asr_chunk_plan(
        samples: &[f32],
        sample_rate: u32,
        model_max_chunk_secs: Option<f32>,
    ) -> (AsrLongFormConfig, Vec<AudioChunk>) {
        let cfg = Self::asr_long_form_config();
        let tuned_limit = model_max_chunk_secs
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| (v * 0.95).max(cfg.min_chunk_secs.max(1.0)));
        let chunks = plan_audio_chunks(samples, sample_rate, &cfg, tuned_limit);
        (cfg, chunks)
    }

    pub(super) fn transcribe_with_chunk_plan<F>(
        request_id: &str,
        stream_tx: Option<&mpsc::UnboundedSender<StreamingOutput>>,
        sequence: &mut usize,
        samples: &[f32],
        sample_rate: u32,
        chunk_plan: &[AudioChunk],
        chunk_cfg: &AsrLongFormConfig,
        mut transcribe_chunk: F,
    ) -> Result<String>
    where
        F: FnMut(&[f32], u32) -> Result<String>,
    {
        if chunk_plan.is_empty() {
            return Err(Error::InvalidInput(
                "ASR chunk planner produced no chunks".to_string(),
            ));
        }

        debug!(
            "ASR long-form chunking enabled for request {}: {} chunks (~{:.1}s audio)",
            request_id,
            chunk_plan.len(),
            samples.len() as f32 / sample_rate.max(1) as f32
        );

        let mut assembler = TranscriptAssembler::new(chunk_cfg.clone());
        for chunk in chunk_plan {
            if chunk.end_sample <= chunk.start_sample || chunk.end_sample > samples.len() {
                continue;
            }
            let chunk_audio = &samples[chunk.start_sample..chunk.end_sample];
            let chunk_text = transcribe_chunk(chunk_audio, sample_rate)?;
            let delta = assembler.push_chunk_text(&chunk_text);
            if !delta.is_empty() {
                if let Some(tx) = stream_tx {
                    Self::stream_text(tx, request_id, sequence, delta)?;
                }
            }
        }

        if let Some(tx) = stream_tx {
            Self::stream_final_marker(tx, request_id, sequence)?;
        }

        Ok(assembler.finish())
    }
}

pub(super) fn decode_audio_base64_with_rate(audio_b64: &str) -> Result<(Vec<f32>, u32)> {
    let audio_bytes = base64_decode(audio_b64)?;
    decode_audio_bytes(&audio_bytes)
}
