use tokio::sync::mpsc;

use crate::error::{Error, Result};

use super::super::output::StreamingOutput;
use super::super::request::EngineCoreRequest;
use super::NativeExecutor;

impl NativeExecutor {
    pub(super) fn stream_sender(
        request: &EngineCoreRequest,
    ) -> Option<mpsc::UnboundedSender<StreamingOutput>> {
        if request.streaming {
            request.streaming_tx.clone()
        } else {
            None
        }
    }

    pub(super) fn stream_text(
        tx: &mpsc::UnboundedSender<StreamingOutput>,
        request_id: &str,
        sequence: &mut usize,
        text: String,
    ) -> Result<()> {
        tx.send(StreamingOutput {
            request_id: request_id.to_string(),
            sequence: *sequence,
            samples: Vec::new(),
            sample_rate: 0,
            is_final: false,
            text: Some(text),
            stats: None,
        })
        .map_err(|_| Error::InferenceError("Streaming output channel closed".to_string()))?;
        *sequence += 1;
        Ok(())
    }

    pub(super) fn stream_audio(
        tx: &mpsc::UnboundedSender<StreamingOutput>,
        request_id: &str,
        sequence: &mut usize,
        samples: Vec<f32>,
        sample_rate: u32,
        is_final: bool,
    ) -> Result<()> {
        tx.send(StreamingOutput {
            request_id: request_id.to_string(),
            sequence: *sequence,
            samples,
            sample_rate,
            is_final,
            text: None,
            stats: None,
        })
        .map_err(|_| Error::InferenceError("Streaming output channel closed".to_string()))?;
        *sequence += 1;
        Ok(())
    }

    pub(super) fn stream_final_marker(
        tx: &mpsc::UnboundedSender<StreamingOutput>,
        request_id: &str,
        sequence: &mut usize,
    ) -> Result<()> {
        Self::stream_audio(tx, request_id, sequence, Vec::new(), 0, true)
    }
}
