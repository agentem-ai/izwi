//! TTS request batching helper.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot, RwLock};

use crate::config::EngineConfig;
use crate::error::{Error, Result};
use crate::models::qwen3_tts::{BatchedSpeakerRequest, Qwen3TtsModel, TtsGenerationParams};
use crate::runtime::types::{GenerationRequest, GenerationResult};

pub struct TtsBatcher {
    tx: mpsc::Sender<BatchItem>,
}

struct BatchItem {
    request: GenerationRequest,
    respond_to: oneshot::Sender<Result<GenerationResult>>,
}

impl TtsBatcher {
    pub fn new(config: EngineConfig, model: Arc<RwLock<Option<Qwen3TtsModel>>>) -> Self {
        let max_batch = config.max_batch_size.max(1);
        let (tx, rx) = mpsc::channel(max_batch * 4);
        tokio::spawn(run_batcher(rx, model, config, max_batch));
        Self { tx }
    }

    pub async fn submit(&self, request: GenerationRequest) -> Result<GenerationResult> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(BatchItem {
                request,
                respond_to: tx,
            })
            .await
            .map_err(|_| Error::InferenceError("Batcher is offline".to_string()))?;
        rx.await
            .map_err(|_| Error::InferenceError("Batcher dropped response".to_string()))?
    }
}

async fn run_batcher(
    mut rx: mpsc::Receiver<BatchItem>,
    model: Arc<RwLock<Option<Qwen3TtsModel>>>,
    config: EngineConfig,
    max_batch: usize,
) {
    let batch_window = Duration::from_millis(4);

    while let Some(first) = rx.recv().await {
        let mut batch = vec![first];
        let deadline = Instant::now() + batch_window;

        while batch.len() < max_batch {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let timeout = deadline.saturating_duration_since(now);
            match tokio::time::timeout(timeout, rx.recv()).await {
                Ok(Some(item)) => batch.push(item),
                _ => break,
            }
        }

        let start = Instant::now();
        let model = model.clone();
        let config = config.clone();

        let mut requests = Vec::with_capacity(batch.len());
        let mut senders = Vec::with_capacity(batch.len());
        for item in batch {
            requests.push(item.request);
            senders.push(item.respond_to);
        }

        let handle = tokio::task::spawn_blocking(move || {
            process_batch_requests(requests, model, config, start.elapsed())
        });

        match handle.await {
            Ok(results) => {
                for (sender, result) in senders.into_iter().zip(results.into_iter()) {
                    let _ = sender.send(result);
                }
            }
            Err(_) => {
                let err_msg = "Batch worker failed".to_string();
                for sender in senders {
                    let _ = sender.send(Err(Error::InferenceError(err_msg.clone())));
                }
            }
        }
    }
}

fn process_batch_requests(
    batch: Vec<GenerationRequest>,
    model: Arc<RwLock<Option<Qwen3TtsModel>>>,
    _config: EngineConfig,
    elapsed: Duration,
) -> Vec<Result<GenerationResult>> {
    let elapsed_ms = elapsed.as_secs_f32() * 1000.0;

    let rt = tokio::runtime::Handle::try_current();
    let model_guard = rt
        .as_ref()
        .map(|r| r.block_on(async { model.read().await }))
        .unwrap_or_else(|_| panic!("No async runtime available"));

    let model = match model_guard.as_ref() {
        Some(model) => model,
        None => {
            let err_msg = "No TTS model loaded".to_string();
            return batch
                .into_iter()
                .map(|_| Err(Error::InferenceError(err_msg.clone())))
                .collect();
        }
    };

    let available_speakers = model.available_speakers();

    let mut results: Vec<Option<Result<GenerationResult>>> = (0..batch.len()).map(|_| None).collect();
    let mut batch_inputs = Vec::new();
    let mut batch_indices = Vec::new();

    for (idx, item) in batch.iter().enumerate() {
        let params = TtsGenerationParams::from_generation_config(&item.config);
        if available_speakers.is_empty() {
            let result = model
                .generate_with_text_params(
                    &item.text,
                    Some("Auto"),
                    item.voice_description.as_deref(),
                    &params,
                )
                .map(|samples| GenerationResult {
                    request_id: item.id.clone(),
                    total_tokens: samples.len() / 256,
                    total_time_ms: elapsed_ms,
                    sample_rate: 24000,
                    samples,
                });
            results[idx] = Some(result);
            continue;
        }

        let requested = item
            .config
            .speaker
            .as_deref()
            .unwrap_or_else(|| available_speakers[0].as_str());
        let speaker = match available_speakers
            .iter()
            .find(|s| s.eq_ignore_ascii_case(requested))
        {
            Some(name) => name.as_str(),
            None => {
                let speaker_list = available_speakers
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let err = Error::InvalidInput(format!(
                    "Unknown speaker '{requested}'. Available speakers: {}",
                    speaker_list
                ));
                results[idx] = Some(Err(err));
                continue;
            }
        };

        batch_indices.push(idx);
        batch_inputs.push(BatchedSpeakerRequest {
            text: item.text.clone(),
            speaker: speaker.to_string(),
            language: Some("Auto".to_string()),
            instruct: item.voice_description.clone(),
            params,
        });
    }

    if !batch_inputs.is_empty() {
        match model.generate_with_speaker_params_batch(&batch_inputs) {
            Ok(samples_list) => {
                for (idx, samples) in batch_indices.into_iter().zip(samples_list.into_iter()) {
                    results[idx] = Some(Ok(GenerationResult {
                        request_id: batch[idx].id.clone(),
                        total_tokens: samples.len() / 256,
                        total_time_ms: elapsed_ms,
                        sample_rate: 24000,
                        samples,
                    }));
                }
            }
            Err(err) => {
                let err_msg = err.to_string();
                for idx in batch_indices {
                    results[idx] = Some(Err(Error::InferenceError(err_msg.clone())));
                }
            }
        }
    }

    results
        .into_iter()
        .map(|result| {
            result.unwrap_or_else(|| Err(Error::InferenceError("Batch result missing".to_string())))
        })
        .collect()
}
