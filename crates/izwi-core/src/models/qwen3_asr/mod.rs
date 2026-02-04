//! Native Qwen3-ASR model loader and inference.

mod audio;
mod config;
mod tokenizer;

use std::path::Path;

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use tracing::info;

use crate::audio::{MelConfig, MelSpectrogram};
use crate::error::{Error, Result};
use crate::models::device::DeviceProfile;
use crate::models::qwen3::{Qwen3Cache, Qwen3Model};
use crate::models::qwen3_asr::config::AudioConfig;

use audio::AudioTower;
use config::Qwen3AsrConfig;
use tokenizer::{AsrTokenizer, SpecialTokenIds};

#[derive(Debug, Deserialize)]
struct PreprocessorConfig {
    #[serde(default)]
    feature_size: usize,
    #[serde(default)]
    n_fft: usize,
    #[serde(default)]
    hop_length: usize,
    #[serde(default)]
    n_samples: usize,
    #[serde(default)]
    nb_max_frames: usize,
}

pub struct Qwen3AsrModel {
    device: DeviceProfile,
    dtype: DType,
    tokenizer: AsrTokenizer,
    specials: SpecialTokenIds,
    audio_tower: AudioTower,
    text_model: Qwen3Model,
    mel: MelSpectrogram,
    preprocessor: PreprocessorConfig,
}

impl Qwen3AsrModel {
    pub fn load(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(config_path)?;
        let config: Qwen3AsrConfig = serde_json::from_str(&config_str)?;

        let tokenizer =
            AsrTokenizer::load(model_dir, config.thinker_config.text_config.vocab_size)?;
        let specials = tokenizer.specials().clone();

        let preprocessor: PreprocessorConfig = {
            let path = model_dir.join("preprocessor_config.json");
            let data = std::fs::read_to_string(path)?;
            serde_json::from_str(&data)?
        };

        let mel_cfg = MelConfig {
            sample_rate: 16_000,
            n_fft: preprocessor.n_fft.max(400),
            hop_length: preprocessor.hop_length.max(160),
            n_mels: preprocessor.feature_size.max(128),
            f_min: 0.0,
            f_max: 8_000.0,
            normalize: true,
        };
        let mel = MelSpectrogram::new(mel_cfg)?;

        let dtype = device.select_dtype(config.thinker_config.dtype.as_deref());
        let weights_path = model_dir.join("model.safetensors");
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device.device) }?;
        let vb = vb.pp("thinker");

        let audio_cfg = config.thinker_config.audio_config.clone();
        let audio_tower = AudioTower::load(audio_cfg, vb.pp("audio_tower"))?;
        let text_cfg = config.thinker_config.text_config.clone();
        let text_model = Qwen3Model::load(text_cfg, vb)?;

        info!("Loaded Qwen3-ASR model on {:?}", device.kind);

        Ok(Self {
            device,
            dtype,
            tokenizer,
            specials,
            audio_tower,
            text_model,
            mel,
            preprocessor,
        })
    }

    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        let audio = if sample_rate != 16_000 {
            resample(audio, sample_rate, 16_000)?
        } else {
            audio.to_vec()
        };

        let mut mel_spec = self.mel.compute(&audio)?;
        if self.preprocessor.nb_max_frames > 0 && mel_spec.len() > self.preprocessor.nb_max_frames {
            mel_spec.truncate(self.preprocessor.nb_max_frames);
        }

        let n_mels = self.mel.config().n_mels;
        if mel_spec.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let frames = mel_spec.len();
        let mut flat = Vec::with_capacity(frames * n_mels);
        for frame in mel_spec.iter() {
            flat.extend_from_slice(frame);
        }

        let mel = Tensor::from_vec(flat, (frames, n_mels), &self.device.device)?
            .transpose(0, 1)? // [n_mels, frames]
            .unsqueeze(0)?
            .unsqueeze(0)? // [1, 1, n_mels, frames]
            .to_dtype(self.dtype)?;

        let audio_embeds = self.audio_tower.forward(&mel, None)?; // [1, t, hidden]
        let audio_len = audio_embeds.dim(1)?;

        let prompt = self.build_prompt(audio_len, language)?;
        let input_ids = Tensor::from_vec(
            prompt.ids.clone(),
            (1, prompt.ids.len()),
            &self.device.device,
        )?;

        let mut cache = Qwen3Cache::new(self.text_model.num_layers());
        let mut embeds =
            self.forward_with_audio(&input_ids, &audio_embeds, prompt.audio_start, &mut cache)?;

        let mut generated: Vec<u32> = Vec::new();
        let mut pos = embeds.dim(1)?;

        let max_tokens = 1024usize;
        for _ in 0..max_tokens {
            let logits = embeds.i((0, embeds.dim(1)? - 1))?;
            let next = argmax(&logits)?;
            if next == self.specials.im_end || next == self.specials.eos {
                break;
            }
            generated.push(next);

            let next_tensor = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
            embeds = self
                .text_model
                .forward(&next_tensor, pos, Some(&mut cache))?;
            pos += 1;
        }

        let text = self.tokenizer.decode_text(&generated)?;
        let parsed = self.parse_output(&text)?;
        Ok(parsed)
    }

    /// Parse model output to extract transcription.
    /// Expected formats per document:
    /// - "Language: English\nTranscription: This is the recognized text."
    /// - "No speech detected in the audio."
    fn parse_output(&self, raw_output: &str) -> Result<String> {
        let trimmed = raw_output.trim();

        // Check for "No speech detected"
        if trimmed.contains("No speech detected") {
            return Ok(String::new());
        }

        // Look for "Transcription:" prefix
        if let Some(idx) = trimmed.find("Transcription:") {
            let after = &trimmed[idx + "Transcription:".len()..];
            return Ok(after.trim().to_string());
        }

        // If no Transcription: prefix, return the full text (minus any Language: prefix)
        if let Some(idx) = trimmed.find('\n') {
            let after_newline = &trimmed[idx + 1..];
            if !after_newline.is_empty() {
                return Ok(after_newline.trim().to_string());
            }
        }

        Ok(trimmed.to_string())
    }

    fn forward_with_audio(
        &self,
        input_ids: &Tensor,
        audio_embeds: &Tensor,
        audio_start: usize,
        cache: &mut Qwen3Cache,
    ) -> Result<Tensor> {
        let embeds = self.text_model.embeddings(input_ids)?;

        // audio_start is the position of audio_start token
        // We need to inject audio_embeds between audio_start (position audio_start)
        // and audio_end (position audio_start + 1)
        // So: [tokens before audio_start, audio_start embedding, audio_embeds, tokens after audio_end]

        // Get embeddings before audio_start (if any)
        let before = if audio_start > 0 {
            embeds.narrow(1, 0, audio_start)?
        } else {
            // Empty tensor with correct shape [1, 0, hidden_size]
            Tensor::zeros((1, 0, embeds.dim(2)?), embeds.dtype(), embeds.device())?
        };

        // Get audio_start token embedding (single token)
        let audio_start_embed = embeds.narrow(1, audio_start, 1)?;

        // Get embeddings after audio_end (skip audio_start and audio_end)
        let after_start = audio_start + 2; // Skip audio_start and audio_end tokens
        let after = if after_start < embeds.dim(1)? {
            embeds.narrow(1, after_start, embeds.dim(1)? - after_start)?
        } else {
            // Empty tensor
            Tensor::zeros((1, 0, embeds.dim(2)?), embeds.dtype(), embeds.device())?
        };

        // Concatenate: before + audio_start_embed + audio_embeds + after
        let embeds = Tensor::cat(&[before, audio_start_embed, audio_embeds.clone(), after], 1)?;

        let position_ids = if self.text_model.uses_mrope() {
            Some(self.build_position_ids(embeds.dim(1)?, audio_start, audio_embeds.dim(1)?)?)
        } else {
            None
        };
        self.text_model
            .forward_with_embeds(&embeds, 0, Some(cache), position_ids.as_ref())
    }

    fn build_prompt(&self, _audio_len: usize, language: Option<&str>) -> Result<PromptTokens> {
        let mut ids = Vec::new();

        // Format: <|audio_start|><|audio_end|> or with language: <|audio_start|><|audio_end|>Language: {}
        ids.push(self.specials.audio_start);
        ids.push(self.specials.audio_end);

        if let Some(lang) = language {
            ids.extend(self.tokenizer.encode_text(&format!("Language: {}", lang))?);
        }

        // Find position where audio embeddings will be injected (right after audio_start)
        let audio_start = 0; // Audio embeddings start right after audio_start token

        Ok(PromptTokens { ids, audio_start })
    }

    fn build_position_ids(
        &self,
        seq_len: usize,
        _audio_start: usize,
        audio_len: usize,
    ) -> Result<Tensor> {
        // MRoPE: 3D position IDs for temporal, height, width dimensions
        // Audio embeddings are injected at positions 1 to audio_len (after audio_start token at pos 0)
        // Audio region: positions 1 to audio_len (inclusive of audio embeddings)
        let audio_embed_start = 1; // audio_start token is at position 0, embeddings start at 1
        let audio_embed_end = audio_embed_start + audio_len;

        let mut data = Vec::with_capacity(3 * seq_len);

        // Temporal dimension (dim 0) - continuous positions for all tokens
        for pos in 0..seq_len {
            data.push(pos as i64);
        }

        // Height dimension (dim 1) - 0 for text, audio positions for audio embeddings
        for pos in 0..seq_len {
            if pos >= audio_embed_start && pos < audio_embed_end {
                // Audio embedding positions use their own temporal positions
                data.push((pos - audio_embed_start) as i64);
            } else {
                data.push(0i64);
            }
        }

        // Width dimension (dim 2) - 0 for text, audio positions for audio embeddings
        for pos in 0..seq_len {
            if pos >= audio_embed_start && pos < audio_embed_end {
                // Audio embedding positions use their own temporal positions
                data.push((pos - audio_embed_start) as i64);
            } else {
                data.push(0i64);
            }
        }

        Tensor::from_vec(data, (3, seq_len), &self.device.device).map_err(Error::from)
    }
}

struct PromptTokens {
    ids: Vec<u32>,
    audio_start: usize,
}

fn resample(audio: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == dst_rate {
        return Ok(audio.to_vec());
    }

    let ratio = dst_rate as f32 / src_rate as f32;
    let out_len = ((audio.len() as f32) * ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f32 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f32;
        let s0 = *audio.get(idx).unwrap_or(&0.0);
        let s1 = *audio.get(idx + 1).unwrap_or(&s0);
        out.push(s0 + frac * (s1 - s0));
    }
    Ok(out)
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    let values = logits.to_vec1::<f32>()?;
    let mut max_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;
    for (idx, &val) in values.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }
    Ok(max_idx as u32)
}
