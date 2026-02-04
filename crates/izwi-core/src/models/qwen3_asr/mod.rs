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

        let tokenizer = AsrTokenizer::load(model_dir, config.thinker_config.text_config.vocab_size)?;
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
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device.device)
        }?;
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

    pub fn transcribe(&self, audio: &[f32], sample_rate: u32, language: Option<&str>) -> Result<String> {
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

        let audio_embeds = self.audio_tower.forward(&mel)?; // [1, t, hidden]
        let audio_len = audio_embeds.dim(1)?;

        let prompt = self.build_prompt(audio_len, language)?;
        let input_ids = Tensor::from_vec(prompt.ids.clone(), (1, prompt.ids.len()), &self.device.device)?;

        let mut cache = Qwen3Cache::new(self.text_model.num_layers());
        let mut embeds = self.forward_with_audio(&input_ids, &audio_embeds, prompt.audio_start, &mut cache)?;

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
            embeds = self.text_model.forward(&next_tensor, pos, Some(&mut cache))?;
            pos += 1;
        }

        let text = self.tokenizer.decode_text(&generated)?;
        Ok(text.trim().to_string())
    }

    fn forward_with_audio(
        &self,
        input_ids: &Tensor,
        audio_embeds: &Tensor,
        audio_start: usize,
        cache: &mut Qwen3Cache,
    ) -> Result<Tensor> {
        let embeds = self.text_model.embeddings(input_ids)?;

        // Replace the single audio placeholder token with the full audio embedding sequence.
        let before = embeds.narrow(1, 0, audio_start)?;
        let after_start = audio_start + 1;
        let after_len = embeds.dim(1)? - after_start;
        let after = embeds.narrow(1, after_start, after_len)?;

        let embeds = Tensor::cat(&[before, audio_embeds.clone(), after], 1)?;
        let position_ids = if self.text_model.uses_mrope() {
            Some(self.build_position_ids(embeds.dim(1)?, audio_start, audio_embeds.dim(1)?)?)
        } else {
            None
        };
        self.text_model
            .forward_with_embeds(&embeds, 0, Some(cache), position_ids.as_ref())
    }

    fn build_prompt(&self, _audio_len: usize, _language: Option<&str>) -> Result<PromptTokens> {
        let mut ids = Vec::new();

        ids.push(self.specials.im_start);
        ids.extend(self.tokenizer.encode_text("system\n")?);
        ids.push(self.specials.im_end);
        ids.extend(self.tokenizer.encode_text("\n")?);

        ids.push(self.specials.im_start);
        ids.extend(self.tokenizer.encode_text("user\n")?);

        ids.push(self.specials.audio_start);
        ids.push(self.specials.audio_token);
        ids.push(self.specials.audio_end);

        ids.push(self.specials.im_end);
        ids.extend(self.tokenizer.encode_text("\n")?);
        ids.push(self.specials.im_start);
        ids.extend(self.tokenizer.encode_text("assistant\n")?);

        let audio_start = ids
            .iter()
            .position(|id| *id == self.specials.audio_token)
            .unwrap_or(0);

        Ok(PromptTokens { ids, audio_start })
    }

    fn build_position_ids(&self, seq_len: usize, audio_start: usize, audio_len: usize) -> Result<Tensor> {
        let mut axis0 = Vec::with_capacity(seq_len);
        let mut axis1 = Vec::with_capacity(seq_len);
        let mut axis2 = Vec::with_capacity(seq_len);

        for idx in 0..seq_len {
            axis0.push(idx as i64);
            if idx >= audio_start && idx < audio_start + audio_len {
                axis1.push(0);
                axis2.push(0);
            } else {
                axis1.push(idx as i64);
                axis2.push(idx as i64);
            }
        }

        let mut data = Vec::with_capacity(3 * seq_len);
        data.extend(axis0);
        data.extend(axis1);
        data.extend(axis2);

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
