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
            n_fft: preprocessor.n_fft,
            hop_length: preprocessor.hop_length,
            n_mels: preprocessor.feature_size,
            f_min: 0.0,
            f_max: 8_000.0,
            normalize: true,
        };
        let mel = MelSpectrogram::new(mel_cfg)?;

        // ASR decoding quality is sensitive to precision; prefer fp32 for stability.
        let dtype = DType::F32;

        // Check for sharded weights (1.7B model) vs single file (0.6B model)
        let index_path = model_dir.join("model.safetensors.index.json");
        let vb = if index_path.exists() {
            // Load sharded weights
            let index_data = std::fs::read_to_string(&index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_data)?;

            // Collect unique shard files from the index
            let weight_map = index
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| {
                    Error::InvalidInput("Invalid model.safetensors.index.json format".to_string())
                })?;

            let mut shard_files: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shard_files.sort();
            shard_files.dedup();

            let shard_paths: Vec<std::path::PathBuf> =
                shard_files.iter().map(|f| model_dir.join(f)).collect();

            info!(
                "Loading sharded ASR model with {} shard files",
                shard_paths.len()
            );
            unsafe { VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, &device.device)? }
        } else {
            // Load single file
            let weights_path = model_dir.join("model.safetensors");
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device.device)? }
        };
        let vb = if vb.contains_tensor("thinker.audio_tower.conv2d1.weight") {
            vb.pp("thinker")
        } else {
            vb
        };

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

        let feature_lens = vec![frames];
        let audio_embeds = self.audio_tower.forward(&mel, Some(&feature_lens))?; // [1, t, hidden]
        let audio_len = audio_embeds.dim(1)?;

        let prompt = self.build_prompt(audio_len, language)?;
        let input_ids = Tensor::from_vec(
            prompt.ids.clone(),
            (1, prompt.ids.len()),
            &self.device.device,
        )?;

        let mut cache = Qwen3Cache::new(self.text_model.num_layers());
        let mut embeds = self.forward_with_audio(
            &input_ids,
            &audio_embeds,
            prompt.audio_pad_start,
            prompt.audio_pad_len,
            &mut cache,
        )?;

        let base_pos = embeds.dim(1)?;
        let mut pos = base_pos;

        let mut generated: Vec<u32> = Vec::new();

        // Long repetitive generations are almost always degenerate for ASR.
        // Keep this bounded to reduce latency and prevent runaway gibberish.
        let max_tokens = 256usize;
        for _ in 0..max_tokens {
            // Get logits for the last position only
            let logits = embeds.i((0, embeds.dim(1)? - 1))?; // [vocab_size]
            let next = argmax(&logits)?;

            if next == self.specials.im_end
                || next == self.specials.eos
                || self.specials.eos_alt == Some(next)
            {
                break;
            }
            generated.push(next);

            // Forward pass for next token with updated position
            let next_tensor = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
            if self.text_model.uses_mrope() {
                let next_embeds = self.text_model.embeddings(&next_tensor)?;
                let position_ids = self.build_position_ids(1, pos, None)?;
                embeds = self.text_model.forward_with_embeds(
                    &next_embeds,
                    pos,
                    Some(&mut cache),
                    Some(&position_ids),
                )?;
            } else {
                embeds = self
                    .text_model
                    .forward(&next_tensor, pos, Some(&mut cache))?;
            }
            pos += 1;
        }

        let text = self.tokenizer.decode_text(&generated)?;
        let parsed = self.parse_output(&text)?;
        Ok(parsed)
    }

    /// Forced alignment: align reference text with audio timestamps.
    /// Returns a vector of (word, start_time_ms, end_time_ms) tuples.
    pub fn force_align(
        &self,
        audio: &[f32],
        sample_rate: u32,
        reference_text: &str,
    ) -> Result<Vec<(String, u32, u32)>> {
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
            .transpose(0, 1)?
            .unsqueeze(0)?
            .unsqueeze(0)?; // [1, 1, n_mels, frames]
                            // NOTE: Keeping mel as F32

        let feature_lens = vec![frames];
        let audio_embeds = self.audio_tower.forward(&mel, Some(&feature_lens))?;
        let audio_len = audio_embeds.dim(1)?;

        // Build alignment prompt with reference text
        let prompt = self.build_alignment_prompt(audio_len, reference_text)?;
        let input_ids = Tensor::from_vec(
            prompt.ids.clone(),
            (1, prompt.ids.len()),
            &self.device.device,
        )?;

        let mut cache = Qwen3Cache::new(self.text_model.num_layers());
        let mut embeds = self.forward_with_audio(
            &input_ids,
            &audio_embeds,
            prompt.audio_pad_start,
            prompt.audio_pad_len,
            &mut cache,
        )?;

        let mut pos = embeds.dim(1)?;

        let mut generated: Vec<u32> = Vec::new();

        let max_tokens = 2048usize;
        for _ in 0..max_tokens {
            let logits = embeds.i((0, embeds.dim(1)? - 1))?;
            let next = argmax(&logits)?;

            if next == self.specials.im_end || next == self.specials.eos {
                break;
            }
            generated.push(next);

            let next_tensor = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
            if self.text_model.uses_mrope() {
                let next_embeds = self.text_model.embeddings(&next_tensor)?;
                let position_ids = self.build_position_ids(1, pos, None)?;
                embeds = self.text_model.forward_with_embeds(
                    &next_embeds,
                    pos,
                    Some(&mut cache),
                    Some(&position_ids),
                )?;
            } else {
                embeds = self
                    .text_model
                    .forward(&next_tensor, pos, Some(&mut cache))?;
            }
            pos += 1;
        }

        let alignment_text = self.tokenizer.decode_text(&generated)?;
        self.parse_alignment(&alignment_text, audio.len() as u32 / 16)
    }

    fn parse_output(&self, raw_output: &str) -> Result<String> {
        let trimmed = collapse_whitespace(raw_output.trim());
        if trimmed.is_empty() {
            return Ok(String::new());
        }

        if trimmed.contains("No speech detected") {
            return Ok(String::new());
        }

        let mut text = trimmed;

        // Qwen3-ASR commonly emits: "language English<asr_text>..."
        // Split on the last tag in case intermediate tags are present.
        if let Some((_, tail)) = text.rsplit_once("<asr_text>") {
            text = tail.to_string();
        }

        // Backward compatibility with older output styles.
        if let Some(idx) = text.find("Transcription:") {
            text = text[idx + "Transcription:".len()..].to_string();
        }

        if text.to_lowercase().starts_with("language ") {
            if let Some(newline_idx) = text.find('\n') {
                text = text[newline_idx + 1..].to_string();
            } else {
                let mut parts = text.splitn(3, ' ');
                let _ = parts.next(); // "language"
                let _ = parts.next(); // language name
                text = parts.next().unwrap_or_default().to_string();
            }
        }

        if text == "<non_speech>" {
            return Ok(String::new());
        }

        text = text.replace("<|im_end|>", "");
        text = collapse_whitespace(text.trim());
        text = strip_runaway_repetition(&text);
        Ok(text.trim().to_string())
    }

    fn forward_with_audio(
        &self,
        input_ids: &Tensor,
        audio_embeds: &Tensor,
        audio_pad_start: usize,
        audio_pad_len: usize,
        cache: &mut Qwen3Cache,
    ) -> Result<Tensor> {
        let embeds = self.text_model.embeddings(input_ids)?;
        let seq_len = embeds.dim(1)?;
        let model_audio_len = audio_embeds.dim(1)?;
        if audio_pad_len == 0 {
            return Err(Error::InvalidInput(
                "Audio placeholder length must be at least 1".to_string(),
            ));
        }
        if model_audio_len != audio_pad_len {
            return Err(Error::InvalidInput(format!(
                "Audio placeholder mismatch: prompt has {audio_pad_len}, embeddings have {model_audio_len}"
            )));
        }

        if audio_pad_start + audio_pad_len > seq_len {
            return Err(Error::InvalidInput(
                "Audio placeholder span is out of prompt bounds".to_string(),
            ));
        }

        // Replace the contiguous <|audio_pad|> span with projected audio embeddings.
        let before = if audio_pad_start > 0 {
            embeds.narrow(1, 0, audio_pad_start)?
        } else {
            Tensor::zeros((1, 0, embeds.dim(2)?), embeds.dtype(), embeds.device())?
        };

        let after_start = audio_pad_start + audio_pad_len;
        let after = if after_start < seq_len {
            embeds.narrow(1, after_start, seq_len - after_start)?
        } else {
            Tensor::zeros((1, 0, embeds.dim(2)?), embeds.dtype(), embeds.device())?
        };

        let embeds = Tensor::cat(&[before, audio_embeds.clone(), after], 1)?;

        let position_ids = if self.text_model.uses_mrope() {
            Some(self.build_position_ids(
                embeds.dim(1)?,
                0,
                Some((audio_pad_start, audio_pad_len)),
            )?)
        } else {
            None
        };
        self.text_model
            .forward_with_embeds(&embeds, 0, Some(cache), position_ids.as_ref())
    }

    fn build_prompt(&self, audio_len: usize, language: Option<&str>) -> Result<PromptTokens> {
        // ASR generation prompt with explicit transcription trigger.
        let mut ids = Vec::new();
        ids.push(self.specials.im_start);
        ids.extend(self.tokenizer.encode_text("user\n")?);
        ids.push(self.specials.audio_start);

        let audio_pad_start = ids.len();
        ids.extend(std::iter::repeat_n(self.specials.audio_token, audio_len));

        ids.push(self.specials.audio_end);
        if let Some(lang) = language {
            let lang = normalized_language_name(lang);
            ids.extend(
                self.tokenizer
                    .encode_text(&format!("Transcribe the audio into {lang}."))?,
            );
        }
        ids.push(self.specials.im_end);
        ids.extend(self.tokenizer.encode_text("\n")?);
        ids.push(self.specials.im_start);
        ids.extend(self.tokenizer.encode_text("assistant\n")?);
        let forced_lang = normalized_language_name(language.unwrap_or("Auto"));
        ids.extend(
            self.tokenizer
                .encode_text(&format!(" language {forced_lang}<asr_text>"))?,
        );

        Ok(PromptTokens {
            ids,
            audio_pad_start,
            audio_pad_len: audio_len,
        })
    }

    fn build_alignment_prompt(
        &self,
        audio_len: usize,
        reference_text: &str,
    ) -> Result<PromptTokens> {
        let mut ids = Vec::new();

        ids.push(self.specials.im_start);
        ids.extend(self.tokenizer.encode_text("system\n")?);
        ids.push(self.specials.im_end);
        ids.extend(self.tokenizer.encode_text("\n")?);

        ids.push(self.specials.im_start);
        ids.extend(self.tokenizer.encode_text("user\n")?);

        ids.push(self.specials.audio_start);
        let audio_pad_start = ids.len();
        ids.extend(std::iter::repeat_n(self.specials.audio_token, audio_len));
        ids.push(self.specials.audio_end);
        ids.extend(
            self.tokenizer
                .encode_text(&format!("Reference: {}\n", reference_text))?,
        );

        ids.push(self.specials.im_end);
        ids.extend(self.tokenizer.encode_text("\n")?);

        ids.push(self.specials.im_start);
        ids.extend(self.tokenizer.encode_text("assistant\n")?);

        Ok(PromptTokens {
            ids,
            audio_pad_start,
            audio_pad_len: audio_len,
        })
    }

    fn parse_alignment(
        &self,
        alignment_text: &str,
        _audio_duration_ms: u32,
    ) -> Result<Vec<(String, u32, u32)>> {
        // Parse alignment output format:
        // Expected: word<|timestamp_0|>word<|timestamp_1|>...
        // or: word[0.00s]word[0.50s]...

        let mut results = Vec::new();
        let mut current_word = String::new();
        let mut last_time_ms: u32 = 0;

        for ch in alignment_text.chars() {
            if ch.is_alphanumeric() || ch == '\'' || ch == '-' {
                current_word.push(ch);
            } else if ch == '<' || ch == '[' {
                // Start of timestamp marker - save current word
                if !current_word.is_empty() {
                    // Parse timestamp
                    let time_ms = last_time_ms; // Simplified - would parse actual timestamp
                    results.push((current_word.clone(), last_time_ms, time_ms + 100));
                    last_time_ms = time_ms + 100;
                    current_word.clear();
                }
            }
        }

        // Handle last word if any
        if !current_word.is_empty() {
            results.push((current_word, last_time_ms, last_time_ms + 100));
        }

        Ok(results)
    }

    fn build_position_ids(
        &self,
        seq_len: usize,
        start_pos: usize,
        audio_span: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        let positions = build_mrope_positions(seq_len, start_pos, audio_span);

        let mut data = Vec::with_capacity(3 * seq_len);
        for _axis in 0..3 {
            data.extend_from_slice(&positions);
        }

        Tensor::from_vec(data, (3, seq_len), &self.device.device).map_err(Error::from)
    }
}

struct PromptTokens {
    ids: Vec<u32>,
    audio_pad_start: usize,
    audio_pad_len: usize,
}

fn normalized_language_name(language: &str) -> String {
    let lang = language.trim();
    if lang.eq_ignore_ascii_case("auto") {
        return "Auto".to_string();
    }

    let mut out = String::with_capacity(lang.len());
    let mut new_word = true;
    for ch in lang.chars() {
        if ch.is_ascii_alphabetic() {
            if new_word {
                out.push(ch.to_ascii_uppercase());
                new_word = false;
            } else {
                out.push(ch.to_ascii_lowercase());
            }
        } else {
            out.push(ch);
            new_word = ch == ' ' || ch == '-' || ch == '_';
        }
    }
    out
}

fn build_mrope_positions(
    seq_len: usize,
    start_pos: usize,
    audio_span: Option<(usize, usize)>,
) -> Vec<i64> {
    if let Some((audio_start, audio_len)) = audio_span {
        let mut pos = Vec::with_capacity(seq_len);
        let mut st = 0usize;
        let mut st_idx = start_pos as i64;

        if audio_start > 0 && audio_start <= seq_len {
            let text_len = audio_start - st;
            for i in 0..text_len {
                pos.push(st_idx + i as i64);
            }
            st = audio_start;
            st_idx += text_len as i64;
        }

        if audio_len > 0 && st < seq_len {
            let audio_take = audio_len.min(seq_len - st);
            for i in 0..audio_take {
                pos.push(st_idx + i as i64);
            }
            st += audio_take;
            st_idx += audio_take as i64;
        }

        if st < seq_len {
            let tail = seq_len - st;
            for i in 0..tail {
                pos.push(st_idx + i as i64);
            }
        }

        pos
    } else {
        (start_pos..start_pos + seq_len).map(|p| p as i64).collect()
    }
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_runaway_repetition(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 16 {
        return text.to_string();
    }

    let mut out: Vec<&str> = Vec::with_capacity(words.len());
    let mut i = 0usize;

    while i < words.len() {
        let mut compressed = false;
        for window in (1..=8).rev() {
            if i + window * 4 > words.len() {
                continue;
            }
            let pattern = &words[i..i + window];
            let mut reps = 1usize;
            while i + (reps + 1) * window <= words.len()
                && &words[i + reps * window..i + (reps + 1) * window] == pattern
            {
                reps += 1;
            }

            if reps >= 4 {
                out.extend_from_slice(pattern);
                out.extend_from_slice(pattern);
                i += reps * window;
                compressed = true;
                break;
            }
        }

        if !compressed {
            out.push(words[i]);
            i += 1;
        }
    }

    out.join(" ")
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
