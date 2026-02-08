//! Native Qwen3-ASR model loader and inference.

mod audio;
mod config;
mod tokenizer;

use std::path::Path;

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use tracing::{debug, info};

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

        // Quantized checkpoints are trained/evaluated in bf16 and can degrade
        // badly when forced through fp32 dequant paths.
        // However, BF16 matmul coverage is incomplete across backends in Candle,
        // so we respect the device's preferred dtype selection.
        let is_quantized = config.quantization.is_some() || config.quantization_config.is_some();
        let dtype = if is_quantized {
            parse_asr_dtype(config.thinker_config.dtype.as_deref())
                .unwrap_or_else(|| device.select_dtype(Some("bfloat16")))
        } else {
            DType::F32
        };

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
        let stop_tokens = collect_stop_token_ids(&self.specials);
        let never_emit_tokens = collect_never_emit_token_ids(&self.specials);

        // Long repetitive generations are almost always degenerate for ASR.
        // Keep this bounded to reduce latency and prevent runaway gibberish.
        let max_tokens = 256usize;
        // Quantized checkpoints may predict stop at step 0. If that happens, force
        // a short non-stop prefix instead of allowing an empty transcript.
        let forced_prefix_len_after_leading_stop = 3usize;
        let mut forced_min_new_tokens = 0usize;
        for step in 0..max_tokens {
            // Get logits for the last position only
            let logits = embeds.i((0, embeds.dim(1)? - 1))?; // [vocab_size]
            let greedy_next = argmax(&logits)?;

            if step == 0 && stop_tokens.contains(&greedy_next) {
                forced_min_new_tokens = forced_prefix_len_after_leading_stop;
            }

            let should_suppress_stop = generated.len() < forced_min_new_tokens;
            let mut excluded_ids = never_emit_tokens.clone();
            if should_suppress_stop {
                excluded_ids.extend_from_slice(&stop_tokens);
            }
            sort_dedup_u32(&mut excluded_ids);

            let greedy_is_excluded = excluded_ids.contains(&greedy_next);

            // Leading-stop masking: if greedy points to a stop token before we've
            // emitted enough tokens, or points at known control-noise tokens,
            // pick the best allowed token instead.
            let next = if greedy_is_excluded {
                if let Some(masked_next) = argmax_excluding(&logits, &excluded_ids)? {
                    debug!(
                        "Suppressing ASR token id={} at step {} -> {} (forced_min_new_tokens={}, suppressed_stop={})",
                        greedy_next,
                        step,
                        masked_next,
                        forced_min_new_tokens,
                        should_suppress_stop
                    );
                    masked_next
                } else {
                    debug!("All ASR logits were filtered by early stop suppression");
                    break;
                }
            } else {
                greedy_next
            };

            if stop_tokens.contains(&next) {
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
        let parsed = self.parse_output(&text, language)?;
        if parsed.is_empty() || looks_like_language_none_loop(&parsed) {
            let preview_ids: Vec<u32> = generated.iter().take(16).copied().collect();
            debug!(
                "ASR decode produced empty/degenerate parsed output after {} tokens (preview ids={:?}, raw='{}', parsed='{}')",
                generated.len(),
                preview_ids,
                text,
                parsed
            );
        }
        Ok(if looks_like_language_none_loop(&parsed) {
            String::new()
        } else {
            parsed
        })
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

    fn parse_output(&self, raw_output: &str, language: Option<&str>) -> Result<String> {
        let mut raw = collapse_whitespace(raw_output.trim());
        if raw.is_empty() {
            return Ok(String::new());
        }

        raw = strip_known_generation_artifacts(&raw);
        raw = collapse_whitespace(raw.trim());
        if raw.is_empty() || raw == "<non_speech>" || raw.contains("No speech detected") {
            return Ok(String::new());
        }

        // Upstream contract: when a language is forced, generation should be
        // text-only because we append "language X<asr_text>" to the prompt.
        if forced_language_name(language).is_some() {
            let mut text = strip_runaway_repetition(raw.trim());
            text = collapse_whitespace(text.trim());
            if looks_like_language_none_loop(&text) {
                return Ok(String::new());
            }
            return Ok(text);
        }

        // Default contract: "language X<asr_text>transcript".
        if let Some((meta, tail)) = raw.split_once("<asr_text>") {
            let mut text = strip_known_generation_artifacts(tail.trim());
            text = strip_runaway_repetition(text.trim());
            text = collapse_whitespace(text.trim());
            if meta.to_ascii_lowercase().contains("language none") && text.is_empty() {
                return Ok(String::new());
            }
            if text == "<non_speech>" || looks_like_language_none_loop(&text) {
                return Ok(String::new());
            }
            return Ok(text);
        }

        // Fallback for malformed outputs without <asr_text>.
        let mut text = strip_runaway_repetition(raw.trim());
        text = trim_leading_language_header(&text);
        text = strip_language_none_prefixes(text.trim());
        text = collapse_whitespace(text.trim());
        if text == "<non_speech>" || looks_like_language_none_loop(&text) {
            return Ok(String::new());
        }
        Ok(text)
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
        // Match upstream Qwen3-ASR prompt contract:
        // <|im_start|>system\n<|im_end|>\n
        // <|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
        // <|im_start|>assistant\n
        // If language is explicitly forced, append: "language {Lang}<asr_text>".
        let forced_language = forced_language_name(language);
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
        ids.push(self.specials.im_end);
        ids.extend(self.tokenizer.encode_text("\n")?);
        ids.push(self.specials.im_start);
        ids.extend(self.tokenizer.encode_text("assistant\n")?);
        if let Some(lang) = forced_language {
            ids.extend(self.tokenizer.encode_text("language ")?);
            ids.extend(self.tokenizer.encode_text(&lang)?);
            if let Some(asr_text) = self.specials.asr_text {
                ids.push(asr_text);
            } else {
                ids.extend(self.tokenizer.encode_text("<asr_text>")?);
            }
        }

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

fn parse_asr_dtype(dtype: Option<&str>) -> Option<DType> {
    match dtype.map(|d| d.trim().to_ascii_lowercase()) {
        Some(d) if d == "bfloat16" || d == "bf16" => Some(DType::BF16),
        Some(d) if d == "float16" || d == "f16" || d == "fp16" => Some(DType::F16),
        Some(d) if d == "float32" || d == "f32" || d == "fp32" => Some(DType::F32),
        _ => None,
    }
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

fn forced_language_name(language: Option<&str>) -> Option<String> {
    let lang = language?.trim();
    if lang.is_empty() || lang.eq_ignore_ascii_case("auto") {
        return None;
    }
    Some(normalized_language_name(lang))
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

fn strip_known_generation_artifacts(text: &str) -> String {
    text.replace("<|fim_prefix|>", " ")
        .replace("<|fim_middle|>", " ")
        .replace("<|fim_suffix|>", " ")
        .replace("<|fim_pad|>", " ")
        .replace("<|im_end|>", " ")
}

fn strip_language_none_prefixes(text: &str) -> String {
    let mut out = text.trim().to_string();
    // Handle concatenated artifacts like "Nonelanguage".
    out = out.replace("Nonelanguage", "None language");
    for _ in 0..8 {
        let lower = out.to_lowercase();
        if lower.starts_with("language none") {
            let mut cut = "language none".len();
            while cut < out.len() {
                let ch = out.as_bytes()[cut] as char;
                if ch.is_whitespace() || ch == ':' || ch == ',' || ch == '.' || ch == ';' {
                    cut += 1;
                } else {
                    break;
                }
            }
            out = out[cut..].trim().to_string();
            continue;
        }
        break;
    }
    out
}

fn trim_leading_language_header(text: &str) -> String {
    let trimmed = text.trim();
    let lower = trimmed.to_ascii_lowercase();
    if !lower.starts_with("language ") {
        return trimmed.to_string();
    }
    if let Some(newline_idx) = trimmed.find('\n') {
        return trimmed[newline_idx + 1..].trim().to_string();
    }

    let mut parts = trimmed.splitn(3, ' ');
    let _ = parts.next(); // "language"
    let _ = parts.next(); // language value
    parts.next().unwrap_or_default().trim().to_string()
}

fn looks_like_language_none_loop(text: &str) -> bool {
    let cleaned = strip_known_generation_artifacts(text);
    let cleaned = cleaned.replace("Nonelanguage", "None language");
    let mut normalized = String::with_capacity(cleaned.len());
    for ch in cleaned.chars() {
        if ch.is_ascii_alphabetic() || ch.is_ascii_whitespace() {
            normalized.push(ch.to_ascii_lowercase());
        } else {
            normalized.push(' ');
        }
    }
    let words: Vec<&str> = normalized.split_whitespace().collect();
    !words.is_empty() && words.iter().all(|w| *w == "language" || *w == "none")
}

fn collect_never_emit_token_ids(specials: &SpecialTokenIds) -> Vec<u32> {
    let mut ids = Vec::new();
    if let Some(id) = specials.fim_prefix {
        ids.push(id);
    }
    if let Some(id) = specials.fim_middle {
        ids.push(id);
    }
    if let Some(id) = specials.fim_suffix {
        ids.push(id);
    }
    if let Some(id) = specials.fim_pad {
        ids.push(id);
    }
    ids
}

fn sort_dedup_u32(ids: &mut Vec<u32>) {
    ids.sort_unstable();
    ids.dedup();
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

fn argmax_excluding(logits: &Tensor, excluded_ids: &[u32]) -> Result<Option<u32>> {
    let logits = logits.to_dtype(DType::F32)?;
    let values = logits.to_vec1::<f32>()?;
    let mut max_idx = None;
    let mut max_val = f32::NEG_INFINITY;
    for (idx, &val) in values.iter().enumerate() {
        if excluded_ids.contains(&(idx as u32)) {
            continue;
        }
        if val > max_val {
            max_val = val;
            max_idx = Some(idx as u32);
        }
    }
    Ok(max_idx)
}

fn collect_stop_token_ids(specials: &SpecialTokenIds) -> Vec<u32> {
    let mut stop_ids = vec![specials.im_end, specials.eos];
    if let Some(alt) = specials.eos_alt {
        if alt != specials.im_end && alt != specials.eos {
            stop_ids.push(alt);
        }
    }
    stop_ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn argmax_excluding_selects_best_non_stop() {
        let logits = Tensor::from_vec(vec![0.1f32, 5.0, 3.0], 3, &Device::Cpu).unwrap();
        let token = argmax_excluding(&logits, &[1]).unwrap();
        assert_eq!(token, Some(2));
    }

    #[test]
    fn argmax_excluding_returns_none_if_all_ids_blocked() {
        let logits = Tensor::from_vec(vec![0.1f32, 5.0, 3.0], 3, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let token = argmax_excluding(&logits, &[0, 1, 2]).unwrap();
        assert_eq!(token, None);
    }

    #[test]
    fn collect_stop_token_ids_deduplicates_alt_eos() {
        let specials = SpecialTokenIds {
            im_start: 1,
            im_end: 2,
            audio_start: 3,
            audio_end: 4,
            audio_token: 5,
            asr_text: Some(7),
            fim_prefix: Some(8),
            fim_middle: Some(9),
            fim_suffix: Some(10),
            fim_pad: Some(11),
            eos: 6,
            eos_alt: Some(6),
            pad: 0,
        };
        let stop_ids = collect_stop_token_ids(&specials);
        assert_eq!(stop_ids, vec![2, 6]);
    }

    #[test]
    fn language_none_loop_detection() {
        assert!(looks_like_language_none_loop(
            "<|fim_middle|>language Nonelanguage Nonelanguage None"
        ));
        assert!(!looks_like_language_none_loop("hello world"));
    }

    #[test]
    fn sort_dedup_u32_works() {
        let mut ids = vec![3, 1, 3, 2, 2];
        sort_dedup_u32(&mut ids);
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn forced_language_name_ignores_auto_and_empty() {
        assert_eq!(forced_language_name(None), None);
        assert_eq!(forced_language_name(Some("")), None);
        assert_eq!(forced_language_name(Some("Auto")), None);
        assert_eq!(
            forced_language_name(Some("english")),
            Some("English".to_string())
        );
    }

    #[test]
    fn trim_leading_language_header_works() {
        assert_eq!(
            trim_leading_language_header("language English hello world"),
            "hello world"
        );
        assert_eq!(
            trim_leading_language_header("language Chinese\n你好世界"),
            "你好世界"
        );
        assert_eq!(trim_leading_language_header("plain text"), "plain text");
    }

    #[test]
    fn parse_asr_dtype_handles_common_aliases() {
        assert_eq!(parse_asr_dtype(Some("bf16")), Some(DType::BF16));
        assert_eq!(parse_asr_dtype(Some("bfloat16")), Some(DType::BF16));
        assert_eq!(parse_asr_dtype(Some("fp16")), Some(DType::F16));
        assert_eq!(parse_asr_dtype(Some("float32")), Some(DType::F32));
        assert_eq!(parse_asr_dtype(Some("unknown")), None);
    }
}
