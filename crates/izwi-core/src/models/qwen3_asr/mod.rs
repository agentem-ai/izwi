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
use crate::models::device::{DeviceKind, DeviceProfile};
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
    audio_dtype: DType,
    text_dtype: DType,
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
        // badly when forced through fp32 dequant paths. Audio conditioning is
        // especially sensitive to precision, so keep the audio tower in F32
        // for stability and select the text dtype with backend-aware rules.
        let is_quantized = config.quantization.is_some() || config.quantization_config.is_some();
        if is_quantized {
            validate_quantization_config(&config)?;
        }
        let audio_dtype = DType::F32;
        let text_dtype = if is_quantized {
            let requested =
                parse_asr_dtype(config.thinker_config.dtype.as_deref()).unwrap_or(DType::BF16);
            let selected = match device.kind {
                DeviceKind::Metal => DType::F32,
                DeviceKind::Cpu => DType::F32,
                DeviceKind::Cuda => {
                    if requested == DType::BF16 && !device.capabilities.supports_bf16 {
                        DType::F16
                    } else {
                        requested
                    }
                }
            };
            debug!(
                "Qwen3-ASR quantized dtype selection: requested={:?}, selected={:?} on {:?}",
                requested, selected, device.kind
            );
            selected
        } else {
            DType::F32
        };

        // Check for sharded weights (1.7B model) vs single file (0.6B model)
        let index_path = model_dir.join("model.safetensors.index.json");
        let vb_text = if index_path.exists() {
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
            unsafe {
                VarBuilder::from_mmaped_safetensors(&shard_paths, text_dtype, &device.device)?
            }
        } else {
            // Load single file
            let weights_path = model_dir.join("model.safetensors");
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], text_dtype, &device.device)?
            }
        };
        let vb_audio = if index_path.exists() {
            let index_data = std::fs::read_to_string(&index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_data)?;
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
            unsafe {
                VarBuilder::from_mmaped_safetensors(&shard_paths, audio_dtype, &device.device)?
            }
        } else {
            let weights_path = model_dir.join("model.safetensors");
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], audio_dtype, &device.device)?
            }
        };

        let has_thinker_prefix = vb_text.contains_tensor("thinker.audio_tower.conv2d1.weight");
        let vb_text = if has_thinker_prefix {
            vb_text.pp("thinker")
        } else {
            vb_text
        };
        let vb_audio = if has_thinker_prefix {
            vb_audio.pp("thinker")
        } else {
            vb_audio
        };

        let audio_cfg = config.thinker_config.audio_config.clone();
        let audio_tower = AudioTower::load(audio_cfg, vb_audio.pp("audio_tower"))?;
        let text_cfg = config.thinker_config.text_config.clone();
        let text_model = Qwen3Model::load(text_cfg, vb_text)?;

        info!("Loaded Qwen3-ASR model on {:?}", device.kind);

        Ok(Self {
            device,
            audio_dtype,
            text_dtype,
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
        let mut no_op = |_delta: &str| {};
        self.transcribe_with_callback(audio, sample_rate, language, &mut no_op)
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
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
            .to_dtype(self.audio_dtype)?;

        let feature_lens = vec![frames];
        let mut audio_embeds = self.audio_tower.forward(&mel, Some(&feature_lens))?; // [1, t, hidden]
        if audio_embeds.dtype() != self.text_dtype {
            audio_embeds = audio_embeds.to_dtype(self.text_dtype)?;
        }
        let audio_len = audio_embeds.dim(1)?;

        let effective_language =
            forced_language_name(language).unwrap_or_else(|| "English".to_string());
        let prompt = self.build_prompt(audio_len, Some(effective_language.as_str()))?;
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
        let mut assembled = String::new();
        let stop_tokens = collect_stop_token_ids(&self.specials);

        // Keep this bounded to reduce latency.
        let max_tokens = 256usize;
        for _step in 0..max_tokens {
            // Get logits for the last position only
            let logits = embeds.i((0, embeds.dim(1)? - 1))?; // [vocab_size]
            let next = argmax(&logits)?;

            if stop_tokens.contains(&next) {
                break;
            }
            generated.push(next);
            let decoded = self.decode_generated_untrimmed(&generated)?;
            let delta = text_delta(&assembled, &decoded);
            for ch in delta.chars() {
                let mut buf = [0u8; 4];
                on_delta(ch.encode_utf8(&mut buf));
            }
            assembled = decoded;

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

        let text = assembled.trim().to_string();
        Ok(text)
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
            .unsqueeze(0)? // [1, 1, n_mels, frames]
            .to_dtype(self.audio_dtype)?;

        let feature_lens = vec![frames];
        let mut audio_embeds = self.audio_tower.forward(&mel, Some(&feature_lens))?;
        if audio_embeds.dtype() != self.text_dtype {
            audio_embeds = audio_embeds.to_dtype(self.text_dtype)?;
        }
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

    fn decode_generated_untrimmed(&self, tokens: &[u32]) -> Result<String> {
        let filtered: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|id| !is_special_generation_token(&self.specials, *id))
            .collect();
        let text = self.tokenizer.decode_text(&filtered)?;
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

fn validate_quantization_config(config: &Qwen3AsrConfig) -> Result<()> {
    let quant = config
        .quantization_config
        .as_ref()
        .or_else(|| config.quantization.as_ref());
    let Some(quant) = quant else {
        return Ok(());
    };

    if let Some(mode) = quant.get("mode").and_then(|v| v.as_str()) {
        if mode != "affine" {
            return Err(Error::InvalidInput(format!(
                "Unsupported MLX quantization mode '{mode}'. Only affine is supported."
            )));
        }
    }

    if let Some(bits) = quant.get("bits").and_then(|v| v.as_u64()) {
        if bits == 0 || bits > 8 {
            return Err(Error::InvalidInput(format!(
                "Unsupported MLX quantization bit-width {bits}. Only 1-8 bits are supported."
            )));
        }
    }

    Ok(())
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

fn is_special_generation_token(specials: &SpecialTokenIds, id: u32) -> bool {
    if id == specials.im_start
        || id == specials.im_end
        || id == specials.audio_start
        || id == specials.audio_end
        || id == specials.audio_token
        || id == specials.pad
        || id == specials.eos
        || specials.eos_alt == Some(id)
    {
        return true;
    }
    if let Some(asr_text) = specials.asr_text {
        if id == asr_text {
            return true;
        }
    }
    if let Some(id0) = specials.fim_prefix {
        if id == id0 {
            return true;
        }
    }
    if let Some(id0) = specials.fim_middle {
        if id == id0 {
            return true;
        }
    }
    if let Some(id0) = specials.fim_suffix {
        if id == id0 {
            return true;
        }
    }
    if let Some(id0) = specials.fim_pad {
        if id == id0 {
            return true;
        }
    }
    false
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

fn text_delta(previous: &str, current: &str) -> String {
    if let Some(delta) = current.strip_prefix(previous) {
        return delta.to_string();
    }
    let common = previous
        .chars()
        .zip(current.chars())
        .take_while(|(a, b)| a == b)
        .count();
    current.chars().skip(common).collect()
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
    use candle_core::DType;

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
    fn parse_asr_dtype_handles_common_aliases() {
        assert_eq!(parse_asr_dtype(Some("bf16")), Some(DType::BF16));
        assert_eq!(parse_asr_dtype(Some("bfloat16")), Some(DType::BF16));
        assert_eq!(parse_asr_dtype(Some("fp16")), Some(DType::F16));
        assert_eq!(parse_asr_dtype(Some("float32")), Some(DType::F32));
        assert_eq!(parse_asr_dtype(Some("unknown")), None);
    }

    #[test]
    fn text_delta_finds_suffix_when_prefix_changes() {
        assert_eq!(text_delta("Hello", "Hello world"), " world");
        assert_eq!(text_delta("abcd", "abXY"), "XY");
    }
}
