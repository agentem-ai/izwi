//! Native Candle implementation for LiquidAI LFM2-Audio.

mod config;
mod conformer;
mod depthformer;
mod lfm_backbone;
mod mimi_decoder;
mod preprocessor;
mod tokenizer;

use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};
use config::Lfm2AudioConfig;
use conformer::ConformerEncoder;
use depthformer::Depthformer;
use lfm_backbone::{LfmBackbone, LfmCache};
use mimi_decoder::MimiDecoder;
use preprocessor::{resample_linear, Lfm2AudioPreprocessor};
use tokenizer::{mel_to_embed_len, ChatState, Lfm2Tokenizer, LfmModality};
use tracing::info;

use crate::error::{Error, Result};
use crate::models::device::DeviceProfile;

pub const LFM2_DEFAULT_S2S_PROMPT: &str = "Respond with interleaved text and audio.";

const TTS_US_MALE_PROMPT: &str = "Perform TTS. Use the US male voice.";
const TTS_US_FEMALE_PROMPT: &str = "Perform TTS. Use the US female voice.";
const TTS_UK_MALE_PROMPT: &str = "Perform TTS. Use the UK male voice.";
const TTS_UK_FEMALE_PROMPT: &str = "Perform TTS. Use the UK female voice.";

const END_OF_AUDIO_TOKEN: u32 = 2048;

pub struct Lfm2AudioModel {
    model_dir: PathBuf,
    device: DeviceProfile,
    cfg: Lfm2AudioConfig,
    tokenizer: Lfm2Tokenizer,
    preprocessor: Lfm2AudioPreprocessor,
    conformer: ConformerEncoder,
    audio_adapter: AudioAdapter,
    lfm: LfmBackbone,
    depthformer: Depthformer,
    mimi: MimiDecoder,
}

struct AudioAdapter {
    norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl AudioAdapter {
    fn load(vb: VarBuilder) -> Result<Self> {
        let ln_dim = vb
            .pp("model.0")
            .get(1, "weight")
            .map(|t| t.dim(0).unwrap_or(0))
            .unwrap_or(512);

        let norm = candle_nn::layer_norm(ln_dim, 1e-5, vb.pp("model.0"))?;
        let linear1 = crate::models::mlx_compat::load_linear(ln_dim, 2048, vb.pp("model.1"))?;
        let linear2 = crate::models::mlx_compat::load_linear(2048, 2048, vb.pp("model.3"))?;

        Ok(Self {
            norm,
            linear1,
            linear2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let x = self.linear1.forward(&x)?;
        let x = x.gelu_erf()?;
        self.linear2.forward(&x).map_err(Error::from)
    }
}

impl Lfm2AudioModel {
    pub fn load(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        validate_model_dir(model_dir)?;

        let cfg: Lfm2AudioConfig = serde_json::from_str(
            std::fs::read_to_string(model_dir.join("config.json"))?.as_str(),
        )
        .map_err(|e| Error::ModelLoadError(format!("Invalid LFM2 config.json: {e}")))?;

        let dtype = match device.kind {
            crate::models::device::DeviceKind::Cuda if device.capabilities.supports_bf16 => {
                DType::BF16
            }
            _ => DType::F32,
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_dir.join("model.safetensors")], dtype, &device.device)?
        };

        let tokenizer = Lfm2Tokenizer::load(model_dir)?;
        let preprocessor = Lfm2AudioPreprocessor::new(cfg.preprocessor.clone())?;
        let conformer = ConformerEncoder::load(cfg.encoder.clone(), vb.pp("conformer"))?;
        let audio_adapter = AudioAdapter::load(vb.pp("audio_adapter"))?;
        let lfm = LfmBackbone::load(cfg.lfm.clone(), vb.pp("lfm"))?;
        let depthformer = Depthformer::load(&cfg, vb.clone())?;
        let mimi = MimiDecoder::load(model_dir, &device.device)?;

        info!("Loaded native LFM2 model from {:?}", model_dir);

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            device,
            cfg,
            tokenizer,
            preprocessor,
            conformer,
            audio_adapter,
            lfm,
            depthformer,
            mimi,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn available_voices(&self) -> Vec<String> {
        vec![
            "US Male".to_string(),
            "US Female".to_string(),
            "UK Male".to_string(),
            "UK Female".to_string(),
        ]
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        _language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let mel = self.prepare_audio_mel(audio, sample_rate)?;

        let mut state = ChatState::new(&self.tokenizer, self.cfg.codebooks, self.preprocessor.features());
        state.new_turn(&self.tokenizer, "system")?;
        state.add_text(
            &self.tokenizer,
            "Transcribe the user speech into text.",
        )?;
        state.end_turn(&self.tokenizer)?;

        state.new_turn(&self.tokenizer, "user")?;
        state.add_audio_mel(&mel.0, mel.1);
        state.end_turn(&self.tokenizer)?;

        state.new_turn(&self.tokenizer, "assistant")?;

        let mut tokens = Vec::new();
        let mut assembled = String::new();
        let mut rng = SimpleRng::new();

        self.generate_sequential(
            &state,
            512,
            None,
            Some(1),
            &mut rng,
            &mut |token| {
                tokens.push(token);
                if let Ok(decoded) = self.tokenizer.decode_text(&tokens) {
                    let delta = text_delta(&assembled, &decoded);
                    if !delta.is_empty() {
                        on_delta(delta.as_str());
                    }
                    assembled = decoded;
                }
            },
            &mut |_frame| {},
        )?;

        Ok(assembled.trim().to_string())
    }

    pub fn synthesize_with_callback(
        &self,
        text: &str,
        speaker_prompt: &str,
        temperature: Option<f32>,
        top_k: Option<usize>,
        max_new_tokens: usize,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<Vec<f32>> {
        let mut state = ChatState::new(&self.tokenizer, self.cfg.codebooks, self.preprocessor.features());
        state.new_turn(&self.tokenizer, "system")?;
        state.add_text(&self.tokenizer, speaker_prompt)?;
        state.end_turn(&self.tokenizer)?;

        state.new_turn(&self.tokenizer, "user")?;
        state.add_text(&self.tokenizer, text)?;
        state.end_turn(&self.tokenizer)?;

        state.new_turn(&self.tokenizer, "assistant")?;

        let mut text_tokens = Vec::new();
        let mut assembled = String::new();
        let mut audio_frames: Vec<Vec<u32>> = vec![Vec::new(); self.cfg.codebooks];
        let mut rng = SimpleRng::new();

        self.generate_interleaved(
            &state,
            max_new_tokens.max(256),
            temperature,
            top_k,
            &mut rng,
            &mut |token| {
                text_tokens.push(token);
                if let Ok(decoded) = self.tokenizer.decode_text(&text_tokens) {
                    let delta = text_delta(&assembled, &decoded);
                    if !delta.is_empty() {
                        on_delta(delta.as_str());
                    }
                    assembled = decoded;
                }
            },
            &mut |frame| {
                if frame.first().copied() == Some(END_OF_AUDIO_TOKEN) {
                    return;
                }
                for (i, &tok) in frame.iter().enumerate() {
                    if i < audio_frames.len() {
                        audio_frames[i].push(tok);
                    }
                }
            },
        )?;

        self.mimi.decode_tokens(&audio_frames)
    }

    pub fn speech_to_speech_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        system_prompt: Option<&str>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        max_new_tokens: usize,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<(String, Vec<f32>)> {
        let mel = self.prepare_audio_mel(audio, sample_rate)?;

        let mut state = ChatState::new(&self.tokenizer, self.cfg.codebooks, self.preprocessor.features());
        state.new_turn(&self.tokenizer, "system")?;
        state.add_text(
            &self.tokenizer,
            system_prompt.unwrap_or(LFM2_DEFAULT_S2S_PROMPT),
        )?;
        state.end_turn(&self.tokenizer)?;

        state.new_turn(&self.tokenizer, "user")?;
        state.add_audio_mel(&mel.0, mel.1);
        state.end_turn(&self.tokenizer)?;

        state.new_turn(&self.tokenizer, "assistant")?;

        let mut text_tokens = Vec::new();
        let mut assembled = String::new();
        let mut audio_frames: Vec<Vec<u32>> = vec![Vec::new(); self.cfg.codebooks];
        let mut rng = SimpleRng::new();

        self.generate_interleaved(
            &state,
            max_new_tokens.max(768),
            temperature,
            top_k,
            &mut rng,
            &mut |token| {
                text_tokens.push(token);
                if let Ok(decoded) = self.tokenizer.decode_text(&text_tokens) {
                    let delta = text_delta(&assembled, &decoded);
                    if !delta.is_empty() {
                        on_delta(delta.as_str());
                    }
                    assembled = decoded;
                }
            },
            &mut |frame| {
                if frame.first().copied() == Some(END_OF_AUDIO_TOKEN) {
                    return;
                }
                for (i, &tok) in frame.iter().enumerate() {
                    if i < audio_frames.len() {
                        audio_frames[i].push(tok);
                    }
                }
            },
        )?;

        let wav = self.mimi.decode_tokens(&audio_frames)?;
        Ok((assembled.trim().to_string(), wav))
    }

    fn prepare_audio_mel(&self, audio: &[f32], sample_rate: u32) -> Result<(Vec<f32>, usize)> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let target_sr = self.preprocessor.sample_rate();
        let mono = if sample_rate == target_sr {
            audio.to_vec()
        } else {
            resample_linear(audio, sample_rate, target_sr)
        };

        let (mel, frames) = self.preprocessor.compute_features(&mono)?;
        let mel = mel.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
        Ok((mel, frames))
    }

    fn build_prefill_embeddings(&self, state: &ChatState) -> Result<Tensor> {
        let text_ids = Tensor::from_vec(
            state.text.clone(),
            state.text.len(),
            self.lfm.embed_tokens_weight().device(),
        )?
        .to_dtype(DType::U32)?;
        let text_emb = self.lfm.embed_sequence(&text_ids)?; // [T_text, D]

        let mut audio_in_rows: Vec<Tensor> = Vec::new();
        if !state.audio_in_lens.is_empty() {
            let total_frames: usize = state.audio_in_lens.iter().sum();
            let audio_in = Tensor::from_vec(
                state.audio_in.clone(),
                (state.features, total_frames),
                self.lfm.embed_tokens_weight().device(),
            )?;

            let mut start = 0usize;
            for &len in &state.audio_in_lens {
                let seg = audio_in.narrow(1, start, len)?;
                let seg = seg.unsqueeze(0)?;
                let (enc, enc_len) = self.conformer.encode(&seg, len)?;
                let enc = enc.i((0, ..enc_len, ..))?;
                let enc = self.audio_adapter.forward(&enc)?;
                for i in 0..enc_len {
                    audio_in_rows.push(enc.i((i, ..))?);
                }
                start += len;
            }
        }

        let mut audio_out_rows: Vec<Tensor> = Vec::new();
        if !state.audio_out.is_empty() {
            let frames = state.audio_out.len() / self.cfg.codebooks;
            let audio_out = Tensor::from_vec(
                state.audio_out.clone(),
                (self.cfg.codebooks, frames),
                self.lfm.embed_tokens_weight().device(),
            )?
            .to_dtype(DType::U32)?;
            for t in 0..frames {
                let frame = audio_out.i((.., t))?;
                let emb = self.depthformer.audio_embedding_sum(&frame)?;
                audio_out_rows.push(emb.squeeze(0)?.squeeze(0)?);
            }
        }

        let mut text_i = 0usize;
        let mut audio_in_i = 0usize;
        let mut audio_out_i = 0usize;
        let mut rows = Vec::with_capacity(state.modality_flag.len());

        for &m in &state.modality_flag {
            match m {
                x if x == LfmModality::Text as u32 => {
                    rows.push(text_emb.i((text_i, ..))?);
                    text_i += 1;
                }
                x if x == LfmModality::AudioIn as u32 => {
                    let row = audio_in_rows
                        .get(audio_in_i)
                        .ok_or_else(|| Error::InferenceError("audio_in/modality mismatch".to_string()))?
                        .clone();
                    rows.push(row);
                    audio_in_i += 1;
                }
                x if x == LfmModality::AudioOut as u32 => {
                    let row = audio_out_rows
                        .get(audio_out_i)
                        .ok_or_else(|| Error::InferenceError("audio_out/modality mismatch".to_string()))?
                        .clone();
                    rows.push(row);
                    audio_out_i += 1;
                }
                _ => {
                    return Err(Error::InferenceError(
                        "Unsupported LFM2 modality flag".to_string(),
                    ));
                }
            }
        }

        if rows.is_empty() {
            return Err(Error::InferenceError(
                "Empty LFM2 prefill embedding sequence".to_string(),
            ));
        }

        Ok(Tensor::stack(&rows, 0)?.unsqueeze(0)?)
    }

    fn generate_sequential(
        &self,
        state: &ChatState,
        max_new_tokens: usize,
        temperature: Option<f32>,
        top_k: Option<usize>,
        rng: &mut SimpleRng,
        on_text: &mut dyn FnMut(u32),
        on_audio: &mut dyn FnMut(&[u32]),
    ) -> Result<()> {
        let mut in_emb = self.build_prefill_embeddings(state)?;
        let mut current_modality = LfmModality::Text;
        let mut cache = LfmCache::new(self.lfm.config());

        for _ in 0..max_new_tokens {
            let out = self.lfm.forward_embeds_cached(&in_emb, &mut cache)?;
            let last = out.i((0, out.dim(1)? - 1, ..))?;

            match current_modality {
                LfmModality::Text => {
                    let logits = last
                        .reshape((1, last.dim(0)?))?
                        .matmul(&self.lfm.embed_tokens_weight().t()?)?
                        .squeeze(0)?;
                    let token = sample_token(
                        &logits,
                        temperature.unwrap_or(0.0) <= 0.0 || top_k == Some(1),
                        temperature,
                        top_k,
                        rng,
                    )?;

                    if token == self.tokenizer.specials().im_end {
                        break;
                    }

                    on_text(token);

                    if token == self.tokenizer.specials().audio_start {
                        current_modality = LfmModality::AudioOut;
                    }

                    in_emb = self.lfm.embed_tokens(token)?;
                }
                LfmModality::AudioOut => {
                    let frame = self.depthformer.sample_audio_frame(
                        &last,
                        temperature,
                        top_k,
                        rng,
                    )?;
                    let mut frame = frame;
                    if frame.first().copied() == Some(END_OF_AUDIO_TOKEN) {
                        for t in &mut frame {
                            *t = END_OF_AUDIO_TOKEN;
                        }
                        current_modality = LfmModality::Text;
                    }

                    on_audio(&frame);
                    let frame_t = Tensor::from_vec(
                        frame,
                        self.cfg.codebooks,
                        self.lfm.embed_tokens_weight().device(),
                    )?
                    .to_dtype(DType::U32)?;
                    in_emb = self.depthformer.audio_embedding_sum(&frame_t)?;
                }
                LfmModality::AudioIn => {}
            }
        }

        Ok(())
    }

    fn generate_interleaved(
        &self,
        state: &ChatState,
        max_new_tokens: usize,
        temperature: Option<f32>,
        top_k: Option<usize>,
        rng: &mut SimpleRng,
        on_text: &mut dyn FnMut(u32),
        on_audio: &mut dyn FnMut(&[u32]),
    ) -> Result<()> {
        let mut in_emb = self.build_prefill_embeddings(state)?;
        let mut current_modality = LfmModality::Text;
        let mut modality_left = self.cfg.interleaved_n_text;
        let mut text_done = false;
        let mut cache = LfmCache::new(self.lfm.config());

        for _ in 0..max_new_tokens {
            modality_left = modality_left.saturating_sub(1);

            let out = self.lfm.forward_embeds_cached(&in_emb, &mut cache)?;
            let last = out.i((0, out.dim(1)? - 1, ..))?;

            match current_modality {
                LfmModality::Text => {
                    let logits = last
                        .reshape((1, last.dim(0)?))?
                        .matmul(&self.lfm.embed_tokens_weight().t()?)?
                        .squeeze(0)?;
                    let token = sample_token(
                        &logits,
                        temperature.unwrap_or(0.0) <= 0.0 || top_k == Some(1),
                        temperature,
                        top_k,
                        rng,
                    )?;

                    if token == self.tokenizer.specials().im_end {
                        break;
                    }

                    on_text(token);

                    if token == self.tokenizer.specials().text_end {
                        text_done = true;
                    }

                    if modality_left == 0 || text_done {
                        current_modality = LfmModality::AudioOut;
                        modality_left = self.cfg.interleaved_n_audio;
                    }

                    in_emb = self.lfm.embed_tokens(token)?;
                }
                LfmModality::AudioOut => {
                    let mut frame = self.depthformer.sample_audio_frame(
                        &last,
                        temperature,
                        top_k,
                        rng,
                    )?;

                    if modality_left == 0 && !text_done {
                        current_modality = LfmModality::Text;
                        modality_left = self.cfg.interleaved_n_text;
                    }

                    if frame.first().copied() == Some(END_OF_AUDIO_TOKEN) {
                        for t in &mut frame {
                            *t = END_OF_AUDIO_TOKEN;
                        }
                        current_modality = LfmModality::Text;
                        modality_left = self.cfg.interleaved_n_text;
                    }

                    on_audio(&frame);

                    let frame_t = Tensor::from_vec(
                        frame,
                        self.cfg.codebooks,
                        self.lfm.embed_tokens_weight().device(),
                    )?
                    .to_dtype(DType::U32)?;
                    in_emb = self.depthformer.audio_embedding_sum(&frame_t)?;
                }
                LfmModality::AudioIn => {}
            }
        }

        Ok(())
    }
}

pub fn lfm2_tts_voice_prompt(speaker: Option<&str>) -> &'static str {
    let normalized = speaker
        .unwrap_or("")
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect::<String>();

    if normalized.contains("ukmale")
        || normalized == "dylan"
        || normalized == "unclefu"
        || normalized == "ukm"
    {
        return TTS_UK_MALE_PROMPT;
    }

    if normalized.contains("ukfemale") || normalized == "vivian" {
        return TTS_UK_FEMALE_PROMPT;
    }

    if normalized.contains("usmale")
        || normalized == "ryan"
        || normalized == "aiden"
        || normalized == "eric"
        || normalized.contains("male")
    {
        return TTS_US_MALE_PROMPT;
    }

    if normalized.contains("usfemale")
        || normalized == "serena"
        || normalized == "sohee"
        || normalized == "onoanna"
        || normalized == "anna"
    {
        return TTS_US_FEMALE_PROMPT;
    }

    TTS_US_FEMALE_PROMPT
}

fn validate_model_dir(model_dir: &Path) -> Result<()> {
    let required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer-e351c8d8-checkpoint125.safetensors",
    ];
    for file in &required_files {
        let path = model_dir.join(file);
        if !path.exists() {
            return Err(Error::ModelLoadError(format!(
                "LFM2 model is missing required file: {}",
                path.display()
            )));
        }
    }

    Ok(())
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

pub(crate) fn sample_token(
    logits: &Tensor,
    greedy: bool,
    temperature: Option<f32>,
    top_k: Option<usize>,
    rng: &mut SimpleRng,
) -> Result<u32> {
    let values = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    if greedy {
        return argmax_from_slice(&values)
            .map(|idx| idx as u32)
            .ok_or_else(|| Error::InferenceError("Empty logits".to_string()));
    }

    let temp = temperature.unwrap_or(1.0).max(1e-5);
    let mut candidates: Vec<(usize, f32)> = values
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| {
            if v.is_finite() {
                Some((i, v / temp))
            } else {
                None
            }
        })
        .collect();

    if candidates.is_empty() {
        return Err(Error::InferenceError(
            "No valid sampling candidates".to_string(),
        ));
    }

    if let Some(k) = top_k {
        if k > 0 && k < candidates.len() {
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            candidates.truncate(k);
        }
    }

    let max_logit = candidates
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut probs: Vec<(usize, f32)> = candidates
        .iter()
        .map(|(idx, v)| (*idx, (*v - max_logit).exp()))
        .collect();

    let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return argmax_from_slice(&values)
            .map(|idx| idx as u32)
            .ok_or_else(|| Error::InferenceError("Empty logits".to_string()));
    }
    for (_, p) in &mut probs {
        *p /= sum;
    }

    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (idx, p) in probs {
        acc += p;
        if r <= acc {
            return Ok(idx as u32);
        }
    }

    argmax_from_slice(&values)
        .map(|idx| idx as u32)
        .ok_or_else(|| Error::InferenceError("Sampling fallback failed".to_string()))
}

fn argmax_from_slice(values: &[f32]) -> Option<usize> {
    let mut best = None;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best = Some(i);
            best_val = v;
        }
    }
    best
}

pub(crate) struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x1234_5678_9abc_def0);
        Self { state: nanos }
    }

    fn next_u32(&mut self) -> u32 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) & 0xffff_ffff) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_known_speakers_to_expected_prompts() {
        assert_eq!(lfm2_tts_voice_prompt(Some("Ryan")), TTS_US_MALE_PROMPT);
        assert_eq!(lfm2_tts_voice_prompt(Some("Serena")), TTS_US_FEMALE_PROMPT);
        assert_eq!(lfm2_tts_voice_prompt(Some("Dylan")), TTS_UK_MALE_PROMPT);
        assert_eq!(lfm2_tts_voice_prompt(Some("Vivian")), TTS_UK_FEMALE_PROMPT);
    }
}
