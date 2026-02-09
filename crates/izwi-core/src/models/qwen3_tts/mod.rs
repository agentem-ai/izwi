//! Native Qwen3-TTS model loader and inference.
//!
//! This module provides native Rust implementation for Qwen3-TTS models,
//! supporting both CustomVoice (preset speakers) and voice cloning modes.

mod config;
mod predictor;
mod speech_tokenizer;
mod talker;
mod tokenizer;

pub use config::{CodePredictorConfig, Qwen3TtsConfig, TalkerConfig};
pub use predictor::{CodePredictor, CodePredictorCache};
pub use speech_tokenizer::SpeechTokenizerDecoder;
pub use talker::{TalkerCache, TalkerModel};
pub use tokenizer::{SpeakerReference, TtsSpecialTokens, TtsTokenizer};

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

use crate::error::{Error, Result};
use crate::models::device::DeviceProfile;

const NEWLINE_TOKEN_ID: u32 = 198;

/// Runtime generation settings for semantic token sampling.
#[derive(Debug, Clone)]
pub struct TtsGenerationParams {
    /// Semantic token temperature. <= 0 means greedy.
    pub temperature: f32,
    /// Top-p nucleus sampling threshold.
    pub top_p: f32,
    /// Top-k sampling cutoff. 0 means disabled.
    pub top_k: usize,
    /// Repetition penalty for previously sampled semantic tokens.
    pub repetition_penalty: f32,
    /// Maximum generated codec frames. `0` means auto (model maximum).
    pub max_frames: usize,
}

impl Default for TtsGenerationParams {
    fn default() -> Self {
        // Mirrors the official generation_config defaults.
        Self {
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.05,
            max_frames: 512,
        }
    }
}

/// Batch input for CustomVoice (preset speaker) generation.
#[derive(Debug, Clone)]
pub struct BatchedSpeakerRequest {
    pub text: String,
    pub speaker: String,
    pub language: Option<String>,
    pub instruct: Option<String>,
    pub params: TtsGenerationParams,
}

impl TtsGenerationParams {
    /// Convert external generation config to TTS sampling params.
    pub fn from_generation_config(cfg: &crate::runtime::GenerationConfig) -> Self {
        Self {
            temperature: cfg.temperature.max(0.0),
            top_p: cfg.top_p.clamp(0.0, 1.0),
            top_k: if cfg.top_k == 0 { 50 } else { cfg.top_k },
            repetition_penalty: cfg.repetition_penalty.max(1.0),
            max_frames: if cfg.max_tokens == 0 {
                // Keep unconstrained requests within a practical latency envelope.
                512
            } else {
                cfg.max_tokens.clamp(16, 8192)
            },
        }
    }
}

/// Qwen3-TTS Model for speech synthesis
pub struct Qwen3TtsModel {
    /// Device configuration
    device: DeviceProfile,
    /// Data type for inference
    dtype: DType,
    /// Tokenizer for text and codec tokens
    tokenizer: TtsTokenizer,
    /// Special token IDs
    specials: TtsSpecialTokens,
    /// Main talker (LLM) model
    talker: TalkerModel,
    /// Code predictor for multi-codebook generation
    code_predictor: CodePredictor,
    /// Speech tokenizer decoder for codec to audio conversion
    speech_tokenizer: SpeechTokenizerDecoder,
    /// Model configuration
    config: Qwen3TtsConfig,
}

impl Qwen3TtsModel {
    /// Load a Qwen3-TTS model from the specified directory
    pub fn load(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        info!("Loading Qwen3-TTS model from {:?}", model_dir);

        // Load configuration
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Qwen3TtsConfig = serde_json::from_str(&config_str)?;

        info!("Model type: {}", config.tts_model_type);
        info!("Model size: {}", config.tts_model_size);

        // BF16 matmul coverage is incomplete across backends in Candle.
        // Keep TTS inference on F32 for correctness/stability.
        let dtype = DType::F32;

        // Load tokenizer
        let specials = TtsSpecialTokens::from_configs(&config, &config.talker_config);
        let tokenizer = TtsTokenizer::load(model_dir, specials.clone(), &config.talker_config)?;

        // Load model weights
        let weights_path = model_dir.join("model.safetensors");
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device.device)? };

        // Load talker model
        info!("Loading talker model...");
        let talker = TalkerModel::load(config.talker_config.clone(), vb.pp("talker"))?;

        // Load code predictor
        info!("Loading code predictor...");
        let num_code_groups = config.talker_config.num_code_groups;
        // For 1.7B model, codec embeddings use talker.text_hidden_size (2048)
        // For 0.6B model, codec embeddings use code_predictor.hidden_size (1024)
        // Detect 1.7B by checking if talker.hidden_size differs from code_predictor.hidden_size
        let mut code_predictor_config = config.talker_config.code_predictor_config.clone();
        if config.talker_config.hidden_size != code_predictor_config.hidden_size {
            // 1.7B case: codec embeddings use text_hidden_size dimension
            code_predictor_config.text_hidden_size = Some(config.talker_config.text_hidden_size);
        }
        let code_predictor = CodePredictor::load(
            code_predictor_config,
            vb.pp("talker.code_predictor"),
            num_code_groups,
        )?;

        // Load speech tokenizer decoder
        info!("Loading speech tokenizer decoder...");
        let speech_tokenizer_path = model_dir.join("speech_tokenizer");
        let speech_tokenizer =
            SpeechTokenizerDecoder::load(&speech_tokenizer_path, device.device.clone(), dtype)?;

        info!("Qwen3-TTS model loaded successfully on {:?}", device.kind);

        Ok(Self {
            device,
            dtype,
            tokenizer,
            specials,
            talker,
            code_predictor,
            speech_tokenizer,
            config,
        })
    }

    /// Generate speech using a preset speaker (CustomVoice mode)
    pub fn generate_with_speaker(
        &self,
        text: &str,
        speaker: &str,
        language: Option<&str>,
        instruct: Option<&str>,
    ) -> Result<Vec<f32>> {
        self.generate_with_speaker_params(
            text,
            speaker,
            language,
            instruct,
            &TtsGenerationParams::default(),
        )
    }

    /// Generate speech with a preset speaker (CustomVoice mode) and explicit sampling params.
    pub fn generate_with_speaker_params(
        &self,
        text: &str,
        speaker: &str,
        language: Option<&str>,
        instruct: Option<&str>,
        params: &TtsGenerationParams,
    ) -> Result<Vec<f32>> {
        info!("Generating speech with speaker: {}", speaker);

        let prompt_ids = self.encode_assistant_prompt_ids(text)?;
        let speaker_id = self.tokenizer.get_speaker_id(speaker).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Unknown speaker '{speaker}'. Available speakers: {}",
                self.tokenizer
                    .available_speakers()
                    .into_iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
        let language_id = self.resolve_language_id(language);
        let instruct_ids = self.encode_instruction_ids(instruct)?;

        debug!(
            "Prompt token length: {}, speaker_id: {}, language_id: {:?}, has_instruction: {}",
            prompt_ids.len(),
            speaker_id,
            language_id,
            instruct_ids.is_some()
        );

        let codec_tokens = self.generate_codec_tokens_conditioned(
            &prompt_ids,
            Some(speaker_id),
            language_id,
            instruct_ids.as_deref(),
            params,
        )?;

        // Decode to audio using speech tokenizer
        self.codec_to_audio(&codec_tokens)
    }

    /// Generate speech for a batch of preset-speaker requests.
    pub fn generate_with_speaker_params_batch(
        &self,
        requests: &[BatchedSpeakerRequest],
    ) -> Result<Vec<Vec<f32>>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        #[derive(Debug)]
        struct PreparedRequest {
            index: usize,
            params: TtsGenerationParams,
            prefill_embeds: Tensor,
            prefill_len: usize,
            trailing_text_hidden: Tensor,
            trailing_text_len: usize,
            tts_pad_embed: Tensor,
            max_frames: usize,
        }

        let text_vocab_size = self.tokenizer.text_vocab_size() as u32;
        let acoustic_vocab_size = self.tokenizer.codec_vocab_size() as u32;
        let talker_codec_vocab_size = self.config.talker_config.vocab_size as u32;
        let semantic_vocab_size = talker_codec_vocab_size.saturating_sub(1024);

        let mut prepared = Vec::with_capacity(requests.len());
        for (idx, req) in requests.iter().enumerate() {
            let prompt_ids = self.encode_assistant_prompt_ids(&req.text)?;
            let speaker_id = self.tokenizer.get_speaker_id(&req.speaker).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Unknown speaker '{}'. Available speakers: {}",
                    req.speaker,
                    self.tokenizer
                        .available_speakers()
                        .into_iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?;
            let language_id = self.resolve_language_id(req.language.as_deref());
            let instruct_ids = self.encode_instruction_ids(req.instruct.as_deref())?;

            let prefill_embeds = self.build_conditioned_prefill_embeddings(
                &prompt_ids,
                Some(speaker_id),
                language_id,
                instruct_ids.as_deref(),
            )?;
            let prefill_len = prefill_embeds.dim(1)?;

            let context_budget = self
                .config
                .talker_config
                .max_position_embeddings
                .saturating_sub(prefill_len + 1);
            if context_budget == 0 {
                return Err(Error::InferenceError(
                    "Input is too long for model context; no room left for audio generation"
                        .to_string(),
                ));
            }
            let max_frames = if req.params.max_frames == 0 {
                context_budget
            } else {
                req.params.max_frames.max(1).min(context_budget)
            };

            let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
                self.build_trailing_text_embeddings_from_prompt(&prompt_ids, max_frames)?;

            prepared.push(PreparedRequest {
                index: idx,
                params: req.params.clone(),
                prefill_embeds,
                prefill_len,
                trailing_text_hidden,
                trailing_text_len,
                tts_pad_embed,
                max_frames,
            });
        }

        let mut groups: HashMap<usize, Vec<PreparedRequest>> = HashMap::new();
        for item in prepared {
            groups.entry(item.prefill_len).or_default().push(item);
        }

        let mut outputs: Vec<Option<Vec<f32>>> = vec![None; requests.len()];

        for (_prefill_len, group) in groups {
            let batch_size = group.len();
            if batch_size == 0 {
                continue;
            }

            let embeds: Vec<Tensor> = group
                .iter()
                .map(|req| req.prefill_embeds.clone())
                .collect();
            let batch_embeds = Tensor::cat(&embeds, 0)?;

            let mut talker_cache = TalkerCache::new(self.talker.num_layers());
            let (last_hidden_batch, last_logits_batch) =
                self.talker
                    .prefill_with_embeds(&batch_embeds, &mut talker_cache, None)?;

            struct DecodeState {
                index: usize,
                params: TtsGenerationParams,
                trailing_text_hidden: Tensor,
                trailing_text_len: usize,
                tts_pad_embed: Tensor,
                max_frames: usize,
                all_code_groups: Vec<Vec<u32>>,
                semantic_history: Vec<u32>,
                last_hidden: Tensor,
                last_logits: Tensor,
                predictor_cache: CodePredictorCache,
                rng: SimpleRng,
                finished: bool,
            }

            let mut states = Vec::with_capacity(batch_size);
            for (batch_idx, req) in group.iter().enumerate() {
                let last_hidden = last_hidden_batch.i(batch_idx)?.unsqueeze(0)?;
                let last_logits = last_logits_batch.i(batch_idx)?.unsqueeze(0)?;
                states.push(DecodeState {
                    index: req.index,
                    params: req.params.clone(),
                    trailing_text_hidden: req.trailing_text_hidden.clone(),
                    trailing_text_len: req.trailing_text_len,
                    tts_pad_embed: req.tts_pad_embed.clone(),
                    max_frames: req.max_frames,
                    all_code_groups: vec![Vec::new(); self.config.talker_config.num_code_groups],
                    semantic_history: Vec::new(),
                    last_hidden,
                    last_logits,
                    predictor_cache: CodePredictorCache::new(self.code_predictor.num_layers()),
                    rng: SimpleRng::new(),
                    finished: false,
                });
            }

            let mut offset = group[0].prefill_len;
            let max_frames = states
                .iter()
                .map(|s| s.max_frames)
                .max()
                .unwrap_or(0);
            let min_tokens_before_eos = 8usize;

            for frame_idx in 0..max_frames {
                let mut step_inputs = Vec::with_capacity(batch_size);
                let mut any_active = false;

                for state in states.iter_mut() {
                    if state.finished || frame_idx >= state.max_frames {
                        step_inputs.push(state.tts_pad_embed.clone());
                        continue;
                    }

                    any_active = true;
                    let allow_eos = state.all_code_groups[0].len() >= min_tokens_before_eos;
                    let semantic_token = sample_semantic(
                        &state.last_logits.i((0, 0))?,
                        semantic_vocab_size,
                        self.specials.codec_eos_token_id,
                        allow_eos,
                        &state.params,
                        &state.semantic_history,
                        &mut state.rng,
                    )?;
                    if allow_eos && semantic_token == self.specials.codec_eos_token_id {
                        state.finished = true;
                        step_inputs.push(state.tts_pad_embed.clone());
                        continue;
                    }

                    state.semantic_history.push(semantic_token);
                    if state.semantic_history.len() > 256 {
                        let drain = state.semantic_history.len() - 256;
                        state.semantic_history.drain(0..drain);
                    }

                    state.all_code_groups[0].push(text_vocab_size + semantic_token);

                    let semantic_embed = self.talker.get_codec_embedding(semantic_token)?;
                    let acoustic_codes = self.code_predictor.generate_acoustic_codes(
                        &state.last_hidden,
                        &semantic_embed,
                        &mut state.predictor_cache,
                    )?;
                    let acoustic_embed_sum = self
                        .code_predictor
                        .get_acoustic_embeddings_sum(&acoustic_codes)?;
                    for (acoustic_idx, &group_token) in acoustic_codes.iter().enumerate() {
                        let group_idx = acoustic_idx + 1;
                        if group_idx < state.all_code_groups.len() {
                            let combined_token = text_vocab_size
                                + group_token
                                + (group_idx as u32 * acoustic_vocab_size);
                            state.all_code_groups[group_idx].push(combined_token);
                        }
                    }

                    let text_addition = if frame_idx < state.trailing_text_len {
                        state
                            .trailing_text_hidden
                            .i((.., frame_idx..frame_idx + 1, ..))?
                    } else {
                        state.tts_pad_embed.clone()
                    };

                    let step_input = semantic_embed
                        .broadcast_add(&acoustic_embed_sum)?
                        .broadcast_add(&text_addition)?;
                    step_inputs.push(step_input);
                }

                if !any_active {
                    break;
                }

                let batch_input = Tensor::cat(&step_inputs, 0)?;
                let (batch_hidden, batch_logits) =
                    self.talker
                        .generate_step_with_embed(&batch_input, &mut talker_cache, offset)?;

                for (idx, state) in states.iter_mut().enumerate() {
                    state.last_hidden = batch_hidden.i(idx)?.unsqueeze(0)?;
                    state.last_logits = batch_logits.i(idx)?.unsqueeze(0)?;
                }

                offset += 1;
            }

            for state in states {
                let samples = self.codec_to_audio(&state.all_code_groups)?;
                outputs[state.index] = Some(samples);
            }
        }

        Ok(outputs
            .into_iter()
            .map(|out| {
                out.ok_or_else(|| {
                    Error::InferenceError("Missing output for batched request".to_string())
                })
            })
            .collect::<Result<Vec<_>>>()?)
    }

    /// Generate speech with voice cloning
    pub fn generate_with_voice_clone(
        &self,
        text: &str,
        reference: &SpeakerReference,
        language: Option<&str>,
    ) -> Result<Vec<f32>> {
        info!("Generating speech with voice cloning");

        let ref_codec_tokens = self.encode_reference_audio(reference)?;
        if ref_codec_tokens.is_empty() || ref_codec_tokens[0].is_empty() {
            return Err(Error::ModelError(
                "Voice cloning reference encoder produced no conditioning tokens".to_string(),
            ));
        }

        // Build input sequence with reference tokens
        let input_ids = self.tokenizer.build_voice_clone_sequence(
            text,
            reference.text.as_str(),
            &ref_codec_tokens,
            language,
            false,
        )?;

        let params = TtsGenerationParams::default();

        // Generate codec tokens
        let codec_tokens =
            self.generate_codec_tokens_legacy(&input_ids, params.max_frames, &params)?;

        // Decode to audio
        self.codec_to_audio(&codec_tokens)
    }

    /// Generate speech without requiring a preset speaker table.
    ///
    /// This path is used for base checkpoints that do not ship `spk_id`
    /// mappings in config, while still supporting plain text synthesis.
    pub fn generate_with_text_params(
        &self,
        text: &str,
        language: Option<&str>,
        instruct: Option<&str>,
        params: &TtsGenerationParams,
    ) -> Result<Vec<f32>> {
        // VoiceDesign checkpoints do not expose preset speaker IDs. For stability,
        // keep this path on legacy decoding and inject instruction text directly
        // into the prompt so style conditioning is still applied.
        let prompt_text = if let Some(ins) = instruct.map(str::trim).filter(|s| !s.is_empty()) {
            format!("Voice design instruction: {ins}\nTarget text: {text}")
        } else {
            text.to_string()
        };
        let input_ids = self
            .tokenizer
            .build_input_sequence(&prompt_text, None, language, false)?;
        let codec_tokens =
            self.generate_codec_tokens_legacy(&input_ids, params.max_frames, params)?;
        self.codec_to_audio(&codec_tokens)
    }

    fn resolve_language_id(&self, language: Option<&str>) -> Option<u32> {
        let normalized = language.map(str::trim).filter(|s| !s.is_empty());
        match normalized {
            Some(lang) if lang.eq_ignore_ascii_case("auto") => None,
            Some(lang) => Some(self.tokenizer.get_language_id(lang)),
            None => None,
        }
    }

    fn encode_instruction_ids(&self, instruct: Option<&str>) -> Result<Option<Vec<u32>>> {
        let Some(text) = instruct.map(str::trim).filter(|s| !s.is_empty()) else {
            return Ok(None);
        };
        // Align instruction prompt shape with upstream VoiceDesign/CustomVoice API.
        let prompt = format!("<|im_start|>user\n{text}<|im_end|>\n");
        let ids = self.tokenizer.encode_text(&prompt, None)?;
        if ids.is_empty() {
            Ok(None)
        } else {
            Ok(Some(ids))
        }
    }

    fn encode_assistant_prompt_ids(&self, text: &str) -> Result<Vec<u32>> {
        // Mirror upstream prompting:
        // <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
        let prompt = format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
        self.tokenizer.encode_text(&prompt, None)
    }

    /// Legacy codec generation path used by not-yet-updated flows (e.g. voice cloning).
    fn generate_codec_tokens_legacy(
        &self,
        input_ids: &[u32],
        max_length: usize,
        params: &TtsGenerationParams,
    ) -> Result<Vec<Vec<u32>>> {
        let mut talker_cache = TalkerCache::new(self.talker.num_layers());
        let mut predictor_cache = CodePredictorCache::new(self.code_predictor.num_layers());
        let text_vocab_size = self.tokenizer.text_vocab_size() as u32;
        let acoustic_vocab_size = self.tokenizer.codec_vocab_size() as u32;
        // Talker logits are over full codec vocab (semantic + control).
        let talker_codec_vocab_size = self.config.talker_config.vocab_size as u32;
        // Official suppression keeps semantic IDs [0, vocab-1024) and EOS.
        let semantic_vocab_size = talker_codec_vocab_size.saturating_sub(1024);

        // Convert input to tensor
        let input_tensor = Tensor::from_vec(
            input_ids.to_vec(),
            (1, input_ids.len()),
            &self.device.device,
        )?;

        // Initial forward pass through talker
        let mut logits = self
            .talker
            .forward(&input_tensor, 0, Some(&mut talker_cache))?;

        // Collect generated tokens
        let mut all_code_groups: Vec<Vec<u32>> =
            vec![Vec::new(); self.config.talker_config.num_code_groups];
        let mut pos = input_ids.len();
        let context_budget = self
            .config
            .talker_config
            .max_position_embeddings
            .saturating_sub(input_ids.len() + 1);
        if context_budget == 0 {
            return Err(Error::InferenceError(
                "Input is too long for model context; no room left for audio generation"
                    .to_string(),
            ));
        }
        let max_length = if max_length == 0 {
            context_budget
        } else {
            max_length.max(1).min(context_budget)
        };
        let min_tokens_before_eos = 8usize;
        let mut rng = SimpleRng::new();
        let mut semantic_history: Vec<u32> = Vec::new();

        for _step in 0..max_length {
            // Get last position logits
            let last_logits = logits.i((0, logits.dim(1)? - 1))?;

            // Sample first codebook token (semantic) from valid semantic ids plus EOS.
            let allow_eos = all_code_groups[0].len() >= min_tokens_before_eos;
            let first_codebook_token = sample_semantic(
                &last_logits,
                semantic_vocab_size,
                self.specials.codec_eos_token_id,
                allow_eos,
                params,
                &semantic_history,
                &mut rng,
            )?;

            // Check for end of sequence
            if allow_eos && first_codebook_token == self.specials.codec_eos_token_id {
                break;
            }
            semantic_history.push(first_codebook_token);
            if semantic_history.len() > 256 {
                let drain = semantic_history.len() - 256;
                semantic_history.drain(0..drain);
            }
            let first_combined_token = text_vocab_size + first_codebook_token;

            // Store semantic token in combined-vocab format.
            all_code_groups[0].push(first_combined_token);

            // Generate remaining codebooks using code predictor
            let first_codebook_tensor =
                Tensor::from_vec(vec![first_codebook_token], (1, 1), &self.device.device)?;
            let predictor_logits = self.code_predictor.forward(
                &first_codebook_tensor,
                pos,
                Some(&mut predictor_cache),
            )?;

            for (acoustic_idx, group_logits) in predictor_logits.iter().enumerate() {
                let group_idx = acoustic_idx + 1;
                if group_idx >= all_code_groups.len() {
                    break;
                }
                let group_token = argmax(&group_logits.i((0, 0))?)?;
                let combined_token =
                    text_vocab_size + group_token + (group_idx as u32 * acoustic_vocab_size);
                all_code_groups[group_idx].push(combined_token);
            }

            // Prepare next input token for talker
            let next_token_tensor =
                Tensor::from_vec(vec![first_combined_token], (1, 1), &self.device.device)?;
            logits = self
                .talker
                .forward(&next_token_tensor, pos, Some(&mut talker_cache))?;
            pos += 1;
        }

        let semantic_len = all_code_groups.first().map(|g| g.len()).unwrap_or(0);
        let semantic_unique = all_code_groups
            .first()
            .map(|g| g.iter().copied().collect::<HashSet<_>>().len())
            .unwrap_or(0);
        let semantic_preview: Vec<u32> = all_code_groups
            .first()
            .map(|g| g.iter().take(24).copied().collect())
            .unwrap_or_default();
        debug!(
            "Voice clone legacy decode stats: semantic_len={}, semantic_unique={}, preview={:?}",
            semantic_len, semantic_unique, semantic_preview
        );

        Ok(all_code_groups)
    }

    /// Official-style CustomVoice generation path:
    /// prefill with fused role/codec/text embeddings, then per-frame semantic+acoustic+trailing-text fusion.
    fn generate_codec_tokens_conditioned(
        &self,
        prompt_ids: &[u32],
        speaker_id: Option<u32>,
        language_id: Option<u32>,
        instruct_ids: Option<&[u32]>,
        params: &TtsGenerationParams,
    ) -> Result<Vec<Vec<u32>>> {
        let mut talker_cache = TalkerCache::new(self.talker.num_layers());
        let mut predictor_cache = CodePredictorCache::new(self.code_predictor.num_layers());
        let text_vocab_size = self.tokenizer.text_vocab_size() as u32;
        let acoustic_vocab_size = self.tokenizer.codec_vocab_size() as u32;
        // Talker logits are over full codec vocab (semantic + control).
        let talker_codec_vocab_size = self.config.talker_config.vocab_size as u32;
        // Official suppression keeps semantic IDs [0, vocab-1024) and EOS.
        let semantic_vocab_size = talker_codec_vocab_size.saturating_sub(1024);

        let prefill_embeds = self.build_conditioned_prefill_embeddings(
            prompt_ids,
            speaker_id,
            language_id,
            instruct_ids,
        )?;
        let prefill_len = prefill_embeds.dim(1)?;
        let (mut last_hidden, mut last_logits) =
            self.talker
                .prefill_with_embeds(&prefill_embeds, &mut talker_cache, None)?;

        let mut all_code_groups: Vec<Vec<u32>> =
            vec![Vec::new(); self.config.talker_config.num_code_groups];
        let min_tokens_before_eos = 8usize;
        let context_budget = self
            .config
            .talker_config
            .max_position_embeddings
            .saturating_sub(prefill_len + 1);
        if context_budget == 0 {
            return Err(Error::InferenceError(
                "Input is too long for model context; no room left for audio generation"
                    .to_string(),
            ));
        }
        let max_frames = if params.max_frames == 0 {
            context_budget
        } else {
            params.max_frames.max(1).min(context_budget)
        };
        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            self.build_trailing_text_embeddings_from_prompt(prompt_ids, max_frames)?;
        let mut offset = prefill_len;
        let mut rng = SimpleRng::new();
        let mut semantic_history: Vec<u32> = Vec::new();

        for frame_idx in 0..max_frames {
            let allow_eos = all_code_groups[0].len() >= min_tokens_before_eos;
            let semantic_token = sample_semantic(
                &last_logits.i((0, 0))?,
                semantic_vocab_size,
                self.specials.codec_eos_token_id,
                allow_eos,
                params,
                &semantic_history,
                &mut rng,
            )?;
            if allow_eos && semantic_token == self.specials.codec_eos_token_id {
                break;
            }
            semantic_history.push(semantic_token);
            if semantic_history.len() > 256 {
                let drain = semantic_history.len() - 256;
                semantic_history.drain(0..drain);
            }

            all_code_groups[0].push(text_vocab_size + semantic_token);

            let semantic_embed = self.talker.get_codec_embedding(semantic_token)?;
            let acoustic_codes = self.code_predictor.generate_acoustic_codes(
                &last_hidden,
                &semantic_embed,
                &mut predictor_cache,
            )?;
            let acoustic_embed_sum = self
                .code_predictor
                .get_acoustic_embeddings_sum(&acoustic_codes)?;
            for (acoustic_idx, &group_token) in acoustic_codes.iter().enumerate() {
                let group_idx = acoustic_idx + 1;
                if group_idx < all_code_groups.len() {
                    let combined_token =
                        text_vocab_size + group_token + (group_idx as u32 * acoustic_vocab_size);
                    all_code_groups[group_idx].push(combined_token);
                }
            }

            let text_addition = if frame_idx < trailing_text_len {
                trailing_text_hidden.i((.., frame_idx..frame_idx + 1, ..))?
            } else {
                tts_pad_embed.clone()
            };

            let step_input = semantic_embed
                .broadcast_add(&acoustic_embed_sum)?
                .broadcast_add(&text_addition)?;
            let (new_hidden, new_logits) =
                self.talker
                    .generate_step_with_embed(&step_input, &mut talker_cache, offset)?;
            last_hidden = new_hidden;
            last_logits = new_logits;
            offset += 1;
        }

        Ok(all_code_groups)
    }

    fn build_conditioned_prefill_embeddings(
        &self,
        prompt_ids: &[u32],
        speaker_id: Option<u32>,
        language_id: Option<u32>,
        instruct_ids: Option<&[u32]>,
    ) -> Result<Tensor> {
        let role_prefix = self.talker.get_projected_text_embeddings(&[
            self.specials.im_start_token_id,
            self.specials.assistant_token_id,
            NEWLINE_TOKEN_ID,
        ])?;

        let mut codec_prefix_ids = Vec::new();
        if let Some(language_id) = language_id {
            codec_prefix_ids.extend_from_slice(&[
                self.specials.codec_think_id,
                self.specials.codec_think_bos_id,
                language_id,
                self.specials.codec_think_eos_id,
            ]);
        } else {
            codec_prefix_ids.extend_from_slice(&[
                self.specials.codec_nothink_id,
                self.specials.codec_think_bos_id,
                self.specials.codec_think_eos_id,
            ]);
        }
        if let Some(speaker_id) = speaker_id {
            codec_prefix_ids.push(speaker_id);
        }
        codec_prefix_ids.push(self.specials.codec_pad_id);
        codec_prefix_ids.push(self.specials.codec_bos_id);

        let prefix_len = codec_prefix_ids.len();
        if prefix_len < 2 {
            return Err(Error::InferenceError(
                "Invalid codec prefix while building conditioned prefill".to_string(),
            ));
        }

        let codec_prefix = self.talker.get_codec_embedding_batch(&codec_prefix_ids)?;
        let codec_without_last = codec_prefix.i((.., ..prefix_len - 1, ..))?;

        let mut tts_overlay_ids = vec![self.specials.tts_pad_token_id; prefix_len - 2];
        tts_overlay_ids.push(self.specials.tts_bos_token_id);
        let tts_overlay = self
            .talker
            .get_projected_text_embeddings(&tts_overlay_ids)?;
        let codec_hidden = tts_overlay.broadcast_add(&codec_without_last)?;

        let mut hidden = Tensor::cat(&[&role_prefix, &codec_hidden], 1)?;

        if let Some(instruct_ids) = instruct_ids {
            if !instruct_ids.is_empty() {
                let instruct_hidden = self.talker.get_projected_text_embeddings(instruct_ids)?;
                hidden = Tensor::cat(&[&instruct_hidden, &hidden], 1)?;
            }
        }

        let first_text_id = if prompt_ids.len() > 3 {
            Some(prompt_ids[3])
        } else {
            prompt_ids.first().copied()
        };

        if let Some(first_text_id) = first_text_id {
            let first_text_proj = self
                .talker
                .get_projected_text_embeddings(&[first_text_id])?;
            let codec_bos_embed = codec_prefix.i((.., prefix_len - 1..prefix_len, ..))?;
            let first_combined = first_text_proj.broadcast_add(&codec_bos_embed)?;
            hidden = Tensor::cat(&[&hidden, &first_combined], 1)?;
        }

        Ok(hidden)
    }

    fn build_trailing_text_embeddings_from_prompt(
        &self,
        prompt_ids: &[u32],
        max_frames: usize,
    ) -> Result<(Tensor, usize, Tensor)> {
        // Upstream uses prompt layout:
        // [role(3), first_text(1), trailing_text(...), trailer(5)]
        let trailing_ids: &[u32] = if prompt_ids.len() > 9 {
            &prompt_ids[4..prompt_ids.len() - 5]
        } else if prompt_ids.len() > 4 {
            &prompt_ids[4..]
        } else {
            &[]
        };

        let trailing = if !trailing_ids.is_empty() {
            let trailing_end = max_frames.min(trailing_ids.len());
            let remaining = self
                .talker
                .get_projected_text_embeddings(&trailing_ids[..trailing_end])?;
            let eos_embed = self
                .talker
                .get_projected_special_embed(self.specials.tts_eos_token_id)?;
            Tensor::cat(&[&remaining, &eos_embed], 1)?
        } else {
            self.talker
                .get_projected_special_embed(self.specials.tts_eos_token_id)?
        };
        let trailing_len = trailing.dim(1)?;
        let tts_pad = self
            .talker
            .get_projected_special_embed(self.specials.tts_pad_token_id)?;
        Ok((trailing, trailing_len, tts_pad))
    }

    /// Encode reference audio to codec tokens for voice cloning
    fn encode_reference_audio(&self, reference: &SpeakerReference) -> Result<Vec<Vec<u32>>> {
        let codec_tokens = self
            .speech_tokenizer
            .encode_reference_audio(&reference.audio_samples, reference.sample_rate)?;
        let num_groups = codec_tokens.len();
        let num_frames = codec_tokens.first().map(|g| g.len()).unwrap_or(0);
        debug!(
            "Reference audio encoded for voice cloning ({} groups x {} frames, transcript chars: {})",
            num_groups,
            num_frames,
            reference.text.len()
        );
        Ok(codec_tokens)
    }

    /// Convert codec tokens to audio waveform
    fn codec_to_audio(&self, codec_tokens: &[Vec<u32>]) -> Result<Vec<f32>> {
        // Convert combined tokens back to raw codec indices for speech tokenizer
        let text_vocab_size = self.tokenizer.text_vocab_size() as u32;
        let codec_vocab_size = self.tokenizer.codec_vocab_size() as u32;

        let mut raw_codec_tokens: Vec<Vec<u32>> = Vec::new();

        for (group_idx, group_tokens) in codec_tokens.iter().enumerate() {
            let mut raw_tokens = Vec::new();
            for &token in group_tokens {
                // Convert combined token back to codec index
                let codec_token = if group_idx == 0 {
                    // First group: token - text_vocab_size
                    if token >= text_vocab_size {
                        token - text_vocab_size
                    } else {
                        token // Already a codec token
                    }
                } else {
                    // Other groups: (token - text_vocab_size) - (group_idx * codec_vocab_size)
                    let offset = text_vocab_size + (group_idx as u32 * codec_vocab_size);
                    if token >= offset {
                        token - offset
                    } else {
                        token
                    }
                };
                raw_tokens.push(codec_token);
            }
            raw_codec_tokens.push(raw_tokens);
        }

        // Decode through speech tokenizer
        let mut audio = self.speech_tokenizer.decode(&raw_codec_tokens)?;
        normalize_audio(&mut audio);
        Ok(audio)
    }

    /// List available preset speakers
    pub fn available_speakers(&self) -> Vec<&String> {
        self.tokenizer.available_speakers()
    }

    /// List available languages
    pub fn available_languages(&self) -> Vec<&String> {
        self.tokenizer.available_languages()
    }

    /// Get the model configuration
    pub fn config(&self) -> &Qwen3TtsConfig {
        &self.config
    }

    /// Get the device
    pub fn device(&self) -> &DeviceProfile {
        &self.device
    }
}

/// Argmax sampling for greedy decoding
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

fn argmax_semantic(
    logits: &Tensor,
    semantic_vocab_size: u32,
    eos_token_id: u32,
    allow_eos: bool,
) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    let values = logits.to_vec1::<f32>()?;
    let mut max_idx: Option<usize> = None;
    let mut max_val = f32::NEG_INFINITY;

    for (idx, &val) in values.iter().enumerate() {
        let token_id = idx as u32;
        let allowed = token_id < semantic_vocab_size || (allow_eos && token_id == eos_token_id);
        if !allowed {
            continue;
        }
        if val > max_val {
            max_val = val;
            max_idx = Some(idx);
        }
    }

    if let Some(idx) = max_idx {
        Ok(idx as u32)
    } else if allow_eos {
        Ok(eos_token_id)
    } else {
        Ok(0)
    }
}

fn sample_semantic(
    logits: &Tensor,
    semantic_vocab_size: u32,
    eos_token_id: u32,
    allow_eos: bool,
    params: &TtsGenerationParams,
    history: &[u32],
    rng: &mut SimpleRng,
) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    let mut values = logits.to_vec1::<f32>()?;

    // Token suppression: keep semantic range and optional EOS only.
    for (idx, v) in values.iter_mut().enumerate() {
        let token_id = idx as u32;
        let allowed = token_id < semantic_vocab_size || (allow_eos && token_id == eos_token_id);
        if !allowed {
            *v = f32::NEG_INFINITY;
        }
    }

    // Repetition penalty over recent semantic history.
    if params.repetition_penalty > 1.0 && !history.is_empty() {
        let mut seen = vec![false; values.len()];
        for &token in history {
            let idx = token as usize;
            if idx < seen.len() {
                seen[idx] = true;
            }
        }
        for (idx, seen_flag) in seen.iter().enumerate() {
            if !*seen_flag {
                continue;
            }
            let v = &mut values[idx];
            if !v.is_finite() {
                continue;
            }
            if *v > 0.0 {
                *v /= params.repetition_penalty;
            } else {
                *v *= params.repetition_penalty;
            }
        }
    }

    // Greedy fallback when sampling is effectively disabled.
    if params.temperature <= 1e-5 {
        return argmax_semantic(&logits, semantic_vocab_size, eos_token_id, allow_eos);
    }

    let temperature = params.temperature.max(1e-5);
    for v in values.iter_mut() {
        if v.is_finite() {
            *v /= temperature;
        }
    }

    let mut candidates: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(idx, &v)| if v.is_finite() { Some(idx) } else { None })
        .collect();
    if candidates.is_empty() {
        return Ok(if allow_eos { eos_token_id } else { 0 });
    }

    // Top-k filtering.
    if params.top_k > 0 && params.top_k < candidates.len() {
        candidates.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(Ordering::Equal));
        candidates.truncate(params.top_k);
    }

    let max_logit = candidates
        .iter()
        .map(|&idx| values[idx])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = candidates
        .iter()
        .map(|&idx| (idx, (values[idx] - max_logit).exp()))
        .collect();

    let mut sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return argmax_semantic(&logits, semantic_vocab_size, eos_token_id, allow_eos);
    }
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    // Top-p filtering over normalized probabilities.
    if params.top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let cutoff = params.top_p.max(1e-6);
        let mut cumsum = 0.0f32;
        let mut keep = 0usize;
        for (_, p) in probs.iter() {
            cumsum += *p;
            keep += 1;
            if cumsum >= cutoff {
                break;
            }
        }
        probs.truncate(keep.max(1));
        sum = probs.iter().map(|(_, p)| *p).sum();
        if sum > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= sum;
            }
        }
    }

    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (idx, p) in probs.iter() {
        acc += *p;
        if r <= acc {
            return Ok(*idx as u32);
        }
    }

    // Numerical fallback: pick max probability candidate.
    probs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| *idx as u32)
        .or_else(|| Some(if allow_eos { eos_token_id } else { 0 }))
        .ok_or_else(|| Error::InferenceError("Failed to sample semantic token".to_string()))
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Self {
            state: seed ^ 0xA076_1D64_78BD_642F,
        }
    }

    fn next_u32(&mut self) -> u32 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / (u32::MAX as f64 + 1.0)) as f32
    }
}

fn normalize_audio(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }

    // Drop non-finite values and remove DC offset.
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for s in samples.iter_mut() {
        if !s.is_finite() {
            *s = 0.0;
            continue;
        }
        sum += *s as f64;
        count += 1;
    }
    if count > 0 {
        let mean = (sum / count as f64) as f32;
        for s in samples.iter_mut() {
            *s -= mean;
        }
    }

    // Peak normalize to avoid hard clipping in WAV encoder.
    let mut peak = 0.0f32;
    for &s in samples.iter() {
        let a = s.abs();
        if a > peak {
            peak = a;
        }
    }
    if peak > 0.95 {
        let scale = 0.95 / peak;
        for s in samples.iter_mut() {
            *s *= scale;
        }
    }

    // Keep output loudness within a practical band for WAV playback.
    let mut power = 0.0f64;
    for &s in samples.iter() {
        power += (s as f64) * (s as f64);
    }
    let rms = (power / samples.len() as f64).sqrt() as f32;
    let max_rms = 0.12f32;
    let min_rms = 0.04f32;
    if rms > max_rms && rms > 1e-6 {
        let scale = max_rms / rms;
        for s in samples.iter_mut() {
            *s *= scale;
        }
    } else if rms < min_rms && rms > 1e-6 {
        let scale = (min_rms / rms).min(8.0);
        for s in samples.iter_mut() {
            *s *= scale;
        }

        // Re-apply peak guard after boosting.
        let mut peak = 0.0f32;
        for &s in samples.iter() {
            let a = s.abs();
            if a > peak {
                peak = a;
            }
        }
        if peak > 0.95 {
            let scale = 0.95 / peak;
            for s in samples.iter_mut() {
                *s *= scale;
            }
        }
    }
}

/// Load a Qwen3-TTS model
pub fn load_model(model_path: &Path, device: DeviceProfile) -> Result<Qwen3TtsModel> {
    Qwen3TtsModel::load(model_path, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_tokens_creation() {
        let main_config = Qwen3TtsConfig {
            architectures: vec!["Qwen3TTSForConditionalGeneration".to_string()],
            model_type: "qwen3_tts".to_string(),
            tokenizer_type: "qwen3_tts_tokenizer_12hz".to_string(),
            tts_model_size: "0b6".to_string(),
            tts_model_type: "custom_voice".to_string(),
            assistant_token_id: 77091,
            im_end_token_id: 151645,
            im_start_token_id: 151644,
            tts_bos_token_id: 151672,
            tts_eos_token_id: 151673,
            tts_pad_token_id: 151671,
            talker_config: TalkerConfig {
                model_type: "qwen3_tts_talker".to_string(),
                hidden_size: 1024,
                intermediate_size: 3072,
                num_hidden_layers: 28,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                head_dim: 128,
                max_position_embeddings: 32768,
                vocab_size: 3072,
                text_vocab_size: 151936,
                text_hidden_size: 2048,
                num_code_groups: 16,
                rms_norm_eps: 1e-6,
                rope_theta: 1_000_000.0,
                hidden_act: "silu".to_string(),
                use_cache: true,
                position_id_per_seconds: 13,
                rope_scaling: None,
                sliding_window: None,
                code_predictor_config: CodePredictorConfig {
                    model_type: "qwen3_tts_talker_code_predictor".to_string(),
                    hidden_size: 1024,
                    intermediate_size: 3072,
                    num_hidden_layers: 5,
                    num_attention_heads: 16,
                    num_key_value_heads: 8,
                    head_dim: 128,
                    max_position_embeddings: 65536,
                    vocab_size: 2048,
                    num_code_groups: 16,
                    rms_norm_eps: 1e-6,
                    rope_theta: 1_000_000.0,
                    hidden_act: "silu".to_string(),
                    use_cache: true,
                    layer_types: vec![],
                    text_hidden_size: None,
                },
                codec_bos_id: 2149,
                codec_eos_token_id: 2150,
                codec_think_id: 2154,
                codec_nothink_id: 2155,
                codec_pad_id: 2148,
                codec_think_bos_id: 2156,
                codec_think_eos_id: 2157,
                spk_id: std::collections::HashMap::new(),
                spk_is_dialect: std::collections::HashMap::new(),
                codec_language_id: std::collections::HashMap::new(),
            },
        };

        let specials = TtsSpecialTokens::from_configs(&main_config, &main_config.talker_config);
        assert_eq!(specials.codec_bos_id, 2149);
        assert_eq!(specials.codec_eos_token_id, 2150);
    }
}
