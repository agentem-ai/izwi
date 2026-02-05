//! Native Qwen3-TTS model loader and inference.
//!
//! This module provides native Rust implementation for Qwen3-TTS models,
//! supporting both CustomVoice (preset speakers) and voice cloning modes.

mod config;
mod predictor;
mod talker;
mod tokenizer;

pub use config::{CodePredictorConfig, Qwen3TtsConfig, TalkerConfig};
pub use predictor::{CodePredictor, CodePredictorCache};
pub use talker::{TalkerCache, TalkerModel};
pub use tokenizer::{SpeakerReference, TtsSpecialTokens, TtsTokenizer};

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;
use tracing::{debug, info};

use crate::error::{Error, Result};
use crate::models::device::DeviceProfile;

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

        // Setup dtype based on device
        let dtype = if device.kind.is_cpu() {
            DType::F32
        } else {
            // Use BF16 for GPU/Metal if supported, otherwise F16
            DType::BF16
        };

        // Load tokenizer
        let specials = TtsSpecialTokens::from_configs(&config, &config.talker_config);
        let tokenizer = TtsTokenizer::load(model_dir, specials.clone(), &config.talker_config)?;

        // Load model weights
        let weights_path = model_dir.join("model.safetensors");
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device.device)? };

        // Load talker model
        info!("Loading talker model...");
        let talker = TalkerModel::load(config.talker_config.clone(), vb.pp("talker.model"))?;

        // Load code predictor
        info!("Loading code predictor...");
        let num_code_groups = config.talker_config.num_code_groups;
        let code_predictor = CodePredictor::load(
            config.talker_config.code_predictor_config.clone(),
            vb.pp("talker.code_predictor.model"),
            num_code_groups,
        )?;

        info!("Qwen3-TTS model loaded successfully on {:?}", device.kind);

        Ok(Self {
            device,
            dtype,
            tokenizer,
            specials,
            talker,
            code_predictor,
            config,
        })
    }

    /// Generate speech using a preset speaker (CustomVoice mode)
    pub fn generate_with_speaker(
        &self,
        text: &str,
        speaker: &str,
        language: Option<&str>,
        _instruct: Option<&str>,
    ) -> Result<Vec<f32>> {
        info!("Generating speech with speaker: {}", speaker);

        // Build input sequence
        let input_ids =
            self.tokenizer
                .build_input_sequence(text, Some(speaker), language, false)?;

        debug!("Input sequence length: {}", input_ids.len());

        // Generate codec tokens
        let codec_tokens = self.generate_codec_tokens(&input_ids)?;

        // Decode to audio using speech tokenizer
        // For now, return empty - audio decoding will be implemented separately
        let _audio = self.codec_to_audio(&codec_tokens)?;

        // TODO: Implement proper audio decoding with speech tokenizer
        // This requires loading and using the separate speech tokenizer model
        Err(Error::ModelError(
            "Audio decoding not yet implemented - speech tokenizer integration required"
                .to_string(),
        ))
    }

    /// Generate speech with voice cloning
    pub fn generate_with_voice_clone(
        &self,
        text: &str,
        reference: &SpeakerReference,
        language: Option<&str>,
    ) -> Result<Vec<f32>> {
        info!("Generating speech with voice cloning");

        // Encode reference audio to codec tokens
        // This requires the speech tokenizer encoder
        let ref_codec_tokens = self.encode_reference_audio(reference)?;

        // Build input sequence with reference tokens
        let input_ids =
            self.tokenizer
                .build_voice_clone_sequence(text, &ref_codec_tokens, language, false)?;

        // Generate codec tokens
        let codec_tokens = self.generate_codec_tokens(&input_ids)?;

        // Decode to audio
        self.codec_to_audio(&codec_tokens)
    }

    /// Generate codec tokens using the talker and code predictor
    fn generate_codec_tokens(&self, input_ids: &[u32]) -> Result<Vec<Vec<u32>>> {
        let mut talker_cache = TalkerCache::new(self.talker.num_layers());
        let mut predictor_cache = CodePredictorCache::new(self.code_predictor.num_layers());

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
        let max_length = 2048; // Maximum audio length in tokens

        for _step in 0..max_length {
            // Get last position logits
            let last_logits = logits.i((0, logits.dim(1)? - 1))?;

            // Sample first codebook token (semantic)
            let first_codebook_token = argmax(&last_logits)?;

            // Check for end of sequence
            if first_codebook_token >= self.tokenizer.text_vocab_size() as u32 {
                let codec_token = first_codebook_token - self.tokenizer.text_vocab_size() as u32;
                if codec_token == self.specials.codec_eos_token_id {
                    break;
                }
            }

            // Add first codebook token
            all_code_groups[0].push(first_codebook_token);

            // Generate remaining codebooks using code predictor
            let first_codebook_tensor =
                Tensor::from_vec(vec![first_codebook_token], (1, 1), &self.device.device)?;
            let predictor_logits = self.code_predictor.forward(
                &first_codebook_tensor,
                pos,
                Some(&mut predictor_cache),
            )?;

            // Sample from each code group's logits
            for (group_idx, group_logits) in predictor_logits.iter().enumerate().skip(1) {
                let group_token = argmax(&group_logits.i((0, 0))?)?;
                // Offset token by text_vocab_size for combined vocab
                let combined_token = self.tokenizer.text_vocab_size() as u32
                    + group_token
                    + (group_idx as u32 * self.tokenizer.codec_vocab_size() as u32);
                all_code_groups[group_idx].push(combined_token);
            }

            // Prepare next input token for talker
            let next_token_tensor =
                Tensor::from_vec(vec![first_codebook_token], (1, 1), &self.device.device)?;
            logits = self
                .talker
                .forward(&next_token_tensor, pos, Some(&mut talker_cache))?;
            pos += 1;
        }

        Ok(all_code_groups)
    }

    /// Encode reference audio to codec tokens for voice cloning
    fn encode_reference_audio(&self, _reference: &SpeakerReference) -> Result<Vec<Vec<u32>>> {
        // This requires the speech tokenizer encoder model
        // For now, return empty - this will be implemented with the speech tokenizer
        Err(Error::ModelError(
            "Reference audio encoding not yet implemented".to_string(),
        ))
    }

    /// Convert codec tokens to audio waveform
    fn codec_to_audio(&self, _codec_tokens: &[Vec<u32>]) -> Result<Vec<f32>> {
        // This requires the speech tokenizer decoder (Code2Wav)
        // For now, return empty - this will be implemented with the speech tokenizer
        Err(Error::ModelError(
            "Codec to audio decoding not yet implemented".to_string(),
        ))
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
