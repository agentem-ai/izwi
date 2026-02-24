//! Kokoro-82M native runtime integration scaffolding (Rust-only).
//!
//! This module intentionally isolates Kokoro-specific loading, phonemization,
//! voice-pack handling, and future Candle inference implementation from the
//! generic runtime orchestration layer.

mod albert;
mod config;
mod decoder;
mod phonemizer;
mod prosody;
mod text_encoder;
mod voice;

pub use config::KokoroConfig;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::pickle::read_pth_tensor_info;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use tracing::info;

use crate::error::{Error, Result};
use crate::models::shared::device::DeviceProfile;

use self::phonemizer::EspeakPhonemizer;
use self::prosody::{
    build_alignment_matrix, KokoroProsodyDebugOutput, KokoroProsodyOutput, KokoroProsodyPredictor,
};
use self::text_encoder::KokoroTextEncoder;
use self::voice::VoiceLibrary;

const CHECKPOINT_FILE: &str = "kokoro-v1_0.pth";
const CONFIG_FILE: &str = "config.json";
const VOICES_DIR: &str = "voices";
const CHECKPOINT_SUBMODULE_KEYS: &[&str] = &["bert", "bert_encoder", "predictor", "text_encoder", "decoder"];

#[derive(Debug, Clone)]
pub struct KokoroPreparedRequest {
    pub phonemes: String,
    pub token_ids: Vec<u32>,
    pub ref_style: Tensor,
    pub speed: f32,
}

#[derive(Debug, Clone)]
pub struct KokoroSynthesisResult {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub tokens_generated: usize,
    pub phonemes: String,
}

#[derive(Debug, Clone)]
pub struct KokoroPredecoderDebugOutput {
    pub prosody: KokoroProsodyDebugOutput,
    pub text_encoder_shape: Vec<usize>,
    pub asr_shape: Vec<usize>,
}

#[derive(Debug, Clone)]
struct KokoroPredecoderOutput {
    prosody: KokoroProsodyOutput,
    text_encoder_shape: Vec<usize>,
    asr: Tensor,
}

#[derive(Debug)]
pub struct KokoroTtsModel {
    model_dir: PathBuf,
    checkpoint_path: PathBuf,
    config: KokoroConfig,
    device: DeviceProfile,
    dtype: DType,
    bert: albert::CustomAlbert,
    bert_encoder: Linear,
    prosody: KokoroProsodyPredictor,
    text_encoder: KokoroTextEncoder,
    decoder: decoder::KokoroDecoder,
    phonemizer: EspeakPhonemizer,
    voices: VoiceLibrary,
    checkpoint_tensor_counts: HashMap<String, usize>,
}

impl KokoroTtsModel {
    pub fn load(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        let config_path = model_dir.join(CONFIG_FILE);
        let checkpoint_path = model_dir.join(CHECKPOINT_FILE);
        let voices_dir = model_dir.join(VOICES_DIR);

        if !config_path.exists() {
            return Err(Error::ModelNotFound(format!(
                "Missing Kokoro config.json at {}",
                config_path.display()
            )));
        }
        if !checkpoint_path.exists() {
            return Err(Error::ModelNotFound(format!(
                "Missing Kokoro checkpoint at {}",
                checkpoint_path.display()
            )));
        }
        if !voices_dir.exists() {
            return Err(Error::ModelNotFound(format!(
                "Missing Kokoro voices directory at {}",
                voices_dir.display()
            )));
        }

        let config: KokoroConfig =
            serde_json::from_str(&std::fs::read_to_string(&config_path).map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed reading Kokoro config {}: {}",
                    config_path.display(),
                    e
                ))
            })?)?;

        let dtype = DType::F32;
        let checkpoint_tensor_counts =
            inspect_and_validate_checkpoint(&checkpoint_path, &device.device, dtype)?;

        let phonemizer = EspeakPhonemizer::auto()?;
        let voices = VoiceLibrary::new(voices_dir, device.device.clone(), dtype)?;
        let bert = {
            let vb = VarBuilder::from_pth_with_state(&checkpoint_path, dtype, "bert", &device.device)
                .map_err(|e| {
                    Error::ModelLoadError(format!(
                        "Failed to create Kokoro BERT VarBuilder for {}: {}",
                        checkpoint_path.display(),
                        e
                    ))
                })?;
            albert::CustomAlbert::load(&albert::AlbertModelConfig::from_kokoro(&config), vb.pp("module"))?
        };
        let bert_encoder = {
            let vb = VarBuilder::from_pth_with_state(
                &checkpoint_path,
                dtype,
                "bert_encoder",
                &device.device,
            )
            .map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to create Kokoro bert_encoder VarBuilder for {}: {}",
                    checkpoint_path.display(),
                    e
                ))
            })?;
            candle_nn::linear(
                config.plbert.hidden_size,
                config.hidden_dim,
                vb.pp("module"),
            )
            .map_err(Error::from)?
        };
        let prosody = {
            let vb = VarBuilder::from_pth_with_state(
                &checkpoint_path,
                dtype,
                "predictor",
                &device.device,
            )
            .map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to create Kokoro predictor VarBuilder for {}: {}",
                    checkpoint_path.display(),
                    e
                ))
            })?;
            KokoroProsodyPredictor::load(&config, vb)?
        };
        let text_encoder = {
            let vb = VarBuilder::from_pth_with_state(
                &checkpoint_path,
                dtype,
                "text_encoder",
                &device.device,
            )
            .map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to create Kokoro text_encoder VarBuilder for {}: {}",
                    checkpoint_path.display(),
                    e
                ))
            })?;
            KokoroTextEncoder::load(&config, vb)?
        };
        let decoder = {
            let vb = VarBuilder::from_pth_with_state(
                &checkpoint_path,
                dtype,
                "decoder",
                &device.device,
            )
            .map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to create Kokoro decoder VarBuilder for {}: {}",
                    checkpoint_path.display(),
                    e
                ))
            })?;
            decoder::KokoroDecoder::load(&config, vb)?
        };

        info!(
            "Loaded Kokoro scaffolding from {:?} (phonemizer={}, submodules={:?})",
            model_dir,
            phonemizer.bin_path().display(),
            checkpoint_tensor_counts
        );

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            checkpoint_path,
            config,
            device,
            dtype,
            bert,
            bert_encoder,
            prosody,
            text_encoder,
            decoder,
            phonemizer,
            voices,
            checkpoint_tensor_counts,
        })
    }

    pub fn available_speakers(&self) -> Result<Vec<String>> {
        self.voices.list_speakers()
    }

    pub fn prepare_request(
        &self,
        text: &str,
        speaker: Option<&str>,
        language: Option<&str>,
        speed: f32,
    ) -> Result<KokoroPreparedRequest> {
        let speaker = self.resolve_speaker(speaker)?;
        let phonemes = self.phonemizer.phonemize(text, language, Some(&speaker))?;
        let phoneme_len = phonemes.chars().count();
        if phoneme_len == 0 {
            return Err(Error::InvalidInput(
                "Kokoro phonemizer produced no phonemes".to_string(),
            ));
        }
        if phoneme_len > 510 {
            return Err(Error::InvalidInput(format!(
                "Kokoro phoneme sequence length {} exceeds supported voice-pack limit (510). Chunking is not implemented yet in the native runtime.",
                phoneme_len
            )));
        }

        let token_ids = self.token_ids_from_phonemes(&phonemes)?;
        if token_ids.len() + 2 > self.config.context_length() {
            return Err(Error::InvalidInput(format!(
                "Kokoro phoneme token length {} exceeds context length {}",
                token_ids.len() + 2,
                self.config.context_length()
            )));
        }

        let ref_style = self.voices.style_for_phoneme_len(&speaker, phoneme_len)?;
        let speed = speed.clamp(0.5, 2.0);

        Ok(KokoroPreparedRequest {
            phonemes,
            token_ids,
            ref_style,
            speed,
        })
    }

    pub fn generate(
        &self,
        text: &str,
        speaker: Option<&str>,
        language: Option<&str>,
        speed: f32,
    ) -> Result<KokoroSynthesisResult> {
        let prepared = self.prepare_request(text, speaker, language, speed)?;
        let predecoder = self.run_predecoder(&prepared)?;
        let style = prepared
            .ref_style
            .i((.., 0..self.config.style_dim))
            .map_err(Error::from)?;
        let samples = self.decoder.forward(
            &predecoder.asr,
            &predecoder.prosody.f0,
            &predecoder.prosody.n,
            &style,
        )?;
        Ok(KokoroSynthesisResult {
            tokens_generated: prepared.token_ids.len(),
            phonemes: prepared.phonemes,
            sample_rate: KokoroConfig::TARGET_SAMPLE_RATE,
            samples,
        })
    }

    pub fn config(&self) -> &KokoroConfig {
        &self.config
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn checkpoint_path(&self) -> &Path {
        &self.checkpoint_path
    }

    pub fn checkpoint_tensor_counts(&self) -> &HashMap<String, usize> {
        &self.checkpoint_tensor_counts
    }

    pub fn run_bert_prosody_debug(
        &self,
        prepared: &KokoroPreparedRequest,
    ) -> Result<KokoroProsodyDebugOutput> {
        let input_ids = self.build_model_input_ids(prepared)?;
        let (_b, seq_len) = input_ids.dims2().map_err(Error::from)?;
        let attention_mask = Tensor::ones((1, seq_len), DType::U32, &self.device.device)?;
        let bert_hidden = self.bert.forward(&input_ids, Some(&attention_mask))?;
        let d_en = self
            .bert_encoder
            .forward(&bert_hidden)
            .map_err(Error::from)?
            .transpose(1, 2)
            .map_err(Error::from)?;
        self.prosody
            .forward_debug(&d_en, &prepared.ref_style, prepared.speed)
    }

    fn run_bert_prosody(
        &self,
        prepared: &KokoroPreparedRequest,
    ) -> Result<KokoroProsodyOutput> {
        let input_ids = self.build_model_input_ids(prepared)?;
        let (_b, seq_len) = input_ids.dims2().map_err(Error::from)?;
        let attention_mask = Tensor::ones((1, seq_len), DType::U32, &self.device.device)?;
        let bert_hidden = self.bert.forward(&input_ids, Some(&attention_mask))?;
        let d_en = self
            .bert_encoder
            .forward(&bert_hidden)
            .map_err(Error::from)?
            .transpose(1, 2)
            .map_err(Error::from)?;
        self.prosody.forward(&d_en, &prepared.ref_style, prepared.speed)
    }

    pub fn run_predecoder_debug(
        &self,
        prepared: &KokoroPreparedRequest,
    ) -> Result<KokoroPredecoderDebugOutput> {
        let out = self.run_predecoder(prepared)?;
        Ok(KokoroPredecoderDebugOutput {
            prosody: KokoroProsodyDebugOutput {
                duration_frames: out.prosody.duration_frames.clone(),
                expanded_frames: out.prosody.expanded_frames,
                f0_shape: out.prosody.f0.shape().dims().to_vec(),
                n_shape: out.prosody.n.shape().dims().to_vec(),
            },
            text_encoder_shape: out.text_encoder_shape,
            asr_shape: out.asr.shape().dims().to_vec(),
        })
    }

    fn run_predecoder(&self, prepared: &KokoroPreparedRequest) -> Result<KokoroPredecoderOutput> {
        let input_ids = self.build_model_input_ids(prepared)?;
        let prosody = self.run_bert_prosody(prepared)?;
        let pred_aln = build_alignment_matrix(&prosody.duration_frames, &self.device.device)?;
        let t_en = self.text_encoder.forward(&input_ids)?;
        let asr = t_en
            .contiguous()
            .map_err(Error::from)?
            .matmul(&pred_aln.contiguous().map_err(Error::from)?)
            .map_err(Error::from)?;
        let text_encoder_shape = t_en.shape().dims().to_vec();
        Ok(KokoroPredecoderOutput {
            prosody,
            text_encoder_shape,
            asr,
        })
    }

    fn build_model_input_ids(&self, prepared: &KokoroPreparedRequest) -> Result<Tensor> {
        let mut input_ids = Vec::with_capacity(prepared.token_ids.len() + 2);
        input_ids.push(0u32);
        input_ids.extend_from_slice(&prepared.token_ids);
        input_ids.push(0u32);
        let seq_len = input_ids.len();
        Tensor::from_vec(input_ids, (1, seq_len), &self.device.device).map_err(Error::from)
    }

    fn resolve_speaker(&self, requested: Option<&str>) -> Result<String> {
        let speakers = self.available_speakers()?;
        if speakers.is_empty() {
            return Err(Error::ModelLoadError(
                "Kokoro voices directory is empty".to_string(),
            ));
        }
        let requested = requested
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("af_heart");
        if let Some(exact) = speakers.iter().find(|s| s.as_str() == requested) {
            return Ok(exact.clone());
        }
        let requested_lower = requested.to_ascii_lowercase();
        if let Some(casefold) = speakers
            .iter()
            .find(|s| s.to_ascii_lowercase() == requested_lower)
        {
            return Ok(casefold.clone());
        }
        Err(Error::InvalidInput(format!(
            "Unknown Kokoro speaker '{requested}'. Available speakers: {}",
            speakers.join(", ")
        )))
    }

    fn token_ids_from_phonemes(&self, phonemes: &str) -> Result<Vec<u32>> {
        let mut token_ids = Vec::with_capacity(phonemes.chars().count());
        let mut unknown = Vec::new();
        for ch in phonemes.chars() {
            let key = ch.to_string();
            if let Some(id) = self.config.vocab.get(&key) {
                token_ids.push(*id);
            } else if ch.is_whitespace() {
                if let Some(id) = self.config.vocab.get(" ") {
                    token_ids.push(*id);
                }
            } else {
                unknown.push(ch);
            }
        }

        if token_ids.is_empty() {
            return Err(Error::TokenizationError(format!(
                "Kokoro phoneme tokenizer produced zero tokens (unknown chars: {:?})",
                unknown
            )));
        }

        if !unknown.is_empty() {
            tracing::warn!(
                "Kokoro phoneme tokenizer skipped {} unknown symbols: {:?}",
                unknown.len(),
                unknown
            );
        }

        Ok(token_ids)
    }
}

fn inspect_and_validate_checkpoint(
    checkpoint_path: &Path,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<HashMap<String, usize>> {
    let mut counts = HashMap::new();
    for key in CHECKPOINT_SUBMODULE_KEYS {
        let infos = read_pth_tensor_info(checkpoint_path, false, Some(key)).map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed to inspect Kokoro checkpoint submodule '{key}' in {}: {}",
                checkpoint_path.display(),
                e
            ))
        })?;
        if infos.is_empty() {
            return Err(Error::ModelLoadError(format!(
                "Kokoro checkpoint submodule '{key}' in {} has no tensors",
                checkpoint_path.display()
            )));
        }
        let _vb = VarBuilder::from_pth_with_state(checkpoint_path, dtype, key, device).map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed to create Candle VarBuilder for Kokoro submodule '{key}' in {}: {}",
                checkpoint_path.display(),
                e
            ))
        })?;
        counts.insert((*key).to_string(), infos.len());
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::shared::device::DeviceSelector;
    use std::path::Path;

    #[test]
    fn kokoro_config_context_length_uses_plbert_positions() {
        let cfg = KokoroConfig {
            istftnet: config::KokoroIstftNetConfig {
                upsample_kernel_sizes: vec![20, 12],
                upsample_rates: vec![10, 6],
                gen_istft_hop_size: 5,
                gen_istft_n_fft: 20,
                resblock_dilation_sizes: vec![vec![1, 3, 5]],
                resblock_kernel_sizes: vec![3],
                upsample_initial_channel: 512,
            },
            dim_in: 64,
            dropout: 0.2,
            hidden_dim: 512,
            max_conv_dim: 512,
            max_dur: 50,
            multispeaker: true,
            n_layer: 3,
            n_mels: 80,
            n_token: 178,
            style_dim: 128,
            text_encoder_kernel_size: 5,
            plbert: config::KokoroPlbertConfig {
                hidden_size: 768,
                num_attention_heads: 12,
                intermediate_size: 2048,
                max_position_embeddings: 512,
                num_hidden_layers: 12,
                dropout: 0.1,
            },
            vocab: HashMap::new(),
        };

        assert_eq!(cfg.context_length(), 512);
    }

    #[test]
    fn kokoro_local_prepare_smoke_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro smoke");
        let model = KokoroTtsModel::load(Path::new(&model_dir), device)
            .expect("load local Kokoro model");
        let prepared = model
            .prepare_request("Hello world.", Some("af_heart"), Some("en-US"), 1.0)
            .expect("prepare Kokoro request");

        assert!(!prepared.phonemes.is_empty());
        assert!(!prepared.token_ids.is_empty());
        assert_eq!(prepared.ref_style.shape().dims(), &[1, 256]);
    }

    #[test]
    fn kokoro_local_bert_prosody_debug_smoke_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro prosody smoke");
        let model = KokoroTtsModel::load(Path::new(&model_dir), device)
            .expect("load local Kokoro model");
        let prepared = model
            .prepare_request("Hello world.", Some("af_heart"), Some("en-US"), 1.0)
            .expect("prepare Kokoro request");
        let debug = model
            .run_bert_prosody_debug(&prepared)
            .expect("run Kokoro BERT/prosody debug");

        assert!(!debug.duration_frames.is_empty());
        assert!(debug.expanded_frames > 0);
        assert_eq!(debug.f0_shape.len(), 2);
        assert_eq!(debug.n_shape.len(), 2);
    }

    #[test]
    fn kokoro_local_predecoder_debug_smoke_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro predecoder smoke");
        let model = KokoroTtsModel::load(Path::new(&model_dir), device)
            .expect("load local Kokoro model");
        let prepared = model
            .prepare_request("Hello world.", Some("af_heart"), Some("en-US"), 1.0)
            .expect("prepare Kokoro request");
        let debug = model
            .run_predecoder_debug(&prepared)
            .expect("run Kokoro predecoder debug");

        assert_eq!(debug.text_encoder_shape.len(), 3);
        assert_eq!(debug.asr_shape.len(), 3);
        assert!(debug.prosody.expanded_frames > 0);
    }

    #[test]
    fn kokoro_local_generate_smoke_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro generate smoke");
        let model = KokoroTtsModel::load(Path::new(&model_dir), device)
            .expect("load local Kokoro model");
        let result = model
            .generate("Hello world.", Some("af_heart"), Some("en-US"), 1.0)
            .expect("run Kokoro generate");

        assert_eq!(result.sample_rate, KokoroConfig::TARGET_SAMPLE_RATE);
        assert!(!result.samples.is_empty());
        assert!(result.samples.iter().all(|v| v.is_finite()));
        assert!(result.samples.len() > 100);
    }
}
