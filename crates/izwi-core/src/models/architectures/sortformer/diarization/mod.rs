mod nemo;

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::ops;
use candle_nn::{
    batch_norm, layer_norm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module,
    ModuleT, VarBuilder,
};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::weights::mlx;
use crate::runtime::{DiarizationConfig, DiarizationResult, DiarizationSegment};

use nemo::{ensure_sortformer_artifacts, SortformerArtifacts};

const TARGET_SAMPLE_RATE: u32 = 16_000;
const MAX_SUPPORTED_SPEAKERS: usize = 4;
const DEFAULT_MIN_SPEECH_MS: f32 = 240.0;
const DEFAULT_MIN_SILENCE_MS: f32 = 200.0;
const PREEMPH: f32 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8;
const NORMALIZE_EPS: f32 = 1e-5;
const REALTIME_VAD_THRESHOLD: f32 = 0.02;
const TS_VAD_FRAME_LENGTH_SECS: f32 = 0.01;
const TS_VAD_UNIT_FRAME_COUNT: usize = 8;

#[derive(Debug, Clone, serde::Deserialize)]
struct SortformerModelConfig {
    sample_rate: Option<u32>,
    max_num_of_spks: Option<usize>,
    preprocessor: Option<SortformerPreprocessorConfig>,
    sortformer_modules: Option<SortformerModulesConfig>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct SortformerPreprocessorConfig {
    sample_rate: Option<u32>,
    window_size: Option<f32>,
    window_stride: Option<f32>,
    features: Option<usize>,
    n_fft: Option<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct SortformerModulesConfig {
    pred_score_threshold: Option<f32>,
}

pub struct SortformerDiarizerModel {
    variant: ModelVariant,
    _artifacts: SortformerArtifacts,
    _checkpoint_tensor_count: usize,
    model: SortformerInferenceModel,
    pred_threshold: f32,
}

impl SortformerDiarizerModel {
    pub fn load(model_dir: &Path, variant: ModelVariant) -> Result<Self> {
        if !variant.is_diarization() {
            return Err(Error::InvalidInput(format!(
                "Variant {} is not a Sortformer diarization model",
                variant.dir_name()
            )));
        }

        let artifacts = ensure_sortformer_artifacts(model_dir, variant)?;
        let tensor_info =
            candle_core::pickle::read_pth_tensor_info(&artifacts.checkpoint_path, false, None)
                .map_err(|e| {
                    Error::ModelLoadError(format!(
                        "Failed to inspect Sortformer checkpoint {}: {}",
                        artifacts.checkpoint_path.display(),
                        e
                    ))
                })?;

        let config: SortformerModelConfig = serde_yaml::from_str(
            &std::fs::read_to_string(&artifacts.model_config_path).map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed reading Sortformer config {}: {}",
                    artifacts.model_config_path.display(),
                    e
                ))
            })?,
        )
        .map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed parsing Sortformer config {}: {}",
                artifacts.model_config_path.display(),
                e
            ))
        })?;

        let sample_rate = config.sample_rate.unwrap_or(TARGET_SAMPLE_RATE);
        if sample_rate != TARGET_SAMPLE_RATE {
            return Err(Error::ModelLoadError(format!(
                "Unsupported Sortformer sample rate {sample_rate}; expected {TARGET_SAMPLE_RATE}"
            )));
        }

        let num_spks = config.max_num_of_spks.unwrap_or(MAX_SUPPORTED_SPEAKERS);
        if num_spks != MAX_SUPPORTED_SPEAKERS {
            return Err(Error::ModelLoadError(format!(
                "Unsupported Sortformer speaker count {num_spks}; expected {MAX_SUPPORTED_SPEAKERS}"
            )));
        }

        let device = Device::Cpu;
        let vb =
            VarBuilder::from_pth(&artifacts.checkpoint_path, DType::F32, &device).map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to load Sortformer checkpoint {}: {}",
                    artifacts.checkpoint_path.display(),
                    e
                ))
            })?;

        let preprocessor_cfg =
            config
                .preprocessor
                .clone()
                .unwrap_or(SortformerPreprocessorConfig {
                    sample_rate: Some(TARGET_SAMPLE_RATE),
                    window_size: Some(0.025),
                    window_stride: Some(0.01),
                    features: Some(128),
                    n_fft: Some(512),
                });

        let model = SortformerInferenceModel::load(&vb, preprocessor_cfg)?;

        let pred_threshold = config
            .sortformer_modules
            .as_ref()
            .and_then(|m| m.pred_score_threshold)
            .unwrap_or(0.25)
            .clamp(0.05, 0.95);

        Ok(Self {
            variant,
            _artifacts: artifacts,
            _checkpoint_tensor_count: tensor_info.len(),
            model,
            pred_threshold,
        })
    }

    pub fn diarize(
        &self,
        audio: &[f32],
        sample_rate: u32,
        config: &DiarizationConfig,
    ) -> Result<DiarizationResult> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }
        if sample_rate == 0 {
            return Err(Error::InvalidInput("Invalid sample rate: 0".to_string()));
        }

        let samples = if sample_rate == TARGET_SAMPLE_RATE {
            audio.to_vec()
        } else {
            resample_linear(audio, sample_rate, TARGET_SAMPLE_RATE)
        };

        let duration_secs = samples.len() as f32 / TARGET_SAMPLE_RATE as f32;
        if samples.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                duration_secs,
                speaker_count: 0,
            });
        }

        let (speaker_probs, frame_stride_samples) =
            self.model.infer_speaker_probabilities(&samples)?;
        if speaker_probs.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                duration_secs,
                speaker_count: 0,
            });
        }

        let min_speech_ms = config
            .min_speech_duration_ms
            .unwrap_or(DEFAULT_MIN_SPEECH_MS)
            .clamp(
                frame_stride_samples as f32 * 1000.0 / TARGET_SAMPLE_RATE as f32,
                5000.0,
            );
        let min_silence_ms = config
            .min_silence_duration_ms
            .unwrap_or(DEFAULT_MIN_SILENCE_MS)
            .clamp(
                frame_stride_samples as f32 * 1000.0 / TARGET_SAMPLE_RATE as f32,
                5000.0,
            );

        let mut gated_probs = speaker_probs;
        if sortformer_rms_gating_enabled() {
            let frame_ms = frame_stride_samples as f32 * 1000.0 / TARGET_SAMPLE_RATE as f32;
            let min_speech_frames = ((min_speech_ms / frame_ms).round() as usize).max(1);
            let min_silence_frames = ((min_silence_ms / frame_ms).round() as usize).max(1);

            let frame_count = gated_probs.len();
            let mut vad_mask = realtime_voice_vad_frame_mask(
                &samples,
                frame_count,
                frame_stride_samples,
                REALTIME_VAD_THRESHOLD,
            );
            smooth_activity_mask(&mut vad_mask, min_speech_frames, min_silence_frames);

            for (frame_idx, active) in vad_mask.iter().copied().enumerate() {
                if !active {
                    for spk in 0..MAX_SUPPORTED_SPEAKERS {
                        gated_probs[frame_idx][spk] = 0.0;
                    }
                }
            }
        }

        let requested_max = config.max_speakers.unwrap_or(MAX_SUPPORTED_SPEAKERS);
        let max_speakers = requested_max.clamp(1, MAX_SUPPORTED_SPEAKERS);
        let requested_min = config.min_speakers.unwrap_or(1);
        let min_speakers = requested_min.clamp(1, max_speakers);

        let postprocessing_params = resolve_postprocessing_params(
            self.pred_threshold,
            min_speech_ms / 1000.0,
            min_silence_ms / 1000.0,
        );

        let mut raw_segments = Vec::<RawSegment>::new();
        let mut speaker_stats = Vec::<SpeakerActivityStats>::new();
        for speaker_idx in 0..MAX_SUPPORTED_SPEAKERS {
            let speaker_segments =
                ts_vad_post_processing(&gated_probs, speaker_idx, &postprocessing_params);
            if speaker_segments.is_empty() {
                speaker_stats.push(SpeakerActivityStats {
                    speaker_idx,
                    total_duration_secs: 0.0,
                    peak_probability: 0.0,
                    segment_count: 0,
                });
                continue;
            }

            let peak_probability = gated_probs
                .iter()
                .map(|row| row[speaker_idx])
                .fold(0.0f32, f32::max);
            let total_duration_secs = speaker_segments
                .iter()
                .map(|(start_secs, end_secs)| (end_secs - start_secs).max(0.0))
                .sum::<f32>();

            for (start_secs, end_secs) in speaker_segments {
                if end_secs <= start_secs {
                    continue;
                }
                let confidence = average_speaker_probability_for_range(
                    &gated_probs,
                    speaker_idx,
                    start_secs,
                    end_secs,
                    frame_stride_samples,
                );
                raw_segments.push(RawSegment {
                    speaker_idx,
                    start_secs,
                    end_secs,
                    confidence,
                });
            }

            speaker_stats.push(SpeakerActivityStats {
                speaker_idx,
                total_duration_secs,
                peak_probability,
                segment_count: raw_segments
                    .iter()
                    .filter(|segment| segment.speaker_idx == speaker_idx)
                    .count(),
            });
        }

        if raw_segments.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                duration_secs,
                speaker_count: 0,
            });
        }

        let selected_speakers = select_speaker_channels(&speaker_stats, min_speakers, max_speakers);
        raw_segments.retain(|segment| selected_speakers.contains(&segment.speaker_idx));

        raw_segments.sort_by(|a, b| {
            a.start_secs
                .total_cmp(&b.start_secs)
                .then(a.speaker_idx.cmp(&b.speaker_idx))
        });

        let mut speaker_first_start = BTreeMap::<usize, f32>::new();
        for segment in &raw_segments {
            speaker_first_start
                .entry(segment.speaker_idx)
                .and_modify(|cur| {
                    if segment.start_secs < *cur {
                        *cur = segment.start_secs;
                    }
                })
                .or_insert(segment.start_secs);
        }

        let mut ordered = speaker_first_start.into_iter().collect::<Vec<_>>();
        ordered.sort_by(|a, b| a.1.total_cmp(&b.1));
        let speaker_remap = ordered
            .iter()
            .enumerate()
            .map(|(i, (speaker_idx, _))| (*speaker_idx, i))
            .collect::<HashMap<_, _>>();

        let speaker_labels = (0..ordered.len())
            .map(|idx| format!("SPEAKER_{idx:02}"))
            .collect::<Vec<_>>();

        let mut segments = raw_segments
            .into_iter()
            .map(|segment| {
                let remapped = speaker_remap
                    .get(&segment.speaker_idx)
                    .copied()
                    .unwrap_or(0);
                DiarizationSegment {
                    speaker: speaker_labels
                        .get(remapped)
                        .cloned()
                        .unwrap_or_else(|| format!("SPEAKER_{remapped:02}")),
                    start_secs: segment.start_secs,
                    end_secs: segment.end_secs,
                    confidence: segment.confidence,
                }
            })
            .collect::<Vec<_>>();

        merge_adjacent_segments(&mut segments, 0.0);
        segments.sort_by(|a, b| {
            a.start_secs
                .total_cmp(&b.start_secs)
                .then(a.speaker.cmp(&b.speaker))
        });

        let speaker_count = segments
            .iter()
            .map(|segment| segment.speaker.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len();

        Ok(DiarizationResult {
            segments,
            duration_secs,
            speaker_count,
        })
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }
}

#[derive(Debug, Clone)]
struct RawSegment {
    speaker_idx: usize,
    start_secs: f32,
    end_secs: f32,
    confidence: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
struct SpeakerActivityStats {
    speaker_idx: usize,
    total_duration_secs: f32,
    peak_probability: f32,
    segment_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct PostProcessingParams {
    onset: f32,
    offset: f32,
    pad_onset: f32,
    pad_offset: f32,
    min_duration_on: f32,
    min_duration_off: f32,
    filter_speech_first: bool,
}

struct SortformerInferenceModel {
    preprocessor: SortformerPreprocessor,
    encoder: SortformerConformerEncoder,
    encoder_proj: Linear,
    transformer: SortformerTransformerEncoder,
    head: SortformerSpeakerHead,
}

impl SortformerInferenceModel {
    fn load(vb: &VarBuilder, preprocessor_cfg: SortformerPreprocessorConfig) -> Result<Self> {
        let preprocessor = SortformerPreprocessor::load(vb, preprocessor_cfg)?;
        let encoder = SortformerConformerEncoder::load(vb.pp("encoder"))?;

        let encoder_proj_w = vb
            .pp("sortformer_modules.encoder_proj")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (proj_out, proj_in) = encoder_proj_w.dims2()?;
        let encoder_proj =
            mlx::load_linear(proj_in, proj_out, vb.pp("sortformer_modules.encoder_proj"))?;

        let transformer = SortformerTransformerEncoder::load(vb.pp("transformer_encoder"))?;
        let head = SortformerSpeakerHead::load(vb.pp("sortformer_modules"))?;

        Ok(Self {
            preprocessor,
            encoder,
            encoder_proj,
            transformer,
            head,
        })
    }

    fn infer_speaker_probabilities(
        &self,
        samples: &[f32],
    ) -> Result<(Vec<[f32; MAX_SUPPORTED_SPEAKERS]>, usize)> {
        let mut normalized = samples.to_vec();
        let max_abs = normalized
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1e-6);
        for sample in &mut normalized {
            *sample /= max_abs;
        }

        let (features, feature_frames) = self.preprocessor.compute_features(&normalized)?;
        let (encoded, encoded_len) = self.encoder.forward(&features, feature_frames)?;
        if encoded_len == 0 {
            return Ok((Vec::new(), self.encoder.frame_stride_samples()));
        }

        let mut x = encoded.i((.., ..encoded_len, ..))?;
        x = x.apply(&self.encoder_proj)?;
        x = self.transformer.forward(&x)?;
        let probs = self.head.forward(&x)?;

        let probs = probs.squeeze(0)?;
        let dims = probs.dims2()?;
        if dims.1 != MAX_SUPPORTED_SPEAKERS {
            return Err(Error::InferenceError(format!(
                "Unexpected Sortformer speaker dimension {}; expected {}",
                dims.1, MAX_SUPPORTED_SPEAKERS
            )));
        }

        let values = probs.flatten_all()?.to_vec1::<f32>()?;
        let mut out = vec![[0.0f32; MAX_SUPPORTED_SPEAKERS]; dims.0];
        for frame in 0..dims.0 {
            for spk in 0..MAX_SUPPORTED_SPEAKERS {
                out[frame][spk] = values[frame * MAX_SUPPORTED_SPEAKERS + spk];
            }
        }

        Ok((out, self.encoder.frame_stride_samples()))
    }
}

struct SortformerPreprocessor {
    sample_rate: usize,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    _window: Vec<f32>,
    padded_window: Vec<f32>,
    fb: Vec<f32>,
    n_mels: usize,
    n_freqs: usize,
}

impl SortformerPreprocessor {
    fn load(vb: &VarBuilder, cfg: SortformerPreprocessorConfig) -> Result<Self> {
        let sample_rate = cfg.sample_rate.unwrap_or(TARGET_SAMPLE_RATE) as usize;
        let n_fft = cfg.n_fft.unwrap_or(512);
        let win_length =
            ((cfg.window_size.unwrap_or(0.025) * sample_rate as f32).round() as usize).max(1);
        let hop_length =
            ((cfg.window_stride.unwrap_or(0.01) * sample_rate as f32).round() as usize).max(1);
        let n_mels = cfg.features.unwrap_or(128);

        let preproc_vb = vb.pp("preprocessor.featurizer");
        let window = match preproc_vb.get_unchecked_dtype("window", DType::F32) {
            Ok(window_tensor) => window_tensor.to_vec1::<f32>()?,
            Err(_) => hann_window(win_length),
        };

        let (fb, loaded_mels, loaded_freqs) = match preproc_vb.get_unchecked_dtype("fb", DType::F32)
        {
            Ok(fb_tensor) => {
                let (_, mels, freqs) = fb_tensor.dims3()?;
                let fb = fb_tensor.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
                (fb, mels, freqs)
            }
            Err(_) => {
                let generated =
                    mel_filterbank(sample_rate, n_fft, n_mels, 0.0, sample_rate as f32 / 2.0);
                (generated, n_mels, n_fft / 2 + 1)
            }
        };

        let n_freqs = n_fft / 2 + 1;
        if loaded_freqs != n_freqs {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Sortformer filterbank bins: expected {}, got {}",
                n_freqs, loaded_freqs
            )));
        }

        let mut padded_window = vec![0.0f32; n_fft];
        let src_len = window.len().min(n_fft);
        let offset = (n_fft - src_len) / 2;
        padded_window[offset..offset + src_len].copy_from_slice(&window[..src_len]);

        Ok(Self {
            sample_rate,
            n_fft,
            win_length,
            hop_length,
            _window: window,
            padded_window,
            fb,
            n_mels: loaded_mels,
            n_freqs,
        })
    }

    fn compute_features(&self, audio: &[f32]) -> Result<(Tensor, usize)> {
        if audio.is_empty() {
            return Ok((
                Tensor::zeros((1, self.n_mels, 1), DType::F32, &Device::Cpu)?,
                0,
            ));
        }

        let mut x = audio.to_vec();
        preemphasis(&mut x, PREEMPH);

        let center_pad = self.n_fft / 2;
        let mut padded = Vec::with_capacity(x.len() + center_pad * 2);
        padded.extend(std::iter::repeat(0.0).take(center_pad));
        padded.extend_from_slice(&x);
        padded.extend(std::iter::repeat(0.0).take(center_pad));

        let frame_count = if padded.len() >= self.n_fft {
            (padded.len() - self.n_fft) / self.hop_length + 1
        } else {
            1
        };

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.n_fft);

        let mut spectrum = vec![0f32; frame_count * self.n_freqs];
        let mut buffer = vec![Complex::<f32>::new(0.0, 0.0); self.n_fft];
        for frame_idx in 0..frame_count {
            let start = frame_idx * self.hop_length;
            let slice = &padded[start..start + self.n_fft];
            for i in 0..self.n_fft {
                buffer[i].re = slice[i] * self.padded_window[i];
                buffer[i].im = 0.0;
            }
            fft.process(&mut buffer);
            for k in 0..self.n_freqs {
                let mag = (buffer[k].re * buffer[k].re + buffer[k].im * buffer[k].im).sqrt();
                spectrum[frame_idx * self.n_freqs + k] = mag * mag;
            }
        }

        let mut mel = vec![0f32; self.n_mels * frame_count];
        for m in 0..self.n_mels {
            for t in 0..frame_count {
                let mut acc = 0f32;
                let spec_row = &spectrum[t * self.n_freqs..(t + 1) * self.n_freqs];
                let fb_row = &self.fb[m * self.n_freqs..(m + 1) * self.n_freqs];
                for f in 0..self.n_freqs {
                    acc += spec_row[f] * fb_row[f];
                }
                mel[m * frame_count + t] = (acc + LOG_GUARD).ln();
            }
        }

        let valid_frames = audio.len() / self.hop_length;
        normalize_per_feature(
            &mut mel,
            self.n_mels,
            frame_count,
            valid_frames.min(frame_count),
        );

        if valid_frames < frame_count {
            for m in 0..self.n_mels {
                for t in valid_frames..frame_count {
                    mel[m * frame_count + t] = 0.0;
                }
            }
        }

        let features = Tensor::from_vec(mel, (1, self.n_mels, frame_count), &Device::Cpu)?;
        Ok((features, valid_frames.min(frame_count)))
    }
}

struct SortformerConformerEncoder {
    pre_encode: ConvSubsamplingDw,
    layers: Vec<ConformerLayer>,
    d_model: usize,
    frame_stride_samples: usize,
}

impl SortformerConformerEncoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let pre_encode = ConvSubsamplingDw::load(vb.pp("pre_encode"))?;

        let mut layers = Vec::new();
        let mut idx = 0usize;
        loop {
            let layer_vb = vb.pp(format!("layers.{idx}"));
            if !layer_vb.contains_tensor("norm_out.weight") {
                break;
            }
            layers.push(ConformerLayer::load(layer_vb)?);
            idx += 1;
        }
        if layers.is_empty() {
            return Err(Error::ModelLoadError(
                "Sortformer Conformer encoder has no layers".to_string(),
            ));
        }

        let d_model = layers[0].d_model();
        Ok(Self {
            pre_encode,
            layers,
            d_model,
            frame_stride_samples: 160 * 8,
        })
    }

    fn forward(&self, features: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        let (mut x, encoded_len) = self.pre_encode.forward(features, feature_frames)?;
        let pos_len = x.dim(1)?;
        let pos_emb = build_rel_positional_embedding(pos_len, self.d_model, x.device())?;
        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb)?;
        }
        Ok((x, encoded_len))
    }

    fn frame_stride_samples(&self) -> usize {
        self.frame_stride_samples
    }
}

struct ConvSubsamplingDw {
    conv0: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv5: Conv2d,
    conv6: Conv2d,
    out: Linear,
}

impl ConvSubsamplingDw {
    fn load(vb: VarBuilder) -> Result<Self> {
        let conv0_w = vb.pp("conv.0").get_unchecked_dtype("weight", DType::F32)?;
        let (out_channels, _, _, _) = conv0_w.dims4()?;

        let stride_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let point_cfg = Conv2dConfig {
            stride: 1,
            padding: 0,
            ..Default::default()
        };

        let conv0 = mlx::load_conv2d(1, out_channels, 3, stride_cfg, vb.pp("conv.0"))?;

        let mut dw_stride_cfg = stride_cfg;
        dw_stride_cfg.groups = out_channels;
        let conv2 = mlx::load_conv2d(1, out_channels, 3, dw_stride_cfg, vb.pp("conv.2"))?;
        let conv3 = mlx::load_conv2d(out_channels, out_channels, 1, point_cfg, vb.pp("conv.3"))?;
        let conv5 = mlx::load_conv2d(1, out_channels, 3, dw_stride_cfg, vb.pp("conv.5"))?;
        let conv6 = mlx::load_conv2d(out_channels, out_channels, 1, point_cfg, vb.pp("conv.6"))?;

        let out_w = vb.pp("out").get_unchecked_dtype("weight", DType::F32)?;
        let (out_dim, in_dim) = out_w.dims2()?;
        let out = mlx::load_linear(in_dim, out_dim, vb.pp("out"))?;

        Ok(Self {
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            out,
        })
    }

    fn forward(&self, features: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        let mut x = features.transpose(1, 2)?.unsqueeze(1)?; // [B,1,T,F]

        x = self.conv0.forward(&x)?;
        x = x.relu()?;

        x = self.conv2.forward(&x)?;
        x = self.conv3.forward(&x)?;
        x = x.relu()?;

        x = self.conv5.forward(&x)?;
        x = self.conv6.forward(&x)?;
        x = x.relu()?;

        let (b, c, t, f) = x.dims4()?;
        let x = x
            .transpose(1, 2)?
            .reshape((b, t, c * f))?
            .apply(&self.out)?;
        let encoded_len = subsampled_len_3x(feature_frames).min(t);
        Ok((x, encoded_len))
    }
}

fn subsampled_len_3x(mut len: usize) -> usize {
    for _ in 0..3 {
        len = len.div_ceil(2);
    }
    len
}

struct ConformerLayer {
    norm_ff1: LayerNorm,
    ff1: FeedForward,
    norm_self_att: LayerNorm,
    self_attn: RelPosSelfAttention,
    norm_conv: LayerNorm,
    conv: ConformerConv,
    norm_ff2: LayerNorm,
    ff2: FeedForward,
    norm_out: LayerNorm,
    d_model: usize,
}

impl ConformerLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let d_model = vb
            .pp("norm_out")
            .get_unchecked_dtype("weight", DType::F32)?
            .dim(0)?;

        let ff_dim = vb
            .pp("feed_forward1.linear1")
            .get_unchecked_dtype("weight", DType::F32)?
            .dims2()?
            .0;

        let norm_ff1 = layer_norm(d_model, 1e-5, vb.pp("norm_feed_forward1"))?;
        let ff1 = FeedForward::load(vb.pp("feed_forward1"), d_model, ff_dim)?;

        let norm_self_att = layer_norm(d_model, 1e-5, vb.pp("norm_self_att"))?;
        let self_attn = RelPosSelfAttention::load(vb.pp("self_attn"), d_model)?;

        let norm_conv = layer_norm(d_model, 1e-5, vb.pp("norm_conv"))?;
        let conv = ConformerConv::load(vb.pp("conv"), d_model)?;

        let norm_ff2 = layer_norm(d_model, 1e-5, vb.pp("norm_feed_forward2"))?;
        let ff2 = FeedForward::load(vb.pp("feed_forward2"), d_model, ff_dim)?;

        let norm_out = layer_norm(d_model, 1e-5, vb.pp("norm_out"))?;

        Ok(Self {
            norm_ff1,
            ff1,
            norm_self_att,
            self_attn,
            norm_conv,
            conv,
            norm_ff2,
            ff2,
            norm_out,
            d_model,
        })
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let mut residual = x.clone();

        let ff1 = self.ff1.forward(&self.norm_ff1.forward(&residual)?)?;
        residual = residual.broadcast_add(&ff1.affine(0.5, 0.0)?)?;

        let attn = self
            .self_attn
            .forward(&self.norm_self_att.forward(&residual)?, pos_emb)?;
        residual = residual.broadcast_add(&attn)?;

        let conv = self.conv.forward(&self.norm_conv.forward(&residual)?)?;
        residual = residual.broadcast_add(&conv)?;

        let ff2 = self.ff2.forward(&self.norm_ff2.forward(&residual)?)?;
        residual = residual.broadcast_add(&ff2.affine(0.5, 0.0)?)?;

        self.norm_out
            .forward(&residual)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    fn load(vb: VarBuilder, d_model: usize, ff_dim: usize) -> Result<Self> {
        let linear1 = mlx::load_linear(d_model, ff_dim, vb.pp("linear1"))?;
        let linear2 = mlx::load_linear(ff_dim, d_model, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = swish(&x)?;
        self.linear2
            .forward(&x)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

struct ConformerConv {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    batch_norm: candle_nn::BatchNorm,
    pointwise_conv2: Conv1d,
    d_model: usize,
}

impl ConformerConv {
    fn load(vb: VarBuilder, d_model: usize) -> Result<Self> {
        let kernel_size = vb
            .pp("depthwise_conv")
            .get_unchecked_dtype("weight", DType::F32)?
            .dims3()?
            .2;

        let pointwise_conv1 = mlx::load_conv1d_no_bias(
            d_model,
            d_model * 2,
            1,
            Conv1dConfig::default(),
            vb.pp("pointwise_conv1"),
        )?;

        let depthwise_conv = mlx::load_conv1d_no_bias(
            d_model,
            d_model,
            kernel_size,
            Conv1dConfig {
                padding: (kernel_size - 1) / 2,
                groups: d_model,
                ..Default::default()
            },
            vb.pp("depthwise_conv"),
        )?;

        let batch_norm = batch_norm(d_model, 1e-5, vb.pp("batch_norm"))?;

        let pointwise_conv2 = mlx::load_conv1d_no_bias(
            d_model,
            d_model,
            1,
            Conv1dConfig::default(),
            vb.pp("pointwise_conv2"),
        )?;

        Ok(Self {
            pointwise_conv1,
            depthwise_conv,
            batch_norm,
            pointwise_conv2,
            d_model,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.transpose(1, 2)?;

        x = self.pointwise_conv1.forward(&x)?;
        let x_a = x.i((.., ..self.d_model, ..))?;
        let x_b = x.i((.., self.d_model.., ..))?;
        x = x_a.broadcast_mul(&ops::sigmoid(&x_b)?)?;

        x = self.depthwise_conv.forward(&x)?;
        x = self.batch_norm.forward_t(&x, false)?;
        x = swish(&x)?;
        x = self.pointwise_conv2.forward(&x)?;

        x.transpose(1, 2).map_err(Error::from)
    }
}

struct RelPosSelfAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
    num_heads: usize,
    head_dim: usize,
    d_model: usize,
}

impl RelPosSelfAttention {
    fn load(vb: VarBuilder, d_model: usize) -> Result<Self> {
        let pos_bias_u = vb.get_unchecked_dtype("pos_bias_u", DType::F32)?;
        let (num_heads, head_dim) = pos_bias_u.dims2()?;
        let pos_bias_v = vb.get((num_heads, head_dim), "pos_bias_v")?;

        if num_heads * head_dim != d_model {
            return Err(Error::ModelLoadError(format!(
                "Sortformer attention head dims mismatch: heads={num_heads}, head_dim={head_dim}, d_model={d_model}"
            )));
        }

        let linear_q = mlx::load_linear(d_model, d_model, vb.pp("linear_q"))?;
        let linear_k = mlx::load_linear(d_model, d_model, vb.pp("linear_k"))?;
        let linear_v = mlx::load_linear(d_model, d_model, vb.pp("linear_v"))?;
        let linear_out = mlx::load_linear(d_model, d_model, vb.pp("linear_out"))?;
        let linear_pos = mlx::load_linear_no_bias(d_model, d_model, vb.pp("linear_pos"))?;

        Ok(Self {
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
            num_heads,
            head_dim,
            d_model,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        let q = self
            .linear_q
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .linear_k
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .linear_v
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((1, 2 * t - 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let pos_bias_u = self
            .pos_bias_u
            .reshape((1, self.num_heads, 1, self.head_dim))?;
        let pos_bias_v = self
            .pos_bias_v
            .reshape((1, self.num_heads, 1, self.head_dim))?;

        let q_u = q.broadcast_add(&pos_bias_u)?.contiguous()?;
        let q_v = q.broadcast_add(&pos_bias_v)?.contiguous()?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let p_t = p.transpose(2, 3)?.contiguous()?;
        let matrix_ac = q_u.matmul(&k_t)?;
        let matrix_bd = rel_shift(&q_v.matmul(&p_t)?)?;
        let matrix_bd = matrix_bd.narrow(3, 0, t)?;

        let scores = matrix_ac
            .broadcast_add(&matrix_bd)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let attn = ops::softmax(&scores, 3)?;

        let out = attn.contiguous()?.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, t, self.d_model))?;

        self.linear_out
            .forward(&out)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

fn rel_shift(x: &Tensor) -> Result<Tensor> {
    let (b, h, qlen, pos_len) = x.dims4()?;
    let x = x.pad_with_zeros(3, 1, 0)?;
    let x = x.reshape((b, h, pos_len + 1, qlen))?;
    let x = x.narrow(2, 1, pos_len)?;
    x.reshape((b, h, qlen, pos_len)).map_err(Error::from)
}

struct SortformerTransformerEncoder {
    layers: Vec<SortformerTransformerLayer>,
}

impl SortformerTransformerEncoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        let mut idx = 0usize;
        loop {
            let layer_vb = vb.pp(format!("layers.{idx}"));
            if !layer_vb.contains_tensor("layer_norm_1.weight") {
                break;
            }
            layers.push(SortformerTransformerLayer::load(layer_vb)?);
            idx += 1;
        }
        if layers.is_empty() {
            return Err(Error::ModelLoadError(
                "Sortformer transformer encoder has no layers".to_string(),
            ));
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }
}

struct SortformerTransformerLayer {
    norm1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out_proj: Linear,
    norm2: LayerNorm,
    dense_in: Linear,
    dense_out: Linear,
    d_model: usize,
    num_heads: usize,
    head_dim: usize,
}

impl SortformerTransformerLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let d_model = vb
            .pp("layer_norm_1")
            .get_unchecked_dtype("weight", DType::F32)?
            .dim(0)?;

        let q_w = vb
            .pp("first_sub_layer.query_net")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (_, q_in) = q_w.dims2()?;
        if q_in != d_model {
            return Err(Error::ModelLoadError(format!(
                "Sortformer transformer query input dim mismatch: expected {d_model}, got {q_in}"
            )));
        }

        let dense_in_w = vb
            .pp("second_sub_layer.dense_in")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (inner_size, dense_in) = dense_in_w.dims2()?;
        if dense_in != d_model {
            return Err(Error::ModelLoadError(format!(
                "Sortformer transformer FFN input dim mismatch: expected {d_model}, got {dense_in}"
            )));
        }

        let num_heads = 8usize;
        if d_model % num_heads != 0 {
            return Err(Error::ModelLoadError(format!(
                "Sortformer transformer hidden size {d_model} is not divisible by {num_heads} heads"
            )));
        }
        let head_dim = d_model / num_heads;

        Ok(Self {
            norm1: layer_norm(d_model, 1e-5, vb.pp("layer_norm_1"))?,
            q: mlx::load_linear(d_model, d_model, vb.pp("first_sub_layer.query_net"))?,
            k: mlx::load_linear(d_model, d_model, vb.pp("first_sub_layer.key_net"))?,
            v: mlx::load_linear(d_model, d_model, vb.pp("first_sub_layer.value_net"))?,
            out_proj: mlx::load_linear(d_model, d_model, vb.pp("first_sub_layer.out_projection"))?,
            norm2: layer_norm(d_model, 1e-5, vb.pp("layer_norm_2"))?,
            dense_in: mlx::load_linear(d_model, inner_size, vb.pp("second_sub_layer.dense_in"))?,
            dense_out: mlx::load_linear(inner_size, d_model, vb.pp("second_sub_layer.dense_out"))?,
            d_model,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let h = self.norm1.forward(x)?;
        let attn = self.self_attention(&h)?;
        let h = residual.broadcast_add(&attn)?;

        let residual2 = h.clone();
        let h2 = self.norm2.forward(&h)?;
        let ff = self
            .dense_out
            .forward(&self.dense_in.forward(&h2)?.relu()?)?;
        residual2.broadcast_add(&ff).map_err(Error::from)
    }

    fn self_attention(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        let q = self
            .q
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q
            .matmul(&k_t)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let attn = ops::softmax(&scores, 3)?;
        let ctx = attn.contiguous()?.matmul(&v)?;

        let ctx = ctx.transpose(1, 2)?.reshape((b, t, self.d_model))?;
        self.out_proj.forward(&ctx).map_err(Error::from)
    }
}

struct SortformerSpeakerHead {
    first_hidden_to_hidden: Linear,
    single_hidden_to_spks: Linear,
}

impl SortformerSpeakerHead {
    fn load(vb: VarBuilder) -> Result<Self> {
        let first_w = vb
            .pp("first_hidden_to_hidden")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (first_out, first_in) = first_w.dims2()?;
        if first_out != first_in {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Sortformer hidden projection shape: [{first_out}, {first_in}]"
            )));
        }

        let second_w = vb
            .pp("single_hidden_to_spks")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (spk_out, spk_in) = second_w.dims2()?;
        if spk_out != MAX_SUPPORTED_SPEAKERS {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Sortformer speaker head output dim {spk_out}; expected {MAX_SUPPORTED_SPEAKERS}"
            )));
        }
        if spk_in != first_out {
            return Err(Error::ModelLoadError(format!(
                "Sortformer speaker head dim mismatch: hidden={first_out}, input={spk_in}"
            )));
        }

        let first_hidden_to_hidden =
            mlx::load_linear(first_in, first_out, vb.pp("first_hidden_to_hidden"))?;
        let single_hidden_to_spks =
            mlx::load_linear(spk_in, spk_out, vb.pp("single_hidden_to_spks"))?;
        Ok(Self {
            first_hidden_to_hidden,
            single_hidden_to_spks,
        })
    }

    fn forward(&self, hidden_out: &Tensor) -> Result<Tensor> {
        let hidden_out = hidden_out.relu()?;
        let hidden_out = self.first_hidden_to_hidden.forward(&hidden_out)?;
        let hidden_out = hidden_out.relu()?;
        let spk_logits = self.single_hidden_to_spks.forward(&hidden_out)?;
        ops::sigmoid(&spk_logits).map_err(Error::from)
    }
}

fn resolve_postprocessing_params(
    pred_threshold: f32,
    min_duration_on: f32,
    min_duration_off: f32,
) -> PostProcessingParams {
    let preset = std::env::var("IZWI_SORTFORMER_PP_PRESET")
        .unwrap_or_else(|_| "model".to_string())
        .to_ascii_lowercase();

    let mut params = match preset.as_str() {
        "callhome" | "callhome_v2" => PostProcessingParams {
            onset: 0.641,
            offset: 0.561,
            pad_onset: 0.229,
            pad_offset: 0.079,
            min_duration_on: 0.511,
            min_duration_off: 0.296,
            filter_speech_first: true,
        },
        "dihard3" | "dihard3_v2" => PostProcessingParams {
            onset: 0.56,
            offset: 1.0,
            pad_onset: 0.063,
            pad_offset: 0.002,
            min_duration_on: 0.007,
            min_duration_off: 0.151,
            filter_speech_first: true,
        },
        _ => PostProcessingParams {
            onset: pred_threshold.clamp(0.0, 1.0),
            offset: pred_threshold.clamp(0.0, 1.0),
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: min_duration_on.max(0.0),
            min_duration_off: min_duration_off.max(0.0),
            filter_speech_first: true,
        },
    };

    if let Some(value) = env_postprocessing_value("IZWI_SORTFORMER_PP_ONSET") {
        params.onset = value.clamp(0.0, 1.0);
    }
    if let Some(value) = env_postprocessing_value("IZWI_SORTFORMER_PP_OFFSET") {
        params.offset = value.clamp(0.0, 1.0);
    }
    if let Some(value) = env_postprocessing_value("IZWI_SORTFORMER_PP_PAD_ONSET") {
        params.pad_onset = value.max(0.0);
    }
    if let Some(value) = env_postprocessing_value("IZWI_SORTFORMER_PP_PAD_OFFSET") {
        params.pad_offset = value.max(0.0);
    }
    if let Some(value) = env_postprocessing_value("IZWI_SORTFORMER_PP_MIN_DURATION_ON") {
        params.min_duration_on = value.max(0.0);
    }
    if let Some(value) = env_postprocessing_value("IZWI_SORTFORMER_PP_MIN_DURATION_OFF") {
        params.min_duration_off = value.max(0.0);
    }
    if let Some(value) = env_flag("IZWI_SORTFORMER_PP_FILTER_SPEECH_FIRST") {
        params.filter_speech_first = value;
    }

    params
}

fn env_postprocessing_value(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite())
}

fn env_flag(key: &str) -> Option<bool> {
    std::env::var(key)
        .ok()
        .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

fn sortformer_rms_gating_enabled() -> bool {
    env_flag("IZWI_SORTFORMER_ENABLE_RMS_GATING").unwrap_or(false)
}

fn select_speaker_channels(
    stats: &[SpeakerActivityStats],
    min_speakers: usize,
    max_speakers: usize,
) -> Vec<usize> {
    let keep = max_speakers.clamp(min_speakers, MAX_SUPPORTED_SPEAKERS);
    let mut ranked = stats.to_vec();
    ranked.sort_by(|a, b| {
        b.total_duration_secs
            .total_cmp(&a.total_duration_secs)
            .then(b.segment_count.cmp(&a.segment_count))
            .then(b.peak_probability.total_cmp(&a.peak_probability))
            .then(a.speaker_idx.cmp(&b.speaker_idx))
    });

    let active = ranked
        .iter()
        .filter(|stat| stat.segment_count > 0 && stat.total_duration_secs > 0.0)
        .map(|stat| stat.speaker_idx)
        .take(keep)
        .collect::<Vec<_>>();

    if active.len() >= min_speakers {
        return active;
    }

    ranked
        .into_iter()
        .take(keep)
        .map(|stat| stat.speaker_idx)
        .collect()
}

fn ts_vad_post_processing(
    probs: &[[f32; MAX_SUPPORTED_SPEAKERS]],
    speaker_idx: usize,
    params: &PostProcessingParams,
) -> Vec<(f32, f32)> {
    let mut repeated = Vec::with_capacity(probs.len() * TS_VAD_UNIT_FRAME_COUNT);
    for row in probs {
        let value = row[speaker_idx].clamp(0.0, 1.0);
        for _ in 0..TS_VAD_UNIT_FRAME_COUNT {
            repeated.push(value);
        }
    }

    filtering(&binarization(&repeated, params), params)
}

fn binarization(sequence: &[f32], params: &PostProcessingParams) -> Vec<(f32, f32)> {
    let mut speech = false;
    let mut start = 0.0f32;
    let mut segments = Vec::new();
    let mut last_index = 0usize;

    for (idx, &value) in sequence.iter().enumerate() {
        last_index = idx;
        if speech {
            if value < params.offset {
                let seg_start = (start - params.pad_onset).max(0.0);
                let seg_end = idx as f32 * TS_VAD_FRAME_LENGTH_SECS + params.pad_offset;
                if seg_end > seg_start {
                    segments.push((seg_start, seg_end));
                }
                start = idx as f32 * TS_VAD_FRAME_LENGTH_SECS;
                speech = false;
            }
        } else if value > params.onset {
            start = idx as f32 * TS_VAD_FRAME_LENGTH_SECS;
            speech = true;
        }
    }

    if speech {
        let seg_start = (start - params.pad_onset).max(0.0);
        let seg_end = last_index as f32 * TS_VAD_FRAME_LENGTH_SECS + params.pad_offset;
        if seg_end > seg_start {
            segments.push((seg_start, seg_end));
        }
    }

    merge_overlap_ranges(&segments)
}

fn filtering(segments: &[(f32, f32)], params: &PostProcessingParams) -> Vec<(f32, f32)> {
    if segments.is_empty() {
        return Vec::new();
    }

    let mut speech_segments = segments.to_vec();
    if params.filter_speech_first {
        if params.min_duration_on > 0.0 {
            speech_segments = filter_short_segments(&speech_segments, params.min_duration_on);
        }
        if params.min_duration_off > 0.0 && speech_segments.len() > 1 {
            let non_speech_segments = get_gap_segments(&speech_segments);
            let short_non_speech_segments = remove_ranges(
                &non_speech_segments,
                &filter_short_segments(&non_speech_segments, params.min_duration_off),
            );
            if !short_non_speech_segments.is_empty() {
                speech_segments.extend(short_non_speech_segments);
                speech_segments = merge_overlap_ranges(&speech_segments);
            }
        }
    } else {
        if params.min_duration_off > 0.0 && speech_segments.len() > 1 {
            let non_speech_segments = get_gap_segments(&speech_segments);
            let short_non_speech_segments = remove_ranges(
                &non_speech_segments,
                &filter_short_segments(&non_speech_segments, params.min_duration_off),
            );
            if !short_non_speech_segments.is_empty() {
                speech_segments.extend(short_non_speech_segments);
                speech_segments = merge_overlap_ranges(&speech_segments);
            }
        }
        if params.min_duration_on > 0.0 {
            speech_segments = filter_short_segments(&speech_segments, params.min_duration_on);
        }
    }
    speech_segments
}

fn remove_ranges(
    original_segments: &[(f32, f32)],
    to_be_removed_segments: &[(f32, f32)],
) -> Vec<(f32, f32)> {
    if original_segments.is_empty() || to_be_removed_segments.is_empty() {
        return original_segments.to_vec();
    }

    original_segments
        .iter()
        .copied()
        .filter(|segment| {
            !to_be_removed_segments.iter().any(|removed| {
                (segment.0 - removed.0).abs() <= f32::EPSILON
                    && (segment.1 - removed.1).abs() <= f32::EPSILON
            })
        })
        .collect()
}

fn filter_short_segments(segments: &[(f32, f32)], threshold: f32) -> Vec<(f32, f32)> {
    segments
        .iter()
        .copied()
        .filter(|(start, end)| (end - start) >= threshold)
        .collect()
}

fn get_gap_segments(segments: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if segments.len() <= 1 {
        return Vec::new();
    }

    let sorted = sort_ranges(segments);
    sorted
        .windows(2)
        .filter_map(|window| {
            let (_, left_end) = window[0];
            let (right_start, _) = window[1];
            (right_start > left_end).then_some((left_end, right_start))
        })
        .collect()
}

fn merge_overlap_ranges(segments: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if segments.len() <= 1 {
        return segments.to_vec();
    }

    let mut sorted = sort_ranges(segments);
    let mut merged = Vec::with_capacity(sorted.len());
    let mut current = sorted.remove(0);
    for segment in sorted {
        if current.1 >= segment.0 {
            current.1 = current.1.max(segment.1);
        } else {
            merged.push(current);
            current = segment;
        }
    }
    merged.push(current);
    merged
}

fn sort_ranges(segments: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut sorted = segments.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    sorted
}

fn average_speaker_probability_for_range(
    probs: &[[f32; MAX_SUPPORTED_SPEAKERS]],
    speaker_idx: usize,
    start_secs: f32,
    end_secs: f32,
    frame_stride_samples: usize,
) -> Option<f32> {
    if probs.is_empty() || end_secs <= start_secs || frame_stride_samples == 0 {
        return None;
    }

    let frame_stride_secs = frame_stride_samples as f32 / TARGET_SAMPLE_RATE as f32;
    let start_frame = (start_secs / frame_stride_secs).floor().max(0.0) as usize;
    let end_frame = ((end_secs / frame_stride_secs).ceil().max(0.0) as usize).min(probs.len());
    if start_frame >= end_frame {
        return None;
    }

    let mut sum = 0.0f32;
    let mut count = 0usize;
    for row in probs.iter().take(end_frame).skip(start_frame) {
        sum += row[speaker_idx];
        count += 1;
    }

    (count > 0).then_some((sum / count as f32).clamp(0.0, 1.0))
}

fn realtime_voice_vad_frame_mask(
    samples: &[f32],
    frame_count: usize,
    frame_stride_samples: usize,
    vad_threshold: f32,
) -> Vec<bool> {
    if frame_count == 0 || frame_stride_samples == 0 {
        return Vec::new();
    }

    let threshold = vad_threshold.clamp(0.001, 1.0);
    let mut mask = vec![false; frame_count];
    for (frame_idx, active) in mask.iter_mut().enumerate().take(frame_count) {
        let start = frame_idx * frame_stride_samples;
        if start >= samples.len() {
            break;
        }
        let end = ((frame_idx + 1) * frame_stride_samples).min(samples.len());
        *active = rms_f32(&samples[start..end]) >= threshold;
    }
    mask
}

fn rms_f32(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

fn smooth_activity_mask(active: &mut [bool], min_speech_frames: usize, min_silence_frames: usize) {
    if active.is_empty() {
        return;
    }

    let mut idx = 0usize;
    while idx < active.len() {
        if active[idx] {
            idx += 1;
            continue;
        }
        let start = idx;
        while idx < active.len() && !active[idx] {
            idx += 1;
        }
        let end = idx;
        let gap_len = end - start;
        let has_left_speech = start > 0 && active[start - 1];
        let has_right_speech = end < active.len() && active[end];
        if has_left_speech && has_right_speech && gap_len <= min_silence_frames {
            for value in &mut active[start..end] {
                *value = true;
            }
        }
    }

    idx = 0;
    while idx < active.len() {
        if !active[idx] {
            idx += 1;
            continue;
        }
        let start = idx;
        while idx < active.len() && active[idx] {
            idx += 1;
        }
        let end = idx;
        if end - start < min_speech_frames {
            for value in &mut active[start..end] {
                *value = false;
            }
        }
    }
}

fn collect_active_regions(active: &[bool]) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut idx = 0usize;
    while idx < active.len() {
        if !active[idx] {
            idx += 1;
            continue;
        }
        let start = idx;
        while idx < active.len() && active[idx] {
            idx += 1;
        }
        let end = idx.saturating_sub(1);
        regions.push((start, end));
    }
    regions
}

fn merge_adjacent_segments(segments: &mut Vec<DiarizationSegment>, merge_gap_secs: f32) {
    if segments.len() <= 1 {
        return;
    }

    let mut by_speaker: BTreeMap<String, Vec<DiarizationSegment>> = BTreeMap::new();
    for segment in segments.drain(..) {
        by_speaker
            .entry(segment.speaker.clone())
            .or_default()
            .push(segment);
    }

    let mut merged_all = Vec::new();
    for (_, mut speaker_segments) in by_speaker {
        speaker_segments.sort_by(|a, b| a.start_secs.total_cmp(&b.start_secs));
        let mut iter = speaker_segments.into_iter();
        let Some(mut current) = iter.next() else {
            continue;
        };

        for segment in iter {
            let gap = (segment.start_secs - current.end_secs).max(0.0);
            if gap <= merge_gap_secs {
                current.end_secs = current.end_secs.max(segment.end_secs);
                current.confidence = match (current.confidence, segment.confidence) {
                    (Some(a), Some(b)) => Some((a + b) / 2.0),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                };
            } else {
                merged_all.push(current);
                current = segment;
            }
        }
        merged_all.push(current);
    }

    merged_all.sort_by(|a, b| {
        a.start_secs
            .total_cmp(&b.start_secs)
            .then(a.speaker.cmp(&b.speaker))
    });
    *segments = merged_all;
}

fn build_rel_positional_embedding(len: usize, d_model: usize, device: &Device) -> Result<Tensor> {
    if len == 0 {
        return Err(Error::InvalidInput(
            "Cannot build positional embedding for empty sequence".to_string(),
        ));
    }

    let pos_len = 2 * len - 1;
    let mut positions = Vec::with_capacity(pos_len);
    for p in (-(len as isize - 1))..=(len as isize - 1) {
        positions.push((-p) as f32);
    }

    let mut emb = vec![0f32; pos_len * d_model];
    let denom = (10_000f32).ln() / d_model as f32;

    for (pi, p) in positions.iter().enumerate() {
        for i in (0..d_model).step_by(2) {
            let div = (-denom * i as f32).exp();
            let angle = p * div;
            emb[pi * d_model + i] = angle.sin();
            if i + 1 < d_model {
                emb[pi * d_model + i + 1] = angle.cos();
            }
        }
    }

    Tensor::from_vec(emb, (1, pos_len, d_model), device).map_err(Error::from)
}

fn swish(x: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&ops::sigmoid(x)?).map_err(Error::from)
}

fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if audio.is_empty() || src_rate == 0 || dst_rate == 0 {
        return Vec::new();
    }
    if src_rate == dst_rate {
        return audio.to_vec();
    }

    let src_len = audio.len();
    let dst_len = ((src_len as u64) * (dst_rate as u64) / (src_rate as u64))
        .max(1)
        .min(usize::MAX as u64) as usize;
    let mut out = Vec::with_capacity(dst_len);

    let scale = src_rate as f64 / dst_rate as f64;
    for i in 0..dst_len {
        let src_pos = i as f64 * scale;
        let idx0 = src_pos.floor() as usize;
        let idx1 = (idx0 + 1).min(src_len.saturating_sub(1));
        let frac = (src_pos - idx0 as f64) as f32;
        let sample0 = audio[idx0];
        let sample1 = audio[idx1];
        out.push(sample0 + (sample1 - sample0) * frac);
    }

    out
}

fn hann_window(win_length: usize) -> Vec<f32> {
    if win_length <= 1 {
        return vec![1.0; win_length.max(1)];
    }

    (0..win_length)
        .map(|i| {
            let x = (2.0 * std::f32::consts::PI * i as f32) / (win_length as f32 - 1.0);
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

fn hz_to_mel_slaney(hz: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    if mel < min_log_mel {
        mel * f_sp
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

fn mel_filterbank(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let nyquist = sample_rate as f32 / 2.0;
    let mel_min = hz_to_mel_slaney(fmin.max(0.0));
    let mel_max = hz_to_mel_slaney(fmax.min(nyquist).max(fmin));

    let mel_points: Vec<f32> = (0..(n_mels + 2))
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz_slaney).collect();
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| nyquist * i as f32 / (n_freqs.saturating_sub(1).max(1)) as f32)
        .collect();

    let mut fb = vec![0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let left = hz_points[m];
        let center = hz_points[m + 1];
        let right = hz_points[m + 2];
        let lower_width = (center - left).max(1e-12);
        let upper_width = (right - center).max(1e-12);
        let enorm = if right > left {
            2.0 / (right - left)
        } else {
            0.0
        };

        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - left) / lower_width;
            let upper = (right - freq) / upper_width;
            fb[m * n_freqs + k] = lower.min(upper).max(0.0) * enorm;
        }
    }

    fb
}

fn preemphasis(x: &mut [f32], preemph: f32) {
    if x.len() < 2 {
        return;
    }

    let mut prev = x[0];
    for sample in x.iter_mut().skip(1) {
        let cur = *sample;
        *sample = cur - preemph * prev;
        prev = cur;
    }
}

fn normalize_per_feature(mel: &mut [f32], n_mels: usize, frames: usize, valid_frames: usize) {
    if valid_frames == 0 {
        return;
    }

    for m in 0..n_mels {
        let row = &mut mel[m * frames..(m + 1) * frames];

        let mean = row[..valid_frames].iter().copied().sum::<f32>() / valid_frames as f32;

        let var = if valid_frames > 1 {
            row[..valid_frames]
                .iter()
                .map(|v| {
                    let d = *v - mean;
                    d * d
                })
                .sum::<f32>()
                / (valid_frames as f32 - 1.0)
        } else {
            0.0
        };

        let std = var.sqrt() + NORMALIZE_EPS;
        for v in row[..valid_frames].iter_mut() {
            *v = (*v - mean) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realtime_voice_vad_frame_mask_uses_rms_threshold() {
        let samples = vec![
            0.0, 0.0, 0.0, 0.0, // silence
            0.3, 0.3, 0.3, 0.3, // speech
            0.0, 0.0, 0.0, 0.0, // silence
            0.4, 0.4, 0.4, 0.4, // speech
        ];
        let mask = realtime_voice_vad_frame_mask(&samples, 4, 4, 0.02);
        assert_eq!(mask, vec![false, true, false, true]);
    }

    #[test]
    fn select_speaker_channels_prefers_active_speakers_by_duration() {
        let stats = vec![
            SpeakerActivityStats {
                speaker_idx: 0,
                total_duration_secs: 8.0,
                peak_probability: 0.70,
                segment_count: 2,
            },
            SpeakerActivityStats {
                speaker_idx: 1,
                total_duration_secs: 2.0,
                peak_probability: 0.90,
                segment_count: 3,
            },
            SpeakerActivityStats {
                speaker_idx: 2,
                total_duration_secs: 5.0,
                peak_probability: 0.60,
                segment_count: 1,
            },
            SpeakerActivityStats {
                speaker_idx: 3,
                total_duration_secs: 0.0,
                peak_probability: 0.99,
                segment_count: 0,
            },
        ];

        let selected = select_speaker_channels(&stats, 1, 2);
        assert_eq!(selected, vec![0, 2]);
    }

    #[test]
    fn select_speaker_channels_backfills_when_min_exceeds_active() {
        let stats = vec![
            SpeakerActivityStats {
                speaker_idx: 0,
                total_duration_secs: 0.0,
                peak_probability: 0.40,
                segment_count: 0,
            },
            SpeakerActivityStats {
                speaker_idx: 1,
                total_duration_secs: 3.0,
                peak_probability: 0.60,
                segment_count: 2,
            },
            SpeakerActivityStats {
                speaker_idx: 2,
                total_duration_secs: 0.0,
                peak_probability: 0.80,
                segment_count: 0,
            },
            SpeakerActivityStats {
                speaker_idx: 3,
                total_duration_secs: 0.0,
                peak_probability: 0.20,
                segment_count: 0,
            },
        ];

        let selected = select_speaker_channels(&stats, 2, 2);
        assert_eq!(selected, vec![1, 2]);
    }

    #[test]
    fn binarization_matches_nemo_threshold_transitions() {
        let params = PostProcessingParams {
            onset: 0.5,
            offset: 0.5,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: 0.0,
            min_duration_off: 0.0,
            filter_speech_first: true,
        };

        let sequence = vec![0.1, 0.6, 0.7, 0.2, 0.1];
        let segments = binarization(&sequence, &params);

        assert_eq!(segments, vec![(0.01, 0.03)]);
    }

    #[test]
    fn filtering_merges_short_non_speech_gaps_like_nemo_default_order() {
        let params = PostProcessingParams {
            onset: 0.5,
            offset: 0.5,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: 0.0,
            min_duration_off: 0.15,
            filter_speech_first: true,
        };

        let segments = vec![(0.0, 0.5), (0.55, 1.0), (1.3, 1.7)];
        let filtered = filtering(&segments, &params);

        assert_eq!(filtered, vec![(0.0, 1.0), (1.3, 1.7)]);
    }

    #[test]
    fn filtering_respects_filter_speech_first_toggle() {
        let segments = vec![(0.0, 0.10), (0.14, 0.22)];
        let speech_first = PostProcessingParams {
            onset: 0.5,
            offset: 0.5,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: 0.12,
            min_duration_off: 0.08,
            filter_speech_first: true,
        };
        let nonspeech_first = PostProcessingParams {
            filter_speech_first: false,
            ..speech_first
        };

        assert!(filtering(&segments, &speech_first).is_empty());
        assert_eq!(filtering(&segments, &nonspeech_first), vec![(0.0, 0.22)]);
    }

    #[test]
    fn smooth_activity_mask_fills_gaps_and_removes_short_bursts() {
        let mut active = vec![
            true, true, false, true, true, false, false, false, true, false, false,
        ];
        smooth_activity_mask(&mut active, 2, 1);
        assert_eq!(
            active,
            vec![true, true, true, true, true, false, false, false, false, false, false]
        );
    }

    #[test]
    fn merge_adjacent_segments_merges_per_speaker_with_overlap_present() {
        let mut segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 1.0,
                confidence: Some(0.8),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.8,
                end_secs: 1.4,
                confidence: Some(0.9),
            },
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 1.05,
                end_secs: 2.0,
                confidence: Some(0.6),
            },
        ];

        merge_adjacent_segments(&mut segments, 0.1);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker, "SPEAKER_00");
        assert!((segments[0].start_secs - 0.0).abs() < 1e-6);
        assert!((segments[0].end_secs - 2.0).abs() < 1e-6);
        assert_eq!(segments[1].speaker, "SPEAKER_01");
    }
}
