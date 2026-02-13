use candle_core::Tensor;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::{Error, Result};

use super::config::PreprocessorConfig;

const LOG_GUARD: f32 = 5.960_464_5e-8;
const NORMALIZE_EPS: f32 = 1e-5;

#[derive(Debug, Clone)]
pub struct Lfm2AudioPreprocessor {
    cfg: PreprocessorConfig,
    padded_window: Vec<f32>,
    fb: Vec<f32>,
    n_mels: usize,
    n_freqs: usize,
    win_length: usize,
    hop_length: usize,
}

impl Lfm2AudioPreprocessor {
    pub fn new(cfg: PreprocessorConfig) -> Result<Self> {
        let win_length = (cfg.window_size * cfg.sample_rate as f32).round() as usize;
        let hop_length = (cfg.window_stride * cfg.sample_rate as f32).round() as usize;
        if win_length == 0 || hop_length == 0 || cfg.n_fft == 0 {
            return Err(Error::ModelLoadError(
                "Invalid LFM2 preprocessor window/hop/n_fft".to_string(),
            ));
        }

        let n_freqs = cfg.n_fft / 2 + 1;
        let n_mels = cfg.features;
        let window = hann_window(win_length);
        let mut padded_window = vec![0f32; cfg.n_fft];
        let offset = (cfg.n_fft - win_length) / 2;
        padded_window[offset..offset + win_length].copy_from_slice(&window);

        let fb = mel_filterbank(
            cfg.sample_rate,
            cfg.n_fft,
            n_mels,
            0.0,
            cfg.sample_rate as f32 / 2.0,
        );

        Ok(Self {
            cfg,
            padded_window,
            fb,
            n_mels,
            n_freqs,
            win_length,
            hop_length,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.cfg.sample_rate as u32
    }

    pub fn features(&self) -> usize {
        self.n_mels
    }

    pub fn compute_features(&self, audio: &[f32]) -> Result<(Tensor, usize)> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        // Keep parity with NeMo-style center padding.
        let center_pad = self.cfg.n_fft / 2;
        let mut padded = Vec::with_capacity(audio.len() + center_pad * 2);
        padded.extend(std::iter::repeat_n(0.0, center_pad));
        padded.extend_from_slice(audio);
        padded.extend(std::iter::repeat_n(0.0, center_pad));

        let frame_count = if padded.len() >= self.cfg.n_fft {
            (padded.len() - self.cfg.n_fft) / self.hop_length + 1
        } else {
            1
        };

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.cfg.n_fft);

        let mut spectrum = vec![0f32; frame_count * self.n_freqs];
        let mut buffer = vec![Complex::<f32>::new(0.0, 0.0); self.cfg.n_fft];

        for frame_idx in 0..frame_count {
            let start = frame_idx * self.hop_length;
            let slice = &padded[start..start + self.cfg.n_fft];

            for i in 0..self.cfg.n_fft {
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
                mel[m * frame_count + t] = if self.cfg.log {
                    (acc + LOG_GUARD).ln()
                } else {
                    acc
                };
            }
        }

        let valid_frames = audio.len() / self.hop_length;
        if self.cfg.normalize == "per_feature" {
            normalize_per_feature(
                &mut mel,
                self.n_mels,
                frame_count,
                valid_frames.min(frame_count),
            );
        }

        if valid_frames < frame_count {
            for m in 0..self.n_mels {
                for t in valid_frames..frame_count {
                    mel[m * frame_count + t] = self.cfg.pad_value;
                }
            }
        }

        let features = Tensor::from_vec(
            mel,
            (1, self.n_mels, frame_count),
            &candle_core::Device::Cpu,
        )?;

        Ok((features, valid_frames.min(frame_count)))
    }
}

pub fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if audio.is_empty() || src_rate == 0 || dst_rate == 0 || src_rate == dst_rate {
        return audio.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((audio.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = vec![0f32; out_len];

    for (i, sample) in out.iter_mut().enumerate() {
        let src_pos = i as f64 / ratio;
        let left = src_pos.floor() as usize;
        let right = (left + 1).min(audio.len().saturating_sub(1));
        let frac = (src_pos - left as f64) as f32;
        *sample = audio[left] * (1.0 - frac) + audio[right] * frac;
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

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

fn mel_filterbank(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax.min(sample_rate as f32 / 2.0));

    let mel_points: Vec<f32> = (0..(n_mels + 2))
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz).collect();
    let bins: Vec<usize> = hz_points
        .iter()
        .map(|hz| {
            (((n_fft + 1) as f32 * *hz / sample_rate as f32).floor() as isize).max(0) as usize
        })
        .collect();

    let mut fb = vec![0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let left = bins[m].min(n_freqs.saturating_sub(1));
        let center = bins[m + 1].min(n_freqs.saturating_sub(1));
        let right = bins[m + 2].min(n_freqs.saturating_sub(1));

        if center > left {
            for k in left..center {
                fb[m * n_freqs + k] = (k - left) as f32 / (center - left) as f32;
            }
        }
        if right > center {
            for k in center..right {
                fb[m * n_freqs + k] = (right - k) as f32 / (right - center) as f32;
            }
        }
    }

    fb
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
