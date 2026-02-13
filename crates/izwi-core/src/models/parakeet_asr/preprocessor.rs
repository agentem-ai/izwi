use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::{Error, Result};

const PREEMPH: f32 = 0.97;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const LOG_GUARD: f32 = 5.960_464_5e-8;
const NORMALIZE_EPS: f32 = 1e-5;

#[derive(Debug, Clone)]
pub struct ParakeetPreprocessor {
    _window: Vec<f32>,       // [win_length]
    padded_window: Vec<f32>, // [n_fft]
    fb: Vec<f32>,            // [n_mels * (n_fft/2+1)]
    n_mels: usize,
    n_freqs: usize,
}

impl ParakeetPreprocessor {
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let window = vb
            .pp("preprocessor.featurizer")
            .get_unchecked_dtype("window", DType::F32)?
            .to_vec1::<f32>()?;

        if window.len() != WIN_LENGTH {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Parakeet window length: expected {}, got {}",
                WIN_LENGTH,
                window.len()
            )));
        }

        let fb_tensor = vb
            .pp("preprocessor.featurizer")
            .get_unchecked_dtype("fb", DType::F32)?;
        let (_, n_mels, n_freqs) = fb_tensor.dims3()?;
        let fb = fb_tensor.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;

        if n_freqs != (N_FFT / 2 + 1) {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Parakeet filterbank bins: expected {}, got {}",
                N_FFT / 2 + 1,
                n_freqs
            )));
        }

        let mut padded_window = vec![0f32; N_FFT];
        let offset = (N_FFT - WIN_LENGTH) / 2;
        padded_window[offset..offset + WIN_LENGTH].copy_from_slice(&window);

        Ok(Self {
            _window: window,
            padded_window,
            fb,
            n_mels,
            n_freqs,
        })
    }

    pub fn compute_features(&self, audio: &[f32]) -> Result<(Tensor, usize)> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let mut x = audio.to_vec();
        preemphasis(&mut x, PREEMPH);

        // torch.stft(center=True, pad_mode="constant")
        let center_pad = N_FFT / 2;
        let mut padded = Vec::with_capacity(x.len() + center_pad * 2);
        padded.extend(std::iter::repeat(0.0).take(center_pad));
        padded.extend_from_slice(&x);
        padded.extend(std::iter::repeat(0.0).take(center_pad));

        let frame_count = if padded.len() >= N_FFT {
            (padded.len() - N_FFT) / HOP_LENGTH + 1
        } else {
            1
        };

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        let mut spectrum = vec![0f32; frame_count * self.n_freqs];
        let mut buffer = vec![Complex::<f32>::new(0.0, 0.0); N_FFT];

        for frame_idx in 0..frame_count {
            let start = frame_idx * HOP_LENGTH;
            let slice = &padded[start..start + N_FFT];

            for i in 0..N_FFT {
                buffer[i].re = slice[i] * self.padded_window[i];
                buffer[i].im = 0.0;
            }

            fft.process(&mut buffer);

            for k in 0..self.n_freqs {
                let mag = (buffer[k].re * buffer[k].re + buffer[k].im * buffer[k].im).sqrt();
                spectrum[frame_idx * self.n_freqs + k] = mag * mag; // mag_power=2
            }
        }

        // Mel projection and log.
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

        // NeMo get_seq_len for center=True case: floor(seq_len / hop_length)
        let valid_frames = audio.len() / HOP_LENGTH;

        // normalize per_feature (along time)
        normalize_per_feature(
            &mut mel,
            self.n_mels,
            frame_count,
            valid_frames.min(frame_count),
        );

        // mask padded frames to zero
        if valid_frames < frame_count {
            for m in 0..self.n_mels {
                for t in valid_frames..frame_count {
                    mel[m * frame_count + t] = 0.0;
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
