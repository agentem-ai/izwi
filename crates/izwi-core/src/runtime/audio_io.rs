//! Audio decode and preprocessing helpers used by runtime task handlers.

use tracing::debug;

use crate::error::{Error, Result};

pub(crate) fn base64_decode(data: &str) -> Result<Vec<u8>> {
    use base64::Engine;

    let payload = if data.starts_with("data:") {
        data.split_once(',').map(|(_, b64)| b64).unwrap_or(data)
    } else {
        data
    };

    let normalized: String = payload.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(normalized.as_bytes())
        .map_err(|e| Error::InferenceError(format!("Base64 decode error: {}", e)))
}

pub(crate) fn decode_wav_bytes(wav_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    use std::io::Cursor;

    let cursor = Cursor::new(wav_bytes);
    let mut reader = hound::WavReader::new(cursor)
        .map_err(|e| Error::InferenceError(format!("Failed to parse WAV: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels.max(1) as usize;

    let mut samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample.max(1) as u32;
            let max_val = if bits > 1 {
                ((1i64 << (bits - 1)) - 1) as f32
            } else {
                1.0
            };
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| (s as f32 / max_val).clamp(-1.0, 1.0))
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };

    if channels > 1 {
        let mut mono = Vec::with_capacity(samples.len() / channels + 1);
        for frame in samples.chunks(channels) {
            if frame.is_empty() {
                continue;
            }
            let sum: f32 = frame.iter().copied().sum();
            mono.push(sum / frame.len() as f32);
        }
        samples = mono;
    }

    for sample in &mut samples {
        if !sample.is_finite() {
            *sample = 0.0;
        } else {
            *sample = sample.clamp(-1.0, 1.0);
        }
    }

    Ok((samples, sample_rate))
}

pub(crate) fn preprocess_reference_audio(mut samples: Vec<f32>, sample_rate: u32) -> Vec<f32> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }

    let original_len = samples.len();

    for sample in &mut samples {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }

    // Remove DC bias.
    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    for sample in &mut samples {
        *sample -= mean;
    }

    let initial_peak = samples.iter().fold(0.0f32, |p, &s| p.max(s.abs()));
    if initial_peak < 1e-5 {
        return Vec::new();
    }

    // Trim leading/trailing silence while keeping short context margins.
    let silence_threshold = (initial_peak * 0.04).max(0.0025);
    let first_idx = samples.iter().position(|s| s.abs() >= silence_threshold);
    let last_idx = samples.iter().rposition(|s| s.abs() >= silence_threshold);
    if let (Some(first), Some(last)) = (first_idx, last_idx) {
        let margin = ((sample_rate as f32) * 0.12) as usize;
        let start = first.saturating_sub(margin);
        let end = (last + margin + 1).min(samples.len());
        samples = samples[start..end].to_vec();
    }

    // Bound reference length to avoid conditioning on long silence/noise tails.
    let max_seconds = 12usize;
    let max_len = sample_rate as usize * max_seconds;
    if samples.len() > max_len && max_len > 0 {
        let window = (sample_rate as usize * 6).clamp(sample_rate as usize, samples.len());
        let best_start = highest_energy_window_start(&samples, window);
        let start = best_start.min(samples.len() - max_len);
        samples = samples[start..start + max_len].to_vec();
    }

    // Normalize into a practical loudness band so encoder sees stable dynamics.
    let mut peak = samples.iter().fold(0.0f32, |p, &s| p.max(s.abs()));
    if peak > 0.95 {
        let scale = 0.95 / peak;
        for sample in &mut samples {
            *sample *= scale;
        }
    }

    let rms = (samples
        .iter()
        .map(|&s| (s as f64) * (s as f64))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt() as f32;
    let min_rms = 0.035f32;
    if rms > 1e-6 && rms < min_rms {
        let gain = (min_rms / rms).min(6.0);
        for sample in &mut samples {
            *sample *= gain;
        }
    }

    // Final hard limit.
    peak = samples.iter().fold(0.0f32, |p, &s| p.max(s.abs()));
    if peak > 0.95 {
        let scale = 0.95 / peak;
        for sample in &mut samples {
            *sample *= scale;
        }
    }

    debug!(
        "Reference preprocessing: {} -> {} samples @ {} Hz",
        original_len,
        samples.len(),
        sample_rate
    );

    samples
}

fn highest_energy_window_start(samples: &[f32], window: usize) -> usize {
    if samples.is_empty() || window == 0 || samples.len() <= window {
        return 0;
    }

    let mut prefix = Vec::with_capacity(samples.len() + 1);
    prefix.push(0.0f64);
    for &sample in samples {
        let e = (sample as f64) * (sample as f64);
        let next = prefix.last().copied().unwrap_or(0.0) + e;
        prefix.push(next);
    }

    let mut best_start = 0usize;
    let mut best_energy = f64::NEG_INFINITY;
    for start in 0..=samples.len() - window {
        let end = start + window;
        let energy = prefix[end] - prefix[start];
        if energy > best_energy {
            best_energy = energy;
            best_start = start;
        }
    }

    best_start
}
