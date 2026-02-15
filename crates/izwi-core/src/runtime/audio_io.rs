//! Audio decode and preprocessing helpers used by runtime task handlers.

use std::io::Cursor;

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

pub(crate) fn decode_audio_bytes(audio_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    if audio_bytes.is_empty() {
        return Err(Error::InvalidInput("Empty audio input".to_string()));
    }

    match decode_audio_bytes_symphonia(audio_bytes) {
        Ok((samples, sample_rate)) => finalize_decoded_audio(samples, sample_rate),
        Err(symphonia_err) => {
            let (samples, sample_rate) =
                decode_wav_bytes_hound(audio_bytes).map_err(|wav_err| {
                    Error::InferenceError(format!(
                    "Failed to decode audio. Symphonia: {symphonia_err}; WAV fallback: {wav_err}"
                ))
                })?;
            finalize_decoded_audio(samples, sample_rate)
        }
    }
}

pub(crate) fn decode_wav_bytes(wav_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    decode_audio_bytes(wav_bytes)
}

fn decode_audio_bytes_symphonia(audio_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;
    use symphonia::default::{get_codecs, get_probe};

    let media_source = MediaSourceStream::new(
        Box::new(Cursor::new(audio_bytes.to_vec())),
        Default::default(),
    );
    let hint = Hint::new();
    let probed = get_probe()
        .format(
            &hint,
            media_source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| Error::InferenceError(format!("Symphonia probe failed: {e}")))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| Error::InferenceError("No default audio track found".to_string()))?;
    let track_id = track.id;
    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut decoder = get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| Error::InferenceError(format!("Failed to create audio decoder: {e}")))?;

    let mut samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err(Error::InferenceError(
                    "Audio stream format reset is not supported".to_string(),
                ));
            }
            Err(SymphoniaError::IoError(_)) => break,
            Err(err) => {
                return Err(Error::InferenceError(format!(
                    "Failed reading audio packets: {err}"
                )));
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err(Error::InferenceError(
                    "Audio decoder reset is not supported".to_string(),
                ));
            }
            Err(err) => {
                return Err(Error::InferenceError(format!(
                    "Failed decoding audio packet: {err}"
                )));
            }
        };

        if sample_rate == 0 {
            sample_rate = decoded.spec().rate;
        }
        let channels = decoded.spec().channels.count().max(1);
        append_decoded_packet(decoded, channels, &mut samples);
    }

    if sample_rate == 0 {
        return Err(Error::InferenceError(
            "Decoded audio is missing sample rate metadata".to_string(),
        ));
    }
    if samples.is_empty() {
        return Err(Error::InferenceError(
            "Decoded audio produced zero samples".to_string(),
        ));
    }

    Ok((samples, sample_rate))
}

fn append_decoded_packet(
    decoded: symphonia::core::audio::AudioBufferRef<'_>,
    channels: usize,
    out: &mut Vec<f32>,
) {
    use symphonia::core::audio::SampleBuffer;

    let mut sample_buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
    sample_buffer.copy_interleaved_ref(decoded);
    let data = sample_buffer.samples();

    if channels <= 1 {
        out.extend_from_slice(data);
        return;
    }

    for frame in data.chunks(channels) {
        if frame.is_empty() {
            continue;
        }
        let sum: f32 = frame.iter().copied().sum();
        out.push(sum / frame.len() as f32);
    }
}

fn decode_wav_bytes_hound(wav_bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
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

    Ok((samples, sample_rate))
}

fn finalize_decoded_audio(mut samples: Vec<f32>, sample_rate: u32) -> Result<(Vec<f32>, u32)> {
    if sample_rate == 0 {
        return Err(Error::InferenceError(
            "Decoded audio has invalid sample rate 0".to_string(),
        ));
    }
    if samples.is_empty() {
        return Err(Error::InferenceError(
            "Decoded audio contains no samples".to_string(),
        ));
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

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    #[test]
    fn base64_decode_accepts_data_uri_payload() {
        let payload = [1u8, 2, 3, 4, 5];
        let b64 = base64::engine::general_purpose::STANDARD.encode(payload);
        let uri = format!("data:audio/mpeg;base64,{b64}");
        let decoded = base64_decode(&uri).expect("data URI decode should succeed");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn decode_audio_bytes_downmixes_stereo_wav() {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            writer.write_sample((0.25f32 * 32767.0) as i16).unwrap();
            writer.write_sample((0.75f32 * 32767.0) as i16).unwrap();
            writer.write_sample((0.5f32 * 32767.0) as i16).unwrap();
            writer.write_sample((-0.5f32 * 32767.0) as i16).unwrap();
            writer.finalize().unwrap();
        }

        let (samples, sample_rate) =
            decode_audio_bytes(&wav_bytes).expect("decode should succeed for WAV bytes");
        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples.len(), 2);
        assert!(
            (samples[0] - 0.5).abs() < 0.02,
            "first sample {}",
            samples[0]
        );
        assert!(samples[1].abs() < 0.02, "second sample {}", samples[1]);
    }
}
