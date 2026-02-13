use std::path::{Path, PathBuf};
use std::time::Instant;

use izwi_core::models::device::DeviceSelector;
use izwi_core::models::lfm2_audio::{lfm2_tts_voice_prompt, Lfm2AudioModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::var("IZWI_LFM2_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/Users/lennex/Library/Application Support/izwi/models/LFM2-Audio-1.5B")
        });
    let audio_path = std::env::var("IZWI_LFM2_AUDIO")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/fox.wav"));

    let device = DeviceSelector::detect_with_preference(Some("cpu"))?;
    println!("device_kind={:?}", device.kind);
    println!("model_dir={}", model_dir.display());
    println!("audio_path={}", audio_path.display());

    let model = Lfm2AudioModel::load(&model_dir, device)?;
    println!("voices={:?}", model.available_voices());

    let (mut audio, sample_rate) = read_wav(&audio_path)?;
    let max_len = (sample_rate as usize * 4).min(audio.len());
    audio.truncate(max_len);
    println!("audio_samples={} sample_rate={}", audio.len(), sample_rate);

    let asr_started = Instant::now();
    let mut asr_delta_chars = 0usize;
    let mut asr_delta = |d: &str| {
        asr_delta_chars += d.chars().count();
    };
    let asr_text = model.transcribe_with_callback(&audio, sample_rate, None, &mut asr_delta)?;
    println!(
        "asr_ms={:.1} asr_delta_chars={} asr_text_len={}",
        asr_started.elapsed().as_secs_f64() * 1000.0,
        asr_delta_chars,
        asr_text.chars().count()
    );
    println!("asr_text={}", asr_text);

    let tts_started = Instant::now();
    let mut tts_delta_chars = 0usize;
    let mut tts_delta = |d: &str| {
        tts_delta_chars += d.chars().count();
    };
    let tts_samples = model.synthesize_with_callback(
        "Native rust Candle smoke test.",
        lfm2_tts_voice_prompt(Some("US Female")),
        Some(0.0),
        Some(1),
        192,
        &mut tts_delta,
    )?;
    println!(
        "tts_ms={:.1} tts_delta_chars={} tts_samples={}",
        tts_started.elapsed().as_secs_f64() * 1000.0,
        tts_delta_chars,
        tts_samples.len()
    );

    let s2s_started = Instant::now();
    let mut s2s_delta_chars = 0usize;
    let mut s2s_delta = |d: &str| {
        s2s_delta_chars += d.chars().count();
    };
    let (s2s_text, s2s_samples) = model.speech_to_speech_with_callback(
        &audio,
        sample_rate,
        None,
        Some(0.0),
        Some(1),
        192,
        &mut s2s_delta,
    )?;
    println!(
        "s2s_ms={:.1} s2s_delta_chars={} s2s_text_len={} s2s_samples={}",
        s2s_started.elapsed().as_secs_f64() * 1000.0,
        s2s_delta_chars,
        s2s_text.chars().count(),
        s2s_samples.len()
    );
    println!("s2s_text={}", s2s_text);

    Ok(())
}

fn read_wav(path: &Path) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels.max(1) as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample.clamp(1, 32) as u32;
            let scale = ((1i64 << (bits - 1)) - 1).max(1) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| (v as f32) / scale))
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    if channels == 1 {
        return Ok((samples, sample_rate));
    }

    let frames = samples.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut sum = 0.0f32;
        for c in 0..channels {
            sum += samples[f * channels + c];
        }
        mono.push(sum / channels as f32);
    }

    Ok((mono, sample_rate))
}
