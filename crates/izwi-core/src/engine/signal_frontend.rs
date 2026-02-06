//! Signal Frontend for audio processing pipeline.
//!
//! This module implements the first stage of the audio inference pipeline:
//! - Voice Activity Detection (VAD) - detects speech vs silence
//! - Feature Extraction - converts audio to mel-spectrograms
//! - Audio Chunking - manages overlapping audio windows
//!
//! # Architecture
//!
//! ```text
//! [ Raw Audio Stream ]
//!         ↓
//! ┌───────────────────────────────────────┐
//! │          SIGNAL FRONTEND              │
//! │  ┌─────────────┐  ┌────────────────┐  │
//! │  │     VAD     │  │    Feature     │  │
//! │  │  (Silero)   │→ │   Extractor    │  │
//! │  └─────────────┘  │  (Mel-Spec)    │  │
//! │                   └────────────────┘  │
//! │  ┌─────────────────────────────────┐  │
//! │  │      Look-ahead Buffer          │  │
//! │  │  (Manages overlapping chunks)   │  │
//! │  └─────────────────────────────────┘  │
//! └───────────────────────────────────────┘
//!         ↓
//! [ Audio Tokens / Features ]
//! ```

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Configuration for the signal frontend.
#[derive(Debug, Clone)]
pub struct SignalFrontendConfig {
    /// Sample rate of input audio (Hz)
    pub sample_rate: u32,
    /// Frame size for feature extraction (samples)
    pub frame_size: usize,
    /// Hop size between frames (samples)
    pub hop_size: usize,
    /// Number of mel filterbank channels
    pub num_mel_bins: usize,
    /// FFT size for spectrogram computation
    pub fft_size: usize,
    /// VAD threshold (0.0 - 1.0)
    pub vad_threshold: f32,
    /// Minimum speech duration to trigger (ms)
    pub min_speech_duration_ms: u32,
    /// Silence duration to end speech segment (ms)
    pub silence_end_duration_ms: u32,
    /// Look-ahead buffer size (ms)
    pub lookahead_buffer_ms: u32,
}

impl Default for SignalFrontendConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_size: 400, // 25ms at 16kHz
            hop_size: 160,   // 10ms at 16kHz
            num_mel_bins: 80,
            fft_size: 512,
            vad_threshold: 0.5,
            min_speech_duration_ms: 250,
            silence_end_duration_ms: 500,
            lookahead_buffer_ms: 100,
        }
    }
}

/// Voice Activity Detection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadState {
    /// No speech detected
    Silence,
    /// Speech detected and ongoing
    Speech,
    /// Speech just ended (in cooldown)
    SpeechEnding,
}

/// Result from VAD processing.
#[derive(Debug, Clone)]
pub struct VadResult {
    /// Current VAD state
    pub state: VadState,
    /// Speech probability (0.0 - 1.0)
    pub speech_probability: f32,
    /// Duration of current state
    pub state_duration: Duration,
    /// Whether this is an interruption (speech during AI output)
    pub is_interruption: bool,
}

/// An audio chunk with features extracted.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Raw audio samples
    pub samples: Vec<f32>,
    /// Mel-spectrogram features (if extracted)
    pub mel_features: Option<Vec<Vec<f32>>>,
    /// Timestamp of chunk start
    pub timestamp: Duration,
    /// Duration of the chunk
    pub duration: Duration,
    /// VAD result for this chunk
    pub vad_result: VadResult,
}

/// Voice Activity Detector.
///
/// Currently implements a simple energy-based VAD.
/// TODO: Integrate Silero VAD for production use.
pub struct VoiceActivityDetector {
    config: SignalFrontendConfig,
    state: VadState,
    state_start: Instant,
    speech_frames: usize,
    silence_frames: usize,
    /// Energy threshold for simple VAD
    energy_threshold: f32,
    /// Smoothed speech probability
    smoothed_prob: f32,
}

impl VoiceActivityDetector {
    /// Create a new VAD instance.
    pub fn new(config: SignalFrontendConfig) -> Self {
        Self {
            config,
            state: VadState::Silence,
            state_start: Instant::now(),
            speech_frames: 0,
            silence_frames: 0,
            energy_threshold: 0.01,
            smoothed_prob: 0.0,
        }
    }

    /// Process an audio frame and return VAD result.
    pub fn process(&mut self, samples: &[f32], is_ai_speaking: bool) -> VadResult {
        // Calculate frame energy (RMS)
        let energy = Self::calculate_rms(samples);

        // Simple energy-based speech probability
        // In production, this would use Silero VAD model
        let raw_prob = (energy / self.energy_threshold).min(1.0);

        // Exponential smoothing
        let alpha = 0.3;
        self.smoothed_prob = alpha * raw_prob + (1.0 - alpha) * self.smoothed_prob;

        let is_speech = self.smoothed_prob > self.config.vad_threshold;

        // State machine for VAD
        let frame_duration_ms =
            (self.config.hop_size as f32 / self.config.sample_rate as f32 * 1000.0) as u32;

        match self.state {
            VadState::Silence => {
                if is_speech {
                    self.speech_frames += 1;
                    let speech_duration = self.speech_frames as u32 * frame_duration_ms;
                    if speech_duration >= self.config.min_speech_duration_ms {
                        self.state = VadState::Speech;
                        self.state_start = Instant::now();
                        self.silence_frames = 0;
                    }
                } else {
                    self.speech_frames = 0;
                }
            }
            VadState::Speech => {
                if !is_speech {
                    self.silence_frames += 1;
                    let silence_duration = self.silence_frames as u32 * frame_duration_ms;
                    if silence_duration >= self.config.silence_end_duration_ms {
                        self.state = VadState::SpeechEnding;
                        self.state_start = Instant::now();
                    }
                } else {
                    self.silence_frames = 0;
                }
            }
            VadState::SpeechEnding => {
                if is_speech {
                    self.state = VadState::Speech;
                    self.state_start = Instant::now();
                    self.silence_frames = 0;
                } else {
                    self.state = VadState::Silence;
                    self.state_start = Instant::now();
                    self.speech_frames = 0;
                }
            }
        }

        VadResult {
            state: self.state,
            speech_probability: self.smoothed_prob,
            state_duration: self.state_start.elapsed(),
            is_interruption: is_speech && is_ai_speaking,
        }
    }

    /// Calculate RMS energy of samples.
    fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Reset VAD state.
    pub fn reset(&mut self) {
        self.state = VadState::Silence;
        self.state_start = Instant::now();
        self.speech_frames = 0;
        self.silence_frames = 0;
        self.smoothed_prob = 0.0;
    }

    /// Get current state.
    pub fn state(&self) -> VadState {
        self.state
    }
}

/// Feature extractor for converting audio to mel-spectrograms.
pub struct FeatureExtractor {
    config: SignalFrontendConfig,
    /// Hanning window for FFT
    window: Vec<f32>,
    /// Mel filterbank
    mel_filterbank: Vec<Vec<f32>>,
}

impl FeatureExtractor {
    /// Create a new feature extractor.
    pub fn new(config: SignalFrontendConfig) -> Self {
        let window = Self::hanning_window(config.frame_size);
        let mel_filterbank =
            Self::create_mel_filterbank(config.fft_size, config.sample_rate, config.num_mel_bins);

        Self {
            config,
            window,
            mel_filterbank,
        }
    }

    /// Extract mel-spectrogram features from audio samples.
    pub fn extract(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mut features = Vec::new();
        let num_frames =
            (samples.len().saturating_sub(self.config.frame_size)) / self.config.hop_size + 1;

        for i in 0..num_frames {
            let start = i * self.config.hop_size;
            let end = (start + self.config.frame_size).min(samples.len());
            let frame = &samples[start..end];

            if let Some(mel_frame) = self.process_frame(frame) {
                features.push(mel_frame);
            }
        }

        features
    }

    /// Process a single frame to get mel-spectrogram.
    fn process_frame(&self, frame: &[f32]) -> Option<Vec<f32>> {
        if frame.len() < self.config.frame_size {
            return None;
        }

        // Apply window
        let windowed: Vec<f32> = frame
            .iter()
            .zip(self.window.iter())
            .map(|(s, w)| s * w)
            .collect();

        // Compute power spectrum (simplified - real implementation would use FFT)
        let power_spectrum = self.compute_power_spectrum(&windowed);

        // Apply mel filterbank
        let mel_features: Vec<f32> = self
            .mel_filterbank
            .iter()
            .map(|filter| {
                let energy: f32 = filter
                    .iter()
                    .zip(power_spectrum.iter())
                    .map(|(f, p)| f * p)
                    .sum();
                // Log mel energy
                (energy.max(1e-10)).ln()
            })
            .collect();

        Some(mel_features)
    }

    /// Compute power spectrum (simplified).
    fn compute_power_spectrum(&self, samples: &[f32]) -> Vec<f32> {
        // Simplified power spectrum using autocorrelation approximation
        // In production, use proper FFT (rustfft crate)
        let n = self.config.fft_size / 2 + 1;
        let mut spectrum = vec![0.0f32; n];

        for (i, spec) in spectrum.iter_mut().enumerate() {
            let freq_bin = i as f32 / n as f32;
            let mut power = 0.0f32;
            for (j, &sample) in samples.iter().enumerate() {
                let phase = 2.0 * std::f32::consts::PI * freq_bin * j as f32;
                power += sample * phase.cos();
            }
            *spec = power * power / samples.len() as f32;
        }

        spectrum
    }

    /// Create Hanning window.
    fn hanning_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos())
            })
            .collect()
    }

    /// Create mel filterbank.
    fn create_mel_filterbank(
        fft_size: usize,
        sample_rate: u32,
        num_mel_bins: usize,
    ) -> Vec<Vec<f32>> {
        let num_fft_bins = fft_size / 2 + 1;
        let nyquist = sample_rate as f32 / 2.0;

        // Mel scale conversion
        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(nyquist);

        // Create mel points
        let mel_points: Vec<f32> = (0..=num_mel_bins + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (num_mel_bins + 1) as f32)
            .collect();

        // Convert to Hz and then to FFT bin indices
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((hz / nyquist) * (num_fft_bins - 1) as f32) as usize)
            .collect();

        // Create triangular filters
        let mut filterbank = Vec::with_capacity(num_mel_bins);
        for i in 0..num_mel_bins {
            let mut filter = vec![0.0f32; num_fft_bins];
            let left = bin_points[i];
            let center = bin_points[i + 1];
            let right = bin_points[i + 2];

            // Rising slope
            for j in left..center {
                if center > left {
                    filter[j] = (j - left) as f32 / (center - left) as f32;
                }
            }

            // Falling slope
            for j in center..right {
                if right > center {
                    filter[j] = (right - j) as f32 / (right - center) as f32;
                }
            }

            filterbank.push(filter);
        }

        filterbank
    }
}

/// Look-ahead buffer for managing overlapping audio chunks.
pub struct LookaheadBuffer {
    config: SignalFrontendConfig,
    /// Buffer of samples
    buffer: VecDeque<f32>,
    /// Maximum buffer size in samples
    max_samples: usize,
    /// Current timestamp
    current_time: Duration,
}

impl LookaheadBuffer {
    /// Create a new look-ahead buffer.
    pub fn new(config: SignalFrontendConfig) -> Self {
        let max_samples =
            (config.sample_rate as usize * config.lookahead_buffer_ms as usize) / 1000;
        Self {
            config,
            buffer: VecDeque::with_capacity(max_samples),
            max_samples,
            current_time: Duration::ZERO,
        }
    }

    /// Push new samples into the buffer.
    pub fn push(&mut self, samples: &[f32]) {
        for &sample in samples {
            if self.buffer.len() >= self.max_samples {
                self.buffer.pop_front();
            }
            self.buffer.push_back(sample);
        }

        let duration =
            Duration::from_secs_f64(samples.len() as f64 / self.config.sample_rate as f64);
        self.current_time += duration;
    }

    /// Get the current buffer contents.
    pub fn get_buffer(&self) -> Vec<f32> {
        self.buffer.iter().copied().collect()
    }

    /// Get buffer with context (includes look-ahead).
    pub fn get_with_context(&self, num_samples: usize) -> Vec<f32> {
        let start = self.buffer.len().saturating_sub(num_samples);
        self.buffer.iter().skip(start).copied().collect()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.current_time = Duration::ZERO;
    }

    /// Get current timestamp.
    pub fn timestamp(&self) -> Duration {
        self.current_time
    }

    /// Get buffer length in samples.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// The complete Signal Frontend.
pub struct SignalFrontend {
    config: SignalFrontendConfig,
    vad: VoiceActivityDetector,
    feature_extractor: FeatureExtractor,
    lookahead_buffer: LookaheadBuffer,
    /// Whether AI is currently outputting audio
    ai_speaking: bool,
}

impl SignalFrontend {
    /// Create a new signal frontend.
    pub fn new(config: SignalFrontendConfig) -> Self {
        Self {
            vad: VoiceActivityDetector::new(config.clone()),
            feature_extractor: FeatureExtractor::new(config.clone()),
            lookahead_buffer: LookaheadBuffer::new(config.clone()),
            config,
            ai_speaking: false,
        }
    }

    /// Process incoming audio samples.
    ///
    /// Returns an AudioChunk with features and VAD result.
    pub fn process(&mut self, samples: &[f32]) -> AudioChunk {
        // Add to look-ahead buffer
        self.lookahead_buffer.push(samples);

        // Run VAD
        let vad_result = self.vad.process(samples, self.ai_speaking);

        // Extract features if speech is detected
        let mel_features = if vad_result.state == VadState::Speech {
            Some(self.feature_extractor.extract(samples))
        } else {
            None
        };

        let duration =
            Duration::from_secs_f64(samples.len() as f64 / self.config.sample_rate as f64);

        AudioChunk {
            samples: samples.to_vec(),
            mel_features,
            timestamp: self.lookahead_buffer.timestamp() - duration,
            duration,
            vad_result,
        }
    }

    /// Set whether AI is currently speaking (for interruption detection).
    pub fn set_ai_speaking(&mut self, speaking: bool) {
        self.ai_speaking = speaking;
    }

    /// Check if an interruption was detected.
    pub fn is_interruption(&self, vad_result: &VadResult) -> bool {
        vad_result.is_interruption
    }

    /// Reset the frontend state.
    pub fn reset(&mut self) {
        self.vad.reset();
        self.lookahead_buffer.clear();
        self.ai_speaking = false;
    }

    /// Get VAD state.
    pub fn vad_state(&self) -> VadState {
        self.vad.state()
    }

    /// Get configuration.
    pub fn config(&self) -> &SignalFrontendConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_silence() {
        let config = SignalFrontendConfig::default();
        let mut vad = VoiceActivityDetector::new(config);

        // Silent samples
        let samples = vec![0.0f32; 160];
        let result = vad.process(&samples, false);

        assert_eq!(result.state, VadState::Silence);
        assert!(result.speech_probability < 0.1);
    }

    #[test]
    fn test_vad_speech() {
        let config = SignalFrontendConfig {
            min_speech_duration_ms: 0, // Instant detection for test
            ..Default::default()
        };
        let mut vad = VoiceActivityDetector::new(config);

        // "Loud" samples
        let samples: Vec<f32> = (0..160).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();

        // Process multiple frames to trigger speech state
        for _ in 0..10 {
            vad.process(&samples, false);
        }

        let result = vad.process(&samples, false);
        assert!(result.speech_probability > 0.1);
    }

    #[test]
    fn test_feature_extractor() {
        let config = SignalFrontendConfig::default();
        let extractor = FeatureExtractor::new(config.clone());

        // Generate test signal
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();

        let features = extractor.extract(&samples);

        assert!(!features.is_empty());
        assert_eq!(features[0].len(), config.num_mel_bins);
    }

    #[test]
    fn test_lookahead_buffer() {
        let config = SignalFrontendConfig {
            lookahead_buffer_ms: 100,
            sample_rate: 16000,
            ..Default::default()
        };
        let mut buffer = LookaheadBuffer::new(config);

        // Push samples
        let samples = vec![1.0f32; 800]; // 50ms
        buffer.push(&samples);

        assert_eq!(buffer.len(), 800);

        // Push more (should not exceed max)
        buffer.push(&samples);
        buffer.push(&samples);

        // Max is 100ms = 1600 samples
        assert!(buffer.len() <= 1600);
    }
}
