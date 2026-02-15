mod nemo;

use std::collections::HashMap;
use std::path::Path;

use candle_core::pickle::read_pth_tensor_info;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::runtime::{DiarizationConfig, DiarizationResult, DiarizationSegment};

use nemo::{ensure_sortformer_artifacts, SortformerArtifacts};

const TARGET_SAMPLE_RATE: u32 = 16_000;
const MAX_SUPPORTED_SPEAKERS: usize = 4;
const FRAME_MS: f32 = 80.0;
const HOP_MS: f32 = 40.0;
const DEFAULT_MIN_SPEECH_MS: f32 = 240.0;
const DEFAULT_MIN_SILENCE_MS: f32 = 200.0;
const FFT_SIZE: usize = 512;

#[derive(Debug, Clone)]
struct FrameFeatures {
    start_sample: usize,
    end_sample: usize,
    rms_db: f32,
    vector: Vec<f32>,
}

pub struct SortformerDiarizerModel {
    variant: ModelVariant,
    _artifacts: SortformerArtifacts,
    _checkpoint_tensor_count: usize,
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
            read_pth_tensor_info(&artifacts.checkpoint_path, false, None).map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to inspect Sortformer checkpoint {}: {}",
                    artifacts.checkpoint_path.display(),
                    e
                ))
            })?;
        Ok(Self {
            variant,
            _artifacts: artifacts,
            _checkpoint_tensor_count: tensor_info.len(),
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

        let frame_len = ((TARGET_SAMPLE_RATE as f32 * FRAME_MS) / 1000.0).round() as usize;
        let hop_len = ((TARGET_SAMPLE_RATE as f32 * HOP_MS) / 1000.0).round() as usize;
        let frame_len = frame_len.max(1);
        let hop_len = hop_len.max(1);

        let frames = extract_frame_features(&samples, frame_len, hop_len)?;
        if frames.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                duration_secs,
                speaker_count: 0,
            });
        }

        let rms_values: Vec<f32> = frames.iter().map(|f| f.rms_db).collect();
        let vad_threshold_db = adaptive_vad_threshold_db(&rms_values);
        let mut active: Vec<bool> = rms_values.iter().map(|db| *db > vad_threshold_db).collect();

        let min_speech_ms = config
            .min_speech_duration_ms
            .unwrap_or(DEFAULT_MIN_SPEECH_MS)
            .clamp(HOP_MS, 5000.0);
        let min_silence_ms = config
            .min_silence_duration_ms
            .unwrap_or(DEFAULT_MIN_SILENCE_MS)
            .clamp(HOP_MS, 5000.0);

        let min_speech_frames = ((min_speech_ms / HOP_MS).round() as usize).max(1);
        let min_silence_frames = ((min_silence_ms / HOP_MS).round() as usize).max(1);
        smooth_activity_mask(&mut active, min_speech_frames, min_silence_frames);

        let regions = collect_active_regions(&active);
        if regions.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                duration_secs,
                speaker_count: 0,
            });
        }

        let mut embeddings = Vec::with_capacity(regions.len());
        for (start_idx, end_idx) in &regions {
            let mut count = 0usize;
            let mut acc = vec![0.0f32; frames[*start_idx].vector.len()];
            for frame in &frames[*start_idx..=*end_idx] {
                for (idx, value) in frame.vector.iter().enumerate() {
                    acc[idx] += *value;
                }
                count += 1;
            }
            if count > 0 {
                for value in &mut acc {
                    *value /= count as f32;
                }
            }
            embeddings.push(acc);
        }

        let requested_max = config.max_speakers.unwrap_or(MAX_SUPPORTED_SPEAKERS);
        let max_speakers = requested_max.clamp(1, MAX_SUPPORTED_SPEAKERS);
        let requested_min = config.min_speakers.unwrap_or(1);
        let min_speakers = requested_min.clamp(1, max_speakers);

        let clustering = cluster_embeddings(&embeddings, min_speakers, max_speakers);
        let labels = clustering.labels;
        let confidences = clustering.confidences;

        let mut cluster_first_start: HashMap<usize, f32> = HashMap::new();
        for (region_idx, (start_idx, _)) in regions.iter().enumerate() {
            let label = labels[region_idx];
            let start_secs = frames[*start_idx].start_sample as f32 / TARGET_SAMPLE_RATE as f32;
            cluster_first_start
                .entry(label)
                .and_modify(|existing| {
                    if start_secs < *existing {
                        *existing = start_secs;
                    }
                })
                .or_insert(start_secs);
        }

        let mut cluster_order: Vec<(usize, f32)> = cluster_first_start.into_iter().collect();
        cluster_order.sort_by(|a, b| a.1.total_cmp(&b.1));
        let cluster_to_speaker_index: HashMap<usize, usize> = cluster_order
            .iter()
            .enumerate()
            .map(|(speaker_idx, (cluster, _))| (*cluster, speaker_idx))
            .collect();

        let mut segments = Vec::with_capacity(regions.len());
        for (region_idx, (start_idx, end_idx)) in regions.iter().enumerate() {
            let label = labels[region_idx];
            let speaker_idx = cluster_to_speaker_index.get(&label).copied().unwrap_or(0);
            let start_secs = frames[*start_idx].start_sample as f32 / TARGET_SAMPLE_RATE as f32;
            let mut end_secs = frames[*end_idx].end_sample as f32 / TARGET_SAMPLE_RATE as f32;
            end_secs = end_secs.min(duration_secs);
            if end_secs <= start_secs {
                continue;
            }

            segments.push(DiarizationSegment {
                speaker: format!("SPEAKER_{speaker_idx:02}"),
                start_secs,
                end_secs,
                confidence: confidences.get(region_idx).copied().flatten(),
            });
        }

        merge_adjacent_segments(&mut segments, (min_silence_frames as f32 * HOP_MS) / 1000.0);

        let mut distinct_speakers: HashMap<String, ()> = HashMap::new();
        for segment in &segments {
            distinct_speakers.insert(segment.speaker.clone(), ());
        }

        let speaker_count = distinct_speakers.len();
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

fn extract_frame_features(
    samples: &[f32],
    frame_len: usize,
    hop_len: usize,
) -> Result<Vec<FrameFeatures>> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let hann: Vec<f32> = (0..FFT_SIZE)
        .map(|i| {
            let phase = (2.0 * std::f32::consts::PI * i as f32) / (FFT_SIZE as f32 - 1.0);
            0.5 - 0.5 * phase.cos()
        })
        .collect();

    let mut out = Vec::new();
    let mut start = 0usize;
    while start < samples.len() {
        let end = (start + frame_len).min(samples.len());
        let frame = &samples[start..end];
        if frame.is_empty() {
            break;
        }

        let rms = (frame.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>()
            / frame.len() as f64)
            .sqrt() as f32;
        let rms_db = 20.0 * (rms.max(1e-7)).log10();

        let mut zc = 0usize;
        for pair in frame.windows(2) {
            if (pair[0] >= 0.0 && pair[1] < 0.0) || (pair[0] < 0.0 && pair[1] >= 0.0) {
                zc += 1;
            }
        }
        let zcr = zc as f32 / frame.len() as f32;

        let mut fft_in = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
        for i in 0..FFT_SIZE {
            let sample = frame.get(i).copied().unwrap_or(0.0);
            fft_in[i] = Complex32::new(sample * hann[i], 0.0);
        }
        fft.process(&mut fft_in);

        let mut total_power = 0.0f32;
        let mut low_power = 0.0f32;
        let mut mid_power = 0.0f32;
        let mut high_power = 0.0f32;
        let mut centroid_num = 0.0f32;

        for (bin, c) in fft_in.iter().take(FFT_SIZE / 2).enumerate() {
            let freq_hz = bin as f32 * TARGET_SAMPLE_RATE as f32 / FFT_SIZE as f32;
            let power = (c.re * c.re + c.im * c.im).max(0.0);
            total_power += power;
            centroid_num += freq_hz * power;

            if freq_hz <= 300.0 {
                low_power += power;
            } else if freq_hz <= 1200.0 {
                mid_power += power;
            } else if freq_hz <= 4000.0 {
                high_power += power;
            }
        }

        let total_power = total_power.max(1e-9);
        let centroid_norm = (centroid_num / total_power) / (TARGET_SAMPLE_RATE as f32 / 2.0);
        let low_ratio = low_power / total_power;
        let mid_ratio = mid_power / total_power;
        let high_ratio = high_power / total_power;

        let vector = vec![
            (rms_db + 100.0) / 100.0,
            zcr,
            centroid_norm,
            low_ratio,
            mid_ratio,
            high_ratio,
        ];

        out.push(FrameFeatures {
            start_sample: start,
            end_sample: end,
            rms_db,
            vector,
        });

        if end >= samples.len() {
            break;
        }
        start = start.saturating_add(hop_len);
    }

    Ok(out)
}

fn adaptive_vad_threshold_db(values: &[f32]) -> f32 {
    if values.is_empty() {
        return -45.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let p35 = percentile_sorted(&sorted, 0.35);
    let p90 = percentile_sorted(&sorted, 0.90);
    (p35 + (p90 - p35) * 0.18).clamp(-55.0, -18.0)
}

fn percentile_sorted(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let clamped = q.clamp(0.0, 1.0);
    let idx = ((sorted.len().saturating_sub(1)) as f32 * clamped).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn smooth_activity_mask(active: &mut [bool], min_speech_frames: usize, min_silence_frames: usize) {
    if active.is_empty() {
        return;
    }

    // Fill short silent gaps between speech regions.
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

    // Remove very short speech bursts.
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

#[derive(Debug, Clone)]
struct ClusteringResult {
    labels: Vec<usize>,
    confidences: Vec<Option<f32>>,
}

fn cluster_embeddings(
    embeddings: &[Vec<f32>],
    min_speakers: usize,
    max_speakers: usize,
) -> ClusteringResult {
    if embeddings.is_empty() {
        return ClusteringResult {
            labels: Vec::new(),
            confidences: Vec::new(),
        };
    }
    if embeddings.len() == 1 {
        return ClusteringResult {
            labels: vec![0],
            confidences: vec![Some(1.0)],
        };
    }

    let max_k = max_speakers.min(embeddings.len()).max(1);
    let min_k = min_speakers.min(max_k).max(1);

    let mut best_labels = vec![0usize; embeddings.len()];
    let mut best_centroids = vec![mean_vector(embeddings)];
    let mut best_score = f32::NEG_INFINITY;

    for k in min_k..=max_k {
        let (labels, centroids) = kmeans(embeddings, k, 40);
        let score =
            silhouette_like_score(embeddings, &labels, &centroids) - 0.03 * (k as f32 - 1.0);
        if score > best_score {
            best_score = score;
            best_labels = labels;
            best_centroids = centroids;
        }
    }

    let confidences = assignment_confidence(embeddings, &best_labels, &best_centroids);
    ClusteringResult {
        labels: best_labels,
        confidences,
    }
}

fn kmeans(embeddings: &[Vec<f32>], k: usize, max_iter: usize) -> (Vec<usize>, Vec<Vec<f32>>) {
    let mut centroids = init_centroids_farthest(embeddings, k);
    let mut labels = vec![0usize; embeddings.len()];

    for _ in 0..max_iter {
        let mut changed = false;
        for (idx, emb) in embeddings.iter().enumerate() {
            let mut best_label = 0usize;
            let mut best_dist = f32::INFINITY;
            for (cluster_idx, centroid) in centroids.iter().enumerate() {
                let dist = squared_l2(emb, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_label = cluster_idx;
                }
            }
            if labels[idx] != best_label {
                labels[idx] = best_label;
                changed = true;
            }
        }

        let mut next_centroids = vec![vec![0.0f32; embeddings[0].len()]; k];
        let mut counts = vec![0usize; k];
        for (emb, label) in embeddings.iter().zip(labels.iter().copied()) {
            counts[label] += 1;
            for (dim, value) in emb.iter().enumerate() {
                next_centroids[label][dim] += *value;
            }
        }
        for cluster_idx in 0..k {
            if counts[cluster_idx] == 0 {
                next_centroids[cluster_idx] = centroids[cluster_idx].clone();
                continue;
            }
            for dim in 0..next_centroids[cluster_idx].len() {
                next_centroids[cluster_idx][dim] /= counts[cluster_idx] as f32;
            }
        }

        centroids = next_centroids;
        if !changed {
            break;
        }
    }

    (labels, centroids)
}

fn init_centroids_farthest(embeddings: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let mut centroids = Vec::with_capacity(k);
    centroids.push(embeddings[0].clone());

    while centroids.len() < k {
        let mut farthest_idx = 0usize;
        let mut farthest_dist = f32::NEG_INFINITY;

        for (idx, emb) in embeddings.iter().enumerate() {
            let nearest = centroids
                .iter()
                .map(|c| squared_l2(emb, c))
                .fold(f32::INFINITY, f32::min);
            if nearest > farthest_dist {
                farthest_dist = nearest;
                farthest_idx = idx;
            }
        }
        centroids.push(embeddings[farthest_idx].clone());
    }

    centroids
}

fn mean_vector(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0f32; vectors[0].len()];
    for vector in vectors {
        for (idx, value) in vector.iter().enumerate() {
            out[idx] += *value;
        }
    }
    for value in &mut out {
        *value /= vectors.len() as f32;
    }
    out
}

fn silhouette_like_score(embeddings: &[Vec<f32>], labels: &[usize], centroids: &[Vec<f32>]) -> f32 {
    if embeddings.is_empty() || centroids.is_empty() {
        return 0.0;
    }
    if centroids.len() == 1 {
        return 0.0;
    }

    let mut score_sum = 0.0f32;
    for (emb, label) in embeddings.iter().zip(labels.iter().copied()) {
        let a = squared_l2(emb, &centroids[label]).sqrt();
        let mut b = f32::INFINITY;
        for (cluster_idx, centroid) in centroids.iter().enumerate() {
            if cluster_idx == label {
                continue;
            }
            b = b.min(squared_l2(emb, centroid).sqrt());
        }
        let denom = a.max(b).max(1e-6);
        score_sum += (b - a) / denom;
    }
    score_sum / embeddings.len() as f32
}

fn assignment_confidence(
    embeddings: &[Vec<f32>],
    labels: &[usize],
    centroids: &[Vec<f32>],
) -> Vec<Option<f32>> {
    if centroids.is_empty() {
        return vec![None; embeddings.len()];
    }
    if centroids.len() == 1 {
        return vec![Some(1.0); embeddings.len()];
    }

    let mut out = Vec::with_capacity(embeddings.len());
    for (emb, label) in embeddings.iter().zip(labels.iter().copied()) {
        let own = squared_l2(emb, &centroids[label]).sqrt();
        let mut other = f32::INFINITY;
        for (cluster_idx, centroid) in centroids.iter().enumerate() {
            if cluster_idx == label {
                continue;
            }
            other = other.min(squared_l2(emb, centroid).sqrt());
        }
        let conf = ((other - own) / (other + own + 1e-6)).clamp(0.0, 1.0);
        out.push(Some(conf));
    }
    out
}

fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = *x - *y;
            d * d
        })
        .sum()
}

fn merge_adjacent_segments(segments: &mut Vec<DiarizationSegment>, merge_gap_secs: f32) {
    if segments.len() <= 1 {
        return;
    }

    let mut merged = Vec::with_capacity(segments.len());
    let mut current = segments[0].clone();
    for segment in segments.iter().skip(1) {
        let gap = (segment.start_secs - current.end_secs).max(0.0);
        if segment.speaker == current.speaker && gap <= merge_gap_secs {
            current.end_secs = current.end_secs.max(segment.end_secs);
            current.confidence = match (current.confidence, segment.confidence) {
                (Some(a), Some(b)) => Some((a + b) / 2.0),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };
        } else {
            merged.push(current);
            current = segment.clone();
        }
    }
    merged.push(current);
    *segments = merged;
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
