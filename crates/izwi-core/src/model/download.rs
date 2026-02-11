//! Model downloading from HuggingFace Hub
//!
//! Features:
//! - Async streaming downloads with byte-level progress
//! - Multi-progress bar terminal display
//! - Non-blocking downloads with background task spawning
//! - Real-time progress via channels

use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest;
use tokio::sync::{broadcast, RwLock};
use tokio::task::JoinHandle;
use tokio::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::error::{Error, Result};
use crate::model::info::ModelVariant;

const HF_BASE_URL: &str = "https://huggingface.co";
const CHUNK_SIZE: usize = 8192; // 8KB chunks for streaming

#[derive(Debug, Clone)]
struct FileDownloadPlan {
    file: String,
    expected_size: u64,
    strict_size_check: bool,
}

/// Progress update for model downloads
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub variant: ModelVariant,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub current_file: String,
    pub current_file_downloaded: u64,
    pub current_file_total: u64,
    pub files_completed: usize,
    pub files_total: usize,
}

impl DownloadProgress {
    /// Overall download percentage (0-100)
    pub fn total_percent(&self) -> f32 {
        if self.total_bytes > 0 {
            (self.downloaded_bytes as f32 / self.total_bytes as f32) * 100.0
        } else {
            0.0
        }
    }

    /// Current file percentage (0-100)
    pub fn file_percent(&self) -> f32 {
        if self.current_file_total > 0 {
            (self.current_file_downloaded as f32 / self.current_file_total as f32) * 100.0
        } else {
            0.0
        }
    }
}

/// Download state for tracking model downloads
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownloadState {
    NotDownloaded,
    Downloading,
    Downloaded,
    Error,
}

use serde::{Deserialize, Serialize};

/// Active download tracking
#[derive(Debug)]
struct ActiveDownload {
    handle: JoinHandle<Result<PathBuf>>,
    progress_tx: broadcast::Sender<DownloadProgress>,
}

impl ActiveDownload {
    /// Subscribe to progress updates
    fn subscribe(&self) -> broadcast::Receiver<DownloadProgress> {
        self.progress_tx.subscribe()
    }
}

/// Shared download state across threads
#[derive(Debug, Clone, Default)]
pub struct DownloadStateManager {
    state: Arc<RwLock<std::collections::HashMap<ModelVariant, DownloadState>>>,
}

impl DownloadStateManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn set_state(&self, variant: ModelVariant, state: DownloadState) {
        let mut map = self.state.write().await;
        map.insert(variant, state);
    }

    pub async fn get_state(&self, variant: ModelVariant) -> DownloadState {
        let map = self.state.read().await;
        map.get(&variant)
            .copied()
            .unwrap_or(DownloadState::NotDownloaded)
    }

    pub async fn is_downloading(&self, variant: ModelVariant) -> bool {
        self.get_state(variant).await == DownloadState::Downloading
    }
}

/// Model downloader for HuggingFace Hub
pub struct ModelDownloader {
    pub models_dir: PathBuf,
    http_client: reqwest::Client,
    active_downloads: Arc<RwLock<std::collections::HashMap<ModelVariant, ActiveDownload>>>,
    multi_progress: MultiProgress,
    state_manager: DownloadStateManager,
}

impl ModelDownloader {
    /// Create a new downloader
    pub fn new(models_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&models_dir)?;

        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600))
            // Avoid macOS SystemConfiguration proxy lookups which can fail in
            // restricted runtime environments (sandboxed agents/CI).
            .no_proxy()
            .build()
            .map_err(|e| Error::HfHubError(format!("Failed to create HTTP client: {}", e)))?;

        let multi_progress = MultiProgress::new();
        multi_progress.set_draw_target(indicatif::ProgressDrawTarget::stderr_with_hz(10));

        Ok(Self {
            models_dir,
            http_client,
            active_downloads: Arc::new(RwLock::new(std::collections::HashMap::new())),
            multi_progress,
            state_manager: DownloadStateManager::new(),
        })
    }

    /// Get download state manager
    pub fn state_manager(&self) -> DownloadStateManager {
        self.state_manager.clone()
    }

    /// Download a file with streaming and progress bar
    async fn download_file_streaming(
        &self,
        repo_id: &str,
        filename: &str,
        dest: &Path,
        file_pb: Option<ProgressBar>,
        progress_tx: Option<broadcast::Sender<DownloadProgress>>,
        progress_template: Option<DownloadProgress>,
    ) -> Result<u64> {
        let url = format!("{}/{}/resolve/main/{}", HF_BASE_URL, repo_id, filename);
        debug!("Downloading from URL: {}", url);

        let response = self
            .http_client
            .get(&url)
            .header("User-Agent", "izwi-audio/0.1.0")
            .send()
            .await
            .map_err(|e| Error::HfHubError(format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::HfHubError(format!(
                "HTTP {} for {}",
                response.status(),
                url
            )));
        }

        // Get content length for progress bar
        let total_size = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);

        if let Some(ref pb) = file_pb {
            pb.set_length(total_size);
        }

        // Create parent directories
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Stream download to file
        let mut file = tokio::fs::File::create(dest).await?;
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();
        let mut last_progress_emit = Instant::now();
        let mut last_progress_bytes = 0u64;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| Error::HfHubError(format!("Stream error: {}", e)))?;
            tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await?;
            downloaded += chunk.len() as u64;

            if let Some(ref pb) = file_pb {
                pb.set_position(downloaded);
            }

            // Send real-time progress update
            if let Some(ref tx) = progress_tx {
                if let Some(ref template) = progress_template {
                    let bytes_delta = downloaded.saturating_sub(last_progress_bytes);
                    let should_emit =
                        bytes_delta >= 256 * 1024
                            || last_progress_emit.elapsed() >= Duration::from_millis(200)
                            || (total_size > 0 && downloaded >= total_size);

                    if should_emit {
                        let progress = DownloadProgress {
                            current_file_downloaded: downloaded,
                            current_file_total: total_size.max(downloaded),
                            // Calculate total downloaded: base bytes from completed files + current file progress
                            downloaded_bytes: template.downloaded_bytes + downloaded,
                            ..template.clone()
                        };
                        let _ = tx.send(progress);
                        last_progress_emit = Instant::now();
                        last_progress_bytes = downloaded;
                    }
                }
            }
        }

        // Ensure we always emit a final per-file progress event, even if throttled updates
        // did not include the final chunk.
        if let Some(ref tx) = progress_tx {
            if let Some(ref template) = progress_template {
                let progress = DownloadProgress {
                    current_file_downloaded: downloaded,
                    current_file_total: total_size.max(downloaded),
                    downloaded_bytes: template.downloaded_bytes + downloaded,
                    ..template.clone()
                };
                let _ = tx.send(progress);
            }
        }

        // Sync file to disk
        file.sync_all().await?;

        debug!("Downloaded {} bytes to {:?}", downloaded, dest);
        Ok(downloaded)
    }

    /// Get the local path for a model variant
    pub fn model_path(&self, variant: ModelVariant) -> PathBuf {
        self.models_dir.join(variant.dir_name())
    }

    /// Check if a model is already downloaded
    pub fn is_downloaded(&self, variant: ModelVariant) -> bool {
        let path = self.model_path(variant);
        if !path.exists() {
            return false;
        }

        // Check for essential files based on model type
        if variant.is_lfm2() {
            // LFM2-Audio requires model.safetensors, config.json, and tokenizer files
            let has_model = path.join("model.safetensors").exists();
            let has_config = path.join("config.json").exists();
            let has_tokenizer = path.join("tokenizer.json").exists();
            return has_model && has_config && has_tokenizer;
        }

        if variant.is_asr() {
            // Qwen3-ASR requires config.json, vocab.json, chat_template.json, and model weights
            let has_config = path.join("config.json").exists();
            let has_vocab = path.join("vocab.json").exists();
            let has_chat_template = path.join("chat_template.json").exists();
            // mlx-community quantized models use model.safetensors + model.safetensors.index.json
            // Original models: 0.6B has single file, 1.7B has sharded weights
            let has_model = if variant.is_quantized() {
                path.join("model.safetensors").exists()
                    && path.join("model.safetensors.index.json").exists()
            } else if matches!(variant, ModelVariant::Qwen3Asr06B) {
                path.join("model.safetensors").exists()
            } else {
                path.join("model-00001-of-00002.safetensors").exists()
            };
            return has_config && has_vocab && has_chat_template && has_model;
        }

        if variant.is_chat() {
            let has_config = path.join("config.json").exists();
            let has_tokenizer = path.join("tokenizer.json").exists()
                || (path.join("vocab.json").exists() && path.join("merges.txt").exists());
            // Some exported variants include index.json, some don't.
            let has_model = path.join("model.safetensors").exists()
                || path.join("model-00001-of-00002.safetensors").exists();
            return has_config && has_tokenizer && has_model;
        }

        if variant.is_forced_aligner() {
            // ForcedAligner has similar structure to ASR
            let has_config = path.join("config.json").exists();
            let has_vocab = path.join("vocab.json").exists();
            let has_model = path.join("model.safetensors").exists();
            return has_config && has_vocab && has_model;
        }

        if variant.is_voxtral() {
            // Voxtral Mini 4B Realtime uses params.json, tekken.json, and consolidated.safetensors
            let has_params = path.join("params.json").exists();
            let has_tekken = path.join("tekken.json").exists();
            let has_model = path.join("consolidated.safetensors").exists();
            return has_params && has_tekken && has_model;
        }

        if variant.is_tokenizer() {
            path.join("tokenizer.json").exists() || path.join("vocab.json").exists()
        } else {
            // Check for model weights
            let has_safetensors = std::fs::read_dir(&path)
                .map(|entries| {
                    entries.filter_map(|e| e.ok()).any(|e| {
                        e.path()
                            .extension()
                            .map(|ext| ext == "safetensors")
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false);

            let has_config = path.join("config.json").exists();
            has_safetensors && has_config
        }
    }

    /// Download a model with multi-progress bar display
    pub async fn download(&self, variant: ModelVariant) -> Result<PathBuf> {
        let repo_id = variant.repo_id();
        let local_dir = self.model_path(variant);

        tokio::fs::create_dir_all(&local_dir).await?;
        info!("Downloading {} to {:?}", repo_id, local_dir);

        // Create overall progress bar
        let files = self.get_model_files(variant);
        let file_sizes = self.get_file_sizes(variant);
        let total_bytes: u64 = file_sizes.iter().sum();

        let overall_pb = self.multi_progress.add(ProgressBar::new(total_bytes));
        overall_pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("##-"),
        );
        overall_pb.set_message(format!("{}", variant.display_name()));

        let mut downloaded_bytes: u64 = 0;

        for (idx, file) in files.iter().enumerate() {
            let dest = local_dir.join(file);

            // Skip if already downloaded
            if dest.exists() {
                debug!("File already exists: {:?}", dest);
                let file_size = tokio::fs::metadata(&dest).await?.len();
                downloaded_bytes += file_size;
                overall_pb.set_position(downloaded_bytes);
                continue;
            }

            // Create file progress bar
            let file_size = file_sizes.get(idx).copied().unwrap_or(0);
            let file_pb = self.multi_progress.add(ProgressBar::new(file_size));
            file_pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "  [{elapsed_precise}] {bar:30.green/white} {bytes}/{total_bytes} {msg}",
                    )
                    .unwrap()
                    .progress_chars("=>-"),
            );
            file_pb.set_message(file.clone());

            // Stream download with progress
            match self
                .download_file_streaming(repo_id, file, &dest, Some(file_pb.clone()), None, None)
                .await
            {
                Ok(bytes_downloaded) => {
                    debug!(
                        "Downloaded: {} -> {:?} ({} bytes)",
                        file, dest, bytes_downloaded
                    );
                    downloaded_bytes += bytes_downloaded;
                    overall_pb.set_position(downloaded_bytes);
                    file_pb.finish_with_message(format!("{} ✓", file));
                }
                Err(e) => {
                    warn!("Failed to download {}: {}", file, e);
                    file_pb.finish_with_message(format!("{} ✗", file));
                    // Some files might be optional, continue
                }
            }

            self.multi_progress.remove(&file_pb);
        }

        overall_pb.finish_with_message(format!("{} ✓", variant.display_name()));
        self.multi_progress.remove(&overall_pb);

        info!("Model downloaded to {:?}", local_dir);
        Ok(local_dir)
    }

    /// Spawn a non-blocking download in the background
    pub async fn spawn_download(
        &self,
        variant: ModelVariant,
    ) -> Result<broadcast::Receiver<DownloadProgress>> {
        // Set initial state
        self.state_manager
            .set_state(variant, DownloadState::Downloading)
            .await;

        // Create broadcast channel for progress (allows multiple subscribers)
        let (progress_tx, _progress_rx) = broadcast::channel(512);
        let downloader = self.clone_downloader();
        let progress_tx_clone = progress_tx.clone();

        // Spawn background download task
        let handle = tokio::spawn(async move {
            let result = downloader
                .download_with_progress(variant, progress_tx_clone)
                .await;

            // Update state based on result
            let final_state = match &result {
                Ok(_) => DownloadState::Downloaded,
                Err(_) => DownloadState::Error,
            };
            downloader
                .state_manager
                .set_state(variant, final_state)
                .await;

            // Remove finished task from active-downloads registry so UI/handlers
            // stop treating completed downloads as active.
            {
                let mut downloads = downloader.active_downloads.write().await;
                downloads.remove(&variant);
            }

            result
        });

        // Store active download
        let active = ActiveDownload {
            handle,
            progress_tx: progress_tx.clone(),
        };

        let mut downloads = self.active_downloads.write().await;
        downloads.insert(variant, active);

        // Return the progress receiver for the caller to use
        Ok(progress_tx.subscribe())
    }

    /// Clone the downloader for use in spawned tasks
    fn clone_downloader(&self) -> Self {
        Self {
            models_dir: self.models_dir.clone(),
            http_client: self.http_client.clone(),
            active_downloads: Arc::clone(&self.active_downloads),
            multi_progress: MultiProgress::new(), // Each spawned task gets its own multi-progress
            state_manager: self.state_manager.clone(),
        }
    }

    /// Check if a download is active
    pub async fn is_download_active(&self, variant: ModelVariant) -> bool {
        let downloads = self.active_downloads.read().await;
        downloads.contains_key(&variant)
    }

    /// Wait for an active download to complete
    pub async fn wait_for_download(&self, variant: ModelVariant) -> Result<Option<PathBuf>> {
        let handle = {
            let mut downloads = self.active_downloads.write().await;
            downloads.remove(&variant).map(|d| d.handle)
        };

        match handle {
            Some(h) => {
                let result = h.await.map_err(|e| Error::DownloadError(e.to_string()))?;
                Some(result).transpose()
            }
            None => Ok(None),
        }
    }

    /// Download model with real-time byte-level progress
    pub async fn download_with_progress(
        &self,
        variant: ModelVariant,
        progress_tx: broadcast::Sender<DownloadProgress>,
    ) -> Result<PathBuf> {
        let repo_id = variant.repo_id();
        let local_dir = self.model_path(variant);

        tokio::fs::create_dir_all(&local_dir).await?;

        info!("Downloading {} to {:?}", repo_id, local_dir);

        let file_plans = self.get_file_download_plans(variant).await;
        let total_files = file_plans.len();
        let total_bytes: u64 = file_plans.iter().map(|plan| plan.expected_size).sum();
        let mut downloaded_bytes: u64 = 0;

        for (idx, plan) in file_plans.iter().enumerate() {
            let file = &plan.file;
            let file_size = plan.expected_size;
            let dest = local_dir.join(file);

            // Skip if already downloaded
            if dest.exists() {
                let actual_size = tokio::fs::metadata(&dest).await?.len();
                let should_redownload =
                    plan.strict_size_check && file_size > 0 && actual_size != file_size;

                if should_redownload {
                    warn!(
                        "Existing file size mismatch for {} (expected {} bytes, found {}), re-downloading",
                        file, file_size, actual_size
                    );
                    tokio::fs::remove_file(&dest).await?;
                } else {
                    debug!("File already exists: {:?} ({} bytes)", dest, actual_size);
                    downloaded_bytes += if file_size > 0 {
                        file_size
                    } else {
                        actual_size
                    };

                    // Send progress update
                    let progress = DownloadProgress {
                        variant,
                        downloaded_bytes,
                        total_bytes,
                        current_file: file.clone(),
                        current_file_downloaded: actual_size,
                        current_file_total: if file_size > 0 {
                            file_size
                        } else {
                            actual_size
                        },
                        files_completed: idx + 1,
                        files_total: total_files,
                    };
                    let _ = progress_tx.send(progress);
                    continue;
                }
            }

            // Stream download with per-chunk progress updates
            let file_pb = self.multi_progress.add(ProgressBar::new(file_size));
            file_pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "  [{elapsed_precise}] {bar:30.green/white} {bytes}/{total_bytes} {msg}",
                    )
                    .unwrap()
                    .progress_chars("=>-"),
            );
            file_pb.set_message(file.clone());

            // Create progress template for real-time updates
            let progress_template = DownloadProgress {
                variant,
                downloaded_bytes, // Base bytes from completed files
                total_bytes,
                current_file: file.clone(),
                current_file_downloaded: 0,
                current_file_total: file_size,
                files_completed: idx,
                files_total: total_files,
            };

            match self
                .download_file_streaming(
                    repo_id,
                    file,
                    &dest,
                    Some(file_pb.clone()),
                    Some(progress_tx.clone()),
                    Some(progress_template),
                )
                .await
            {
                Ok(bytes_downloaded) => {
                    if plan.strict_size_check && file_size > 0 && bytes_downloaded != file_size {
                        let _ = tokio::fs::remove_file(&dest).await;
                        return Err(Error::DownloadError(format!(
                            "Downloaded size mismatch for {}: expected {} bytes, got {} bytes",
                            file, file_size, bytes_downloaded
                        )));
                    }

                    downloaded_bytes += bytes_downloaded;

                    let progress = DownloadProgress {
                        variant,
                        downloaded_bytes,
                        total_bytes,
                        current_file: file.clone(),
                        current_file_downloaded: bytes_downloaded,
                        current_file_total: file_size,
                        files_completed: idx + 1,
                        files_total: total_files,
                    };
                    let _ = progress_tx.send(progress);

                    file_pb.finish_with_message(format!("{} ✓", file));
                }
                Err(e) => {
                    warn!("Failed to download {}: {}", file, e);
                    file_pb.finish_with_message(format!("{} ✗", file));
                    let _ = tokio::fs::remove_file(&dest).await;
                    return Err(Error::DownloadError(format!(
                        "Failed to download required file {}: {}",
                        file, e
                    )));
                }
            }

            self.multi_progress.remove(&file_pb);
        }

        // Send final completion progress
        let progress = DownloadProgress {
            variant,
            downloaded_bytes,
            total_bytes,
            current_file: String::new(),
            current_file_downloaded: 0,
            current_file_total: 0,
            files_completed: total_files,
            files_total: total_files,
        };
        let _ = progress_tx.send(progress);

        info!("Model downloaded to {:?}", local_dir);
        Ok(local_dir)
    }

    /// Get list of files to download for a model variant
    /// Based on actual repo structure on HuggingFace
    fn get_model_files(&self, variant: ModelVariant) -> Vec<String> {
        // LFM2-Audio has a different file structure
        if variant.is_lfm2() {
            return vec![
                "config.json".to_string(),
                "model.safetensors".to_string(),
                "tokenizer.json".to_string(),
                "tokenizer_config.json".to_string(),
                "special_tokens_map.json".to_string(),
                "tokenizer-e351c8d8-checkpoint125.safetensors".to_string(),
                "chat_template.jinja".to_string(),
            ];
        }

        // Qwen3-ASR models
        if variant.is_asr() {
            let mut files = vec![
                "config.json".to_string(),
                "chat_template.json".to_string(),
                "generation_config.json".to_string(),
                "merges.txt".to_string(),
                "preprocessor_config.json".to_string(),
                "tokenizer_config.json".to_string(),
                "vocab.json".to_string(),
            ];
            // mlx-community quantized models use model.safetensors + model.safetensors.index.json
            // Original models: 0.6B has single file, 1.7B has sharded weights
            if variant.is_quantized() {
                files.push("model.safetensors".to_string());
                files.push("model.safetensors.index.json".to_string());
            } else if matches!(variant, ModelVariant::Qwen3Asr06B) {
                files.push("model.safetensors".to_string());
            } else {
                files.extend([
                    "model-00001-of-00002.safetensors".to_string(),
                    "model-00002-of-00002.safetensors".to_string(),
                    "model.safetensors.index.json".to_string(),
                ]);
            }
            return files;
        }

        if variant.is_chat() {
            return vec![
                "config.json".to_string(),
                "generation_config.json".to_string(),
                "chat_template.jinja".to_string(),
                "tokenizer.json".to_string(),
                "tokenizer_config.json".to_string(),
                "special_tokens_map.json".to_string(),
                "vocab.json".to_string(),
                "merges.txt".to_string(),
                "model.safetensors".to_string(),
                "model.safetensors.index.json".to_string(),
            ];
        }

        // Qwen3-ForcedAligner model
        if variant.is_forced_aligner() {
            return vec![
                "config.json".to_string(),
                "generation_config.json".to_string(),
                "merges.txt".to_string(),
                "model.safetensors".to_string(),
                "preprocessor_config.json".to_string(),
                "tokenizer_config.json".to_string(),
                "vocab.json".to_string(),
            ];
        }

        // Voxtral Mini 4B Realtime model - single consolidated safetensors file
        if variant.is_voxtral() {
            return vec![
                "params.json".to_string(),
                "tekken.json".to_string(),
                "consolidated.safetensors".to_string(),
            ];
        }

        let mut files = vec![
            "config.json".to_string(),
            "generation_config.json".to_string(),
        ];

        if variant.is_tokenizer() {
            // Tokenizer model files (Qwen3-TTS-Tokenizer-12Hz)
            files.extend([
                "preprocessor_config.json".to_string(),
                "model.safetensors".to_string(),
            ]);
        } else {
            // TTS model files - Qwen3-TTS uses vocab.json + merges.txt, not tokenizer.json
            files.extend([
                "tokenizer_config.json".to_string(),
                "vocab.json".to_string(),
                "merges.txt".to_string(),
                "preprocessor_config.json".to_string(),
            ]);
            // All Qwen3-TTS models use single model.safetensors file
            files.push("model.safetensors".to_string());
            // mlx-community quantized TTS models also have model.safetensors.index.json
            if variant.is_quantized() {
                files.push("model.safetensors.index.json".to_string());
            }
            // Speech tokenizer files (audio codec for decoding)
            files.extend([
                "speech_tokenizer/config.json".to_string(),
                "speech_tokenizer/configuration.json".to_string(),
                "speech_tokenizer/model.safetensors".to_string(),
                "speech_tokenizer/preprocessor_config.json".to_string(),
            ]);
        }

        files
    }

    /// Get actual file size from HTTP HEAD request
    async fn get_actual_file_size(&self, repo_id: &str, filename: &str) -> Result<u64> {
        let url = format!("{}/{}/resolve/main/{}", HF_BASE_URL, repo_id, filename);

        let response = self
            .http_client
            .head(&url)
            .header("User-Agent", "izwi-audio/0.1.0")
            .send()
            .await
            .map_err(|e| Error::HfHubError(format!("HEAD request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::HfHubError(format!(
                "HTTP {} for HEAD {}",
                response.status(),
                url
            )));
        }

        let size = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);

        Ok(size)
    }

    /// Get actual file sizes for all model files
    async fn get_file_download_plans(&self, variant: ModelVariant) -> Vec<FileDownloadPlan> {
        let files = self.get_model_files(variant);
        let repo_id = variant.repo_id();
        let local_dir = self.model_path(variant);

        let mut plans = Vec::with_capacity(files.len());

        for file in &files {
            let dest = local_dir.join(file);

            // Prefer remote file sizes for accurate total bytes and integrity checks.
            match self.get_actual_file_size(&repo_id, file).await {
                Ok(size) => plans.push(FileDownloadPlan {
                    file: file.clone(),
                    expected_size: size,
                    strict_size_check: true,
                }),
                Err(_) => {
                    // Fall back to local size if present (resume semantics), else estimate.
                    let fallback_size = if dest.exists() {
                        tokio::fs::metadata(&dest)
                            .await
                            .map(|m| m.len())
                            .unwrap_or_else(|_| self.get_single_file_size_estimate(variant, file))
                    } else {
                        self.get_single_file_size_estimate(variant, file)
                    };

                    plans.push(FileDownloadPlan {
                        file: file.clone(),
                        expected_size: fallback_size,
                        strict_size_check: false,
                    });
                }
            }
        }

        plans
    }

    /// Get estimated size for a single file (fallback when HEAD fails)
    fn get_single_file_size_estimate(&self, variant: ModelVariant, file: &str) -> u64 {
        if file.contains("model.safetensors") && !file.contains("index") {
            if file.contains("00001") || file.contains("00002") {
                2_000_000_000
            } else {
                match variant {
                    ModelVariant::Qwen3Tts12Hz06BBase
                    | ModelVariant::Qwen3Tts12Hz06BCustomVoice => 1_800_000_000,
                    ModelVariant::Qwen3Tts12Hz17BBase
                    | ModelVariant::Qwen3Tts12Hz17BCustomVoice
                    | ModelVariant::Qwen3Tts12Hz17BVoiceDesign => 3_850_000_000,
                    ModelVariant::Qwen3Asr06B => 1_800_000_000,
                    ModelVariant::Qwen306B4Bit => 800_000_000,
                    ModelVariant::Lfm2Audio15B => 2_900_000_000,
                    ModelVariant::VoxtralMini4BRealtime2602 => 8_900_000_000,
                    _ => 1_500_000_000,
                }
            }
        } else if file.contains("tokenizer") && file.contains("safetensors") {
            300_000_000
        } else if file.contains("speech_tokenizer") && file.contains("safetensors") {
            100_000_000
        } else if file.ends_with(".json") || file.ends_with(".txt") || file.ends_with(".jinja") {
            if file == "tekken.json" {
                15_000_000
            } else {
                100_000
            }
        } else {
            10_000_000
        }
    }

    /// Get estimated file sizes for a model variant (in bytes)
    /// These are approximate sizes based on model architecture
    fn get_file_sizes(&self, variant: ModelVariant) -> Vec<u64> {
        let files = self.get_model_files(variant);
        files
            .iter()
            .map(|file| self.get_single_file_size_estimate(variant, file))
            .collect()
    }

    /// Get download size for a model (if available from cache)
    pub fn get_cached_size(&self, variant: ModelVariant) -> Option<u64> {
        let path = self.model_path(variant);
        if path.exists() {
            Self::dir_size(&path).ok()
        } else {
            None
        }
    }

    /// Subscribe to progress updates for an active download
    pub async fn subscribe_progress(
        &self,
        variant: ModelVariant,
    ) -> Result<broadcast::Receiver<DownloadProgress>> {
        let downloads = self.active_downloads.read().await;
        if let Some(active) = downloads.get(&variant) {
            Ok(active.subscribe())
        } else {
            Err(Error::DownloadError(format!(
                "No active download for {}",
                variant
            )))
        }
    }

    /// Cancel an active download
    pub async fn cancel_download(&self, variant: ModelVariant) -> Result<()> {
        let handle = {
            let mut downloads = self.active_downloads.write().await;
            downloads.remove(&variant).map(|d| d.handle)
        };

        if let Some(h) = handle {
            h.abort();
            // Clean up partial downloads
            let model_path = self.model_path(variant);
            if model_path.exists() {
                let _ = tokio::fs::remove_dir_all(&model_path).await;
            }
            // Update state
            self.state_manager
                .set_state(variant, DownloadState::NotDownloaded)
                .await;
            Ok(())
        } else {
            Err(Error::DownloadError(format!(
                "No active download for {}",
                variant
            )))
        }
    }

    /// Calculate directory size recursively
    fn dir_size(path: &Path) -> Result<u64> {
        let mut size = 0;
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_file() {
                size += metadata.len();
            } else if metadata.is_dir() {
                size += Self::dir_size(&entry.path())?;
            }
        }
        Ok(size)
    }
}
