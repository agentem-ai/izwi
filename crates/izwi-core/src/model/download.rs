//! Model downloading from HuggingFace Hub

use hf_hub::api::sync::Api;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::error::{Error, Result};
use crate::model::info::ModelVariant;

const HF_BASE_URL: &str = "https://huggingface.co";

/// Progress update for model downloads
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub variant: ModelVariant,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub progress_percent: f32,
    pub current_file: Option<String>,
    pub files_completed: usize,
    pub files_total: usize,
}

/// Model downloader for HuggingFace Hub
pub struct ModelDownloader {
    pub api: Api,
    pub models_dir: PathBuf,
    http_client: Client,
}

impl ModelDownloader {
    /// Create a new downloader
    pub fn new(models_dir: PathBuf) -> Result<Self> {
        // Ensure models directory exists
        std::fs::create_dir_all(&models_dir)?;

        let api = Api::new().map_err(|e| Error::HfHubError(e.to_string()))?;
        let http_client = Client::builder()
            .timeout(std::time::Duration::from_secs(3600)) // 1 hour timeout for large files
            .build()
            .map_err(|e| Error::HfHubError(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            api,
            models_dir,
            http_client,
        })
    }

    /// Download a file directly from HuggingFace using HTTP
    fn download_file_http(&self, repo_id: &str, filename: &str, dest: &Path) -> Result<()> {
        let url = format!("{}/{}/resolve/main/{}", HF_BASE_URL, repo_id, filename);
        debug!("Downloading from URL: {}", url);

        let response = self
            .http_client
            .get(&url)
            .header("User-Agent", "izwi-audio/0.1.0")
            .send()
            .map_err(|e| Error::HfHubError(format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::HfHubError(format!(
                "HTTP {} for {}",
                response.status(),
                url
            )));
        }

        // Create parent directories
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Write to file
        let bytes = response
            .bytes()
            .map_err(|e| Error::HfHubError(format!("Failed to read response: {}", e)))?;

        let mut file = File::create(dest)?;
        file.write_all(&bytes)?;

        debug!("Downloaded {} bytes to {:?}", bytes.len(), dest);
        Ok(())
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
            let has_model = path.join("model.safetensors").exists()
                || path.join("model-00001-of-00002.safetensors").exists();
            return has_config && has_vocab && has_chat_template && has_model;
        }

        if variant.is_forced_aligner() {
            // ForcedAligner has similar structure to ASR
            let has_config = path.join("config.json").exists();
            let has_vocab = path.join("vocab.json").exists();
            let has_model = path.join("model.safetensors").exists();
            return has_config && has_vocab && has_model;
        }

        if variant.is_voxtral() {
            // Voxtral has sharded safetensors and tokenizer files
            let has_config = path.join("config.json").exists();
            let has_tokenizer =
                path.join("tokenizer.json").exists() || path.join("tokenizer.model").exists();
            let has_model = path.join("model-00001-of-00004.safetensors").exists();
            return has_config && has_tokenizer && has_model;
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

    /// Download a model from HuggingFace Hub
    pub fn download(&self, variant: ModelVariant) -> Result<PathBuf> {
        let repo_id = variant.repo_id();
        let local_dir = self.model_path(variant);

        std::fs::create_dir_all(&local_dir)?;
        info!("Downloading {} to {:?}", repo_id, local_dir);

        // Create progress bar
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap(),
        );
        pb.set_message(format!("Downloading {}", variant.display_name()));

        // List and download all files
        let files = self.get_model_files(variant);

        for file in &files {
            pb.set_message(format!("Downloading: {}", file));
            debug!("Downloading file: {}", file);

            let dest = local_dir.join(file);

            // Skip if already downloaded
            if dest.exists() {
                debug!("File already exists: {:?}", dest);
                continue;
            }

            // Use direct HTTP download (more reliable than hf-hub for some repos)
            match self.download_file_http(repo_id, file, &dest) {
                Ok(()) => {
                    debug!("Downloaded: {} -> {:?}", file, dest);
                }
                Err(e) => {
                    warn!("Failed to download {}: {}", file, e);
                    // Some files might be optional, continue
                }
            }
        }

        pb.finish_with_message(format!("Downloaded {}", variant.display_name()));

        info!("Model downloaded to {:?}", local_dir);
        Ok(local_dir)
    }

    /// Download model with progress channel
    pub async fn download_with_progress(
        &self,
        variant: ModelVariant,
        progress_tx: mpsc::Sender<DownloadProgress>,
    ) -> Result<PathBuf> {
        let repo_id = variant.repo_id();
        let local_dir = self.model_path(variant);

        std::fs::create_dir_all(&local_dir)?;

        info!("Downloading {} to {:?}", repo_id, local_dir);

        let files = self.get_model_files(variant);
        let total_files = files.len();

        let file_sizes = self.get_file_sizes(variant);
        let total_bytes: u64 = file_sizes.iter().sum();
        let mut downloaded_bytes: u64 = 0;

        for (idx, file) in files.iter().enumerate() {
            let file_size = file_sizes.get(idx).copied().unwrap_or(0);
            let progress = DownloadProgress {
                variant,
                downloaded_bytes,
                total_bytes,
                progress_percent: if total_bytes > 0 {
                    (downloaded_bytes as f32 / total_bytes as f32) * 100.0
                } else {
                    (idx as f32 / total_files as f32) * 100.0
                },
                current_file: Some(file.clone()),
                files_completed: idx,
                files_total: total_files,
            };
            let _ = progress_tx.send(progress).await;

            let dest = local_dir.join(file);

            // Skip if already downloaded
            if dest.exists() {
                debug!("File already exists: {:?}", dest);
                downloaded_bytes += file_size;
                continue;
            }

            // Use direct HTTP download (more reliable than hf-hub for some repos)
            match self.download_file_http(repo_id, file, &dest) {
                Ok(()) => {
                    debug!("Downloaded: {} -> {:?}", file, dest);
                    downloaded_bytes += file_size;
                }
                Err(e) => {
                    warn!("Failed to download {}: {}", file, e);
                }
            }
        }

        // Send completion
        let progress = DownloadProgress {
            variant,
            downloaded_bytes: total_bytes,
            total_bytes,
            progress_percent: 100.0,
            current_file: None,
            files_completed: total_files,
            files_total: total_files,
        };
        let _ = progress_tx.send(progress).await;

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
            // 0.6B has single model file, 1.7B has sharded weights
            if matches!(variant, ModelVariant::Qwen3Asr06B) {
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
            // All Qwen3-TTS models (0.6B and 1.7B) use single model.safetensors file
            files.push("model.safetensors".to_string());
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

    /// Get estimated file sizes for a model variant (in bytes)
    /// These are approximate sizes based on model architecture
    fn get_file_sizes(&self, variant: ModelVariant) -> Vec<u64> {
        let files = self.get_model_files(variant);
        files
            .iter()
            .map(|file| {
                // Estimate file sizes based on filename patterns
                if file.contains("model.safetensors") && !file.contains("index") {
                    if file.contains("00001") || file.contains("00002") {
                        // Sharded model files ~2GB each
                        2_000_000_000
                    } else {
                        // Single model file - varies by model
                        match variant {
                            ModelVariant::Qwen3Tts12Hz06BBase
                            | ModelVariant::Qwen3Tts12Hz06BCustomVoice => 1_800_000_000,
                            ModelVariant::Qwen3Tts12Hz17BBase
                            | ModelVariant::Qwen3Tts12Hz17BCustomVoice
                            | ModelVariant::Qwen3Tts12Hz17BVoiceDesign => 3_850_000_000,
                            ModelVariant::Qwen3Asr06B => 1_800_000_000,
                            ModelVariant::Lfm2Audio15B => 2_900_000_000,
                            ModelVariant::VoxtralMini4BRealtime2602 => 8_900_000_000, // ~8.9GB single file
                            _ => 1_500_000_000,
                        }
                    }
                } else if file.contains("tokenizer") && file.contains("safetensors") {
                    // Tokenizer model ~300MB
                    300_000_000
                } else if file.contains("speech_tokenizer") && file.contains("safetensors") {
                    // Speech tokenizer ~100MB
                    100_000_000
                } else if file.ends_with(".json")
                    || file.ends_with(".txt")
                    || file.ends_with(".jinja")
                {
                    // Config/text files are small
                    100_000
                } else {
                    // Default for unknown files
                    10_000_000
                }
            })
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
