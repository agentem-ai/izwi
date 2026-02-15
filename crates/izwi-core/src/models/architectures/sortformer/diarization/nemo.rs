use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{Error, Result};
use crate::model::ModelVariant;

#[derive(Debug, Clone)]
pub struct SortformerArtifacts {
    pub nemo_path: PathBuf,
    pub extracted_dir: PathBuf,
    pub model_config_path: PathBuf,
    pub checkpoint_path: PathBuf,
}

pub fn ensure_sortformer_artifacts(
    model_dir: &Path,
    variant: ModelVariant,
) -> Result<SortformerArtifacts> {
    let nemo_filename = match variant {
        ModelVariant::DiarStreamingSortformer4SpkV21 => "diar_streaming_sortformer_4spk-v2.1.nemo",
        _ => {
            return Err(Error::InvalidInput(format!(
                "Unsupported Sortformer diarization variant: {}",
                variant.dir_name()
            )));
        }
    };

    let nemo_path = model_dir.join(nemo_filename);
    if !nemo_path.exists() {
        return Err(Error::ModelNotFound(format!(
            "Missing .nemo checkpoint for {} at {}",
            variant.dir_name(),
            nemo_path.display()
        )));
    }

    let extracted_dir = model_dir.join("sortformer-native");
    fs::create_dir_all(&extracted_dir).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to create Sortformer cache directory {}: {}",
            extracted_dir.display(),
            e
        ))
    })?;

    let model_config_path = extracted_dir.join("model_config.yaml");
    let checkpoint_path = extracted_dir.join("model_weights.ckpt");
    if !model_config_path.exists() || !checkpoint_path.exists() {
        let listing = tar_list(&nemo_path)?;
        // Validate checkpoint presence to fail fast on malformed archives.
        let checkpoint_entry = find_tar_entry(&listing, "model_weights.ckpt")?;
        let model_config_entry = find_tar_entry(&listing, "model_config.yaml")?;
        extract_tar_entry_to_file(&nemo_path, &model_config_entry, &model_config_path)?;
        extract_tar_entry_to_file(&nemo_path, &checkpoint_entry, &checkpoint_path)?;
    }

    Ok(SortformerArtifacts {
        nemo_path,
        extracted_dir,
        model_config_path,
        checkpoint_path,
    })
}

fn tar_list(nemo_path: &Path) -> Result<Vec<String>> {
    let output = Command::new("tar")
        .arg("-tf")
        .arg(nemo_path)
        .output()
        .map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed to list .nemo archive {}: {}",
                nemo_path.display(),
                e
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::ModelLoadError(format!(
            "Failed to list .nemo archive {}: {}",
            nemo_path.display(),
            stderr.trim()
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect())
}

fn find_tar_entry(entries: &[String], suffix: &str) -> Result<String> {
    entries
        .iter()
        .find(|e| e.ends_with(suffix))
        .cloned()
        .ok_or_else(|| {
            Error::ModelLoadError(format!("Unable to locate {} inside .nemo archive", suffix))
        })
}

fn extract_tar_entry_to_file(nemo_path: &Path, entry: &str, dest: &Path) -> Result<()> {
    let tmp_path = dest.with_extension("tmp");
    let mut tmp_file = File::create(&tmp_path).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed creating temp extraction file {}: {}",
            tmp_path.display(),
            e
        ))
    })?;

    let output = Command::new("tar")
        .arg("-xOf")
        .arg(nemo_path)
        .arg(entry)
        .output()
        .map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed extracting {} from {}: {}",
                entry,
                nemo_path.display(),
                e
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::ModelLoadError(format!(
            "Failed extracting {} from {}: {}",
            entry,
            nemo_path.display(),
            stderr.trim()
        )));
    }

    tmp_file.write_all(&output.stdout).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed writing extracted file {}: {}",
            tmp_path.display(),
            e
        ))
    })?;

    fs::rename(&tmp_path, dest).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed moving extracted artifact into {}: {}",
            dest.display(),
            e
        ))
    })?;

    Ok(())
}
