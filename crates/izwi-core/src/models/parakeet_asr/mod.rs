//! Parakeet ASR model integration.
//!
//! Parakeet checkpoints are distributed as `.nemo` archives. Inference is
//! delegated to NVIDIA NeMo through a Python subprocess.

use std::path::{Path, PathBuf};
use std::process::Command;

use hound::{SampleFormat, WavSpec, WavWriter};
use serde::Deserialize;
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::model::ModelVariant;

const PYTHON_NEMO_CHECK_SCRIPT: &str = r#"
import nemo.collections.asr as nemo_asr
assert hasattr(nemo_asr.models, "ASRModel")
print("ok")
"#;

const PYTHON_TRANSCRIBE_SCRIPT: &str = r#"
import json
import sys

import nemo.collections.asr as nemo_asr

def item_to_text(item):
    if isinstance(item, str):
        return item
    if hasattr(item, "text"):
        return item.text
    if isinstance(item, dict):
        for key in ("text", "pred_text", "transcript"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    return str(item)

def main():
    if len(sys.argv) < 3:
        raise ValueError("expected model_path and audio_path arguments")

    model_path = sys.argv[1]
    audio_path = sys.argv[2]

    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=model_path,
        map_location="cpu",
    )
    output = model.transcribe([audio_path])
    if isinstance(output, (list, tuple)):
        item = output[0] if output else ""
    else:
        item = output

    text = item_to_text(item)
    print(json.dumps({"text": text}, ensure_ascii=False))

if __name__ == "__main__":
    main()
"#;

#[derive(Debug, Clone)]
enum Runner {
    NemoPython { executable: String },
    External { executable: String },
}

#[derive(Debug, Clone)]
pub struct ParakeetAsrModel {
    variant: ModelVariant,
    nemo_path: PathBuf,
    runner: Runner,
}

#[derive(Debug, Deserialize)]
struct RunnerJsonOutput {
    text: String,
}

impl ParakeetAsrModel {
    pub fn load(model_dir: &Path, variant: ModelVariant) -> Result<Self> {
        if !variant.is_parakeet() {
            return Err(Error::InvalidInput(format!(
                "Variant {} is not a Parakeet model",
                variant.dir_name()
            )));
        }

        let nemo_path = model_dir.join(expected_nemo_filename(variant));
        if !nemo_path.exists() {
            return Err(Error::ModelNotFound(format!(
                "Missing .nemo checkpoint for {} at {}",
                variant.dir_name(),
                nemo_path.display()
            )));
        }

        let runner = if let Ok(executable) = std::env::var("IZWI_PARAKEET_RUNNER") {
            let executable = executable.trim().to_string();
            if executable.is_empty() {
                return Err(Error::ConfigError(
                    "IZWI_PARAKEET_RUNNER is set but empty".to_string(),
                ));
            }
            Runner::External { executable }
        } else {
            let executable = resolve_python_executable()?;
            ensure_nemo_available(&executable)?;
            Runner::NemoPython { executable }
        };

        Ok(Self {
            variant,
            nemo_path,
            runner,
        })
    }

    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_with_callback(audio, sample_rate, language, &mut no_op)
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        _language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let mono_16khz = if sample_rate == 16_000 {
            audio.to_vec()
        } else {
            resample_linear(audio, sample_rate, 16_000)
        };

        let wav_path = write_temp_wav(&mono_16khz, 16_000)?;

        let output = match self.invoke_runner(&wav_path) {
            Ok(text) => text,
            Err(err) => {
                let _ = std::fs::remove_file(&wav_path);
                return Err(err);
            }
        };
        let _ = std::fs::remove_file(&wav_path);

        for ch in output.chars() {
            let mut buf = [0u8; 4];
            on_delta(ch.encode_utf8(&mut buf));
        }

        Ok(output.trim().to_string())
    }

    fn invoke_runner(&self, wav_path: &Path) -> Result<String> {
        let output = match &self.runner {
            Runner::External { executable } => Command::new(executable)
                .arg("--model")
                .arg(&self.nemo_path)
                .arg("--audio")
                .arg(wav_path)
                .output(),
            Runner::NemoPython { executable } => Command::new(executable)
                .arg("-c")
                .arg(PYTHON_TRANSCRIBE_SCRIPT)
                .arg("--")
                .arg(&self.nemo_path)
                .arg(wav_path)
                .output(),
        }
        .map_err(|e| {
            Error::InferenceError(format!(
                "Failed to invoke Parakeet runner for {}: {}",
                self.variant.dir_name(),
                e
            ))
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::InferenceError(format!(
                "Parakeet runner failed for {}: {}",
                self.variant.dir_name(),
                stderr.trim()
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if stdout.is_empty() {
            return Err(Error::InferenceError(format!(
                "Parakeet runner returned empty output for {}",
                self.variant.dir_name()
            )));
        }

        if let Ok(parsed) = serde_json::from_str::<RunnerJsonOutput>(&stdout) {
            return Ok(parsed.text);
        }

        Ok(stdout)
    }
}

fn expected_nemo_filename(variant: ModelVariant) -> &'static str {
    match variant {
        ModelVariant::ParakeetTdt06BV2 => "parakeet-tdt-0.6b-v2.nemo",
        ModelVariant::ParakeetTdt06BV3 => "parakeet-tdt-0.6b-v3.nemo",
        _ => unreachable!("checked by caller"),
    }
}

fn resolve_python_executable() -> Result<String> {
    let mut candidates = Vec::new();
    if let Ok(explicit) = std::env::var("IZWI_PARAKEET_PYTHON") {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            candidates.push(trimmed.to_string());
        }
    }
    candidates.push("python3".to_string());
    candidates.push("python".to_string());

    for candidate in candidates {
        let status = Command::new(&candidate).arg("--version").status();
        if let Ok(status) = status {
            if status.success() {
                return Ok(candidate);
            }
        }
    }

    Err(Error::ConfigError(
        "Unable to find a Python executable for Parakeet ASR. Set IZWI_PARAKEET_PYTHON."
            .to_string(),
    ))
}

fn ensure_nemo_available(python: &str) -> Result<()> {
    let output = Command::new(python)
        .arg("-c")
        .arg(PYTHON_NEMO_CHECK_SCRIPT)
        .output()
        .map_err(|e| {
            Error::ConfigError(format!(
                "Failed to run Python environment check for Parakeet: {}",
                e
            ))
        })?;

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(Error::ConfigError(format!(
        "Parakeet ASR requires NVIDIA NeMo in Python. Install with `pip install -U nemo_toolkit['asr']`. Details: {}",
        stderr.trim()
    )))
}

fn write_temp_wav(samples: &[f32], sample_rate: u32) -> Result<PathBuf> {
    let path = std::env::temp_dir().join(format!("izwi-parakeet-{}.wav", Uuid::new_v4()));
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(&path, spec).map_err(|e| {
        Error::AudioError(format!(
            "Failed to create temporary WAV for Parakeet transcription: {}",
            e
        ))
    })?;

    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let value = if clamped < 0.0 {
            (clamped * 32768.0) as i16
        } else {
            (clamped * 32767.0) as i16
        };
        writer
            .write_sample(value)
            .map_err(|e| Error::AudioError(format!("Failed writing temporary WAV: {}", e)))?;
    }

    writer
        .finalize()
        .map_err(|e| Error::AudioError(format!("Failed finalizing temporary WAV: {}", e)))?;

    Ok(path)
}

fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || audio.len() < 2 {
        return audio.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((audio.len() as f64) * ratio).round().max(1.0) as usize;

    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = (i as f64) / ratio;
        let left = src_pos.floor() as usize;
        let right = left.saturating_add(1).min(audio.len() - 1);
        let frac = (src_pos - left as f64) as f32;
        let sample = audio[left] * (1.0 - frac) + audio[right] * frac;
        out.push(sample);
    }
    out
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn parakeet_lifecycle_with_external_runner() {
        let _guard = env_lock().lock().expect("env lock poisoned");

        let root = std::env::temp_dir().join(format!("izwi-parakeet-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();

        let nemo_path = root.join("parakeet-tdt-0.6b-v2.nemo");
        std::fs::write(&nemo_path, b"mock-nemo").unwrap();

        let runner_path = root.join("mock-runner.sh");
        let runner_script = r#"#!/bin/sh
set -eu
model=""
audio=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)
      model="$2"
      shift 2
      ;;
    --audio)
      audio="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [ ! -f "$model" ] || [ ! -f "$audio" ]; then
  echo "missing model or audio file" >&2
  exit 1
fi
printf '{"text":"parakeet mock transcription"}\n'
"#;
        std::fs::write(&runner_path, runner_script).unwrap();

        let mut perms = std::fs::metadata(&runner_path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&runner_path, perms).unwrap();

        unsafe {
            std::env::set_var("IZWI_PARAKEET_RUNNER", &runner_path);
        }

        let model = ParakeetAsrModel::load(&root, ModelVariant::ParakeetTdt06BV2).unwrap();
        let audio = vec![0.0_f32; 16_000];
        let text = model.transcribe(&audio, 16_000, None).unwrap();
        assert_eq!(text, "parakeet mock transcription");

        unsafe {
            std::env::remove_var("IZWI_PARAKEET_RUNNER");
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
