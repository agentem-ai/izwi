use std::path::{Path, PathBuf};
use std::time::Duration;

pub fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.1} {}", size, UNITS[unit_index])
}

pub fn format_number(num: usize) -> String {
    if num < 1000 {
        num.to_string()
    } else if num < 1_000_000 {
        format!("{:.1}K", num as f64 / 1000.0)
    } else {
        format!("{:.1}M", num as f64 / 1_000_000.0)
    }
}

pub fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

pub fn models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("IZWI_MODELS_DIR") {
        return PathBuf::from(dir);
    }

    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("izwi")
        .join("models")
}

pub fn model_dir_if_present(variant: &str) -> Option<PathBuf> {
    let path = models_dir().join(variant);
    if path.is_dir() && directory_has_files(&path) {
        Some(path)
    } else {
        None
    }
}

pub fn directory_size_bytes(path: &Path) -> Option<u64> {
    if !path.is_dir() {
        return None;
    }

    let mut total = 0u64;
    let mut stack = vec![path.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).ok()?;
        for entry in entries {
            let entry = entry.ok()?;
            let entry_path = entry.path();
            let meta = entry.metadata().ok()?;
            if meta.is_dir() {
                stack.push(entry_path);
            } else {
                total = total.saturating_add(meta.len());
            }
        }
    }

    Some(total)
}

fn directory_has_files(path: &Path) -> bool {
    std::fs::read_dir(path)
        .ok()
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false)
}
