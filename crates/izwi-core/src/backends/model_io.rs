use serde::{Deserialize, Serialize};

use super::types::BackendKind;

/// GGUF file loading mode for memory mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GgufMmapMode {
    /// Backend-aware default policy.
    #[default]
    Auto,
    /// Force mmap.
    On,
    /// Force buffered I/O.
    Off,
}

impl GgufMmapMode {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" => None,
            "auto" => Some(Self::Auto),
            "on" | "true" | "1" | "yes" => Some(Self::On),
            "off" | "false" | "0" | "no" => Some(Self::Off),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

pub fn gguf_mmap_mode_from_env() -> Option<GgufMmapMode> {
    std::env::var("IZWI_GGUF_MMAP")
        .ok()
        .as_deref()
        .and_then(GgufMmapMode::parse)
}

pub fn resolve_gguf_mmap_mode(configured: Option<GgufMmapMode>) -> GgufMmapMode {
    configured
        .or_else(gguf_mmap_mode_from_env)
        .unwrap_or(GgufMmapMode::Auto)
}

/// Default mmap policy used for `auto` mode.
pub fn auto_gguf_mmap_for_backend(kind: BackendKind) -> bool {
    match kind {
        // Keep mmap enabled by default across current targets.
        BackendKind::Cpu | BackendKind::Metal | BackendKind::Cuda => true,
    }
}

pub fn gguf_mmap_enabled(mode: GgufMmapMode, kind: BackendKind) -> bool {
    match mode {
        GgufMmapMode::Auto => auto_gguf_mmap_for_backend(kind),
        GgufMmapMode::On => true,
        GgufMmapMode::Off => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env_test_lock;

    #[test]
    fn parse_mode_supports_aliases() {
        assert_eq!(GgufMmapMode::parse("auto"), Some(GgufMmapMode::Auto));
        assert_eq!(GgufMmapMode::parse("on"), Some(GgufMmapMode::On));
        assert_eq!(GgufMmapMode::parse("true"), Some(GgufMmapMode::On));
        assert_eq!(GgufMmapMode::parse("1"), Some(GgufMmapMode::On));
        assert_eq!(GgufMmapMode::parse("yes"), Some(GgufMmapMode::On));
        assert_eq!(GgufMmapMode::parse("off"), Some(GgufMmapMode::Off));
        assert_eq!(GgufMmapMode::parse("false"), Some(GgufMmapMode::Off));
        assert_eq!(GgufMmapMode::parse("0"), Some(GgufMmapMode::Off));
        assert_eq!(GgufMmapMode::parse("no"), Some(GgufMmapMode::Off));
    }

    #[test]
    fn parse_mode_rejects_unknown_values() {
        assert_eq!(GgufMmapMode::parse(""), None);
        assert_eq!(GgufMmapMode::parse("invalid"), None);
    }

    #[test]
    fn resolve_mode_prefers_explicit_value_then_env_then_auto_default() {
        let _guard = env_test_lock().lock().expect("env lock poisoned");

        std::env::remove_var("IZWI_GGUF_MMAP");
        assert_eq!(resolve_gguf_mmap_mode(None), GgufMmapMode::Auto);

        std::env::set_var("IZWI_GGUF_MMAP", "off");
        assert_eq!(resolve_gguf_mmap_mode(None), GgufMmapMode::Off);

        assert_eq!(
            resolve_gguf_mmap_mode(Some(GgufMmapMode::On)),
            GgufMmapMode::On
        );

        std::env::remove_var("IZWI_GGUF_MMAP");
    }

    #[test]
    fn auto_policy_currently_enables_mmap_for_all_backends() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            assert!(gguf_mmap_enabled(GgufMmapMode::Auto, backend));
        }
    }
}
