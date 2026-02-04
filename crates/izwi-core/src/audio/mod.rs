//! Audio processing utilities for TTS output

mod codec;
mod encoder;
mod preprocessing;
mod streaming;

pub use codec::{AudioCodec, CodecConfig};
pub use encoder::{AudioEncoder, AudioFormat};
pub use preprocessing::{MelConfig, MelSpectrogram};
pub use streaming::{AudioChunkBuffer, StreamingConfig};
