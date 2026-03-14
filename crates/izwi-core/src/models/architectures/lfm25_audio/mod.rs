//! Native LFM2.5 Audio GGUF architecture support.

mod backbone;
mod bundle;
mod audio_output;
mod conformer;
mod config;
mod detokenizer;
mod model;
mod preprocessor;
mod tokenizer;

pub use model::Lfm25AudioModel;
