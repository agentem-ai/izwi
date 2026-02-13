//! OpenAI-compatible audio resources.

pub mod speech;
pub mod speech_to_speech;
pub mod transcriptions;

use axum::{extract::DefaultBodyLimit, routing::post, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    // Multipart audio uploads can be several MB; raise the default extractor cap.
    // 64 MiB supports typical local transcription use without being unbounded.
    const AUDIO_UPLOAD_LIMIT_BYTES: usize = 64 * 1024 * 1024;

    Router::new()
        .route(
            "/audio/speech",
            post(speech::speech).layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .route(
            "/audio/transcriptions",
            post(transcriptions::transcriptions)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .route(
            "/audio/speech-to-speech",
            post(speech_to_speech::speech_to_speech)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
}
