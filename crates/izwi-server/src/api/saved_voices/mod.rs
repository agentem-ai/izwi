//! Persistent saved voice routes for reusable voice cloning references.

mod handlers;

use axum::{extract::DefaultBodyLimit, routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    const AUDIO_UPLOAD_LIMIT_BYTES: usize = 64 * 1024 * 1024;

    Router::new()
        .route(
            "/voices",
            get(handlers::list_saved_voices)
                .post(handlers::create_saved_voice)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .route(
            "/voices/:voice_id",
            get(handlers::get_saved_voice).delete(handlers::delete_saved_voice),
        )
        .route(
            "/voices/:voice_id/audio",
            get(handlers::get_saved_voice_audio),
        )
}
