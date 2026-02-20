//! First-party speech history routes for Text-to-Speech, Voice Design, and Voice Cloning.

mod handlers;

use axum::{extract::DefaultBodyLimit, routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    const AUDIO_UPLOAD_LIMIT_BYTES: usize = 64 * 1024 * 1024;

    Router::new()
        .route(
            "/text-to-speech/records",
            get(handlers::list_text_to_speech_records)
                .post(handlers::create_text_to_speech_record)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .route(
            "/text-to-speech/records/:record_id",
            get(handlers::get_text_to_speech_record).delete(handlers::delete_text_to_speech_record),
        )
        .route(
            "/text-to-speech/records/:record_id/audio",
            get(handlers::get_text_to_speech_record_audio),
        )
        .route(
            "/voice-design/records",
            get(handlers::list_voice_design_records)
                .post(handlers::create_voice_design_record)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .route(
            "/voice-design/records/:record_id",
            get(handlers::get_voice_design_record).delete(handlers::delete_voice_design_record),
        )
        .route(
            "/voice-design/records/:record_id/audio",
            get(handlers::get_voice_design_record_audio),
        )
        .route(
            "/voice-cloning/records",
            get(handlers::list_voice_cloning_records)
                .post(handlers::create_voice_cloning_record)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .route(
            "/voice-cloning/records/:record_id",
            get(handlers::get_voice_cloning_record).delete(handlers::delete_voice_cloning_record),
        )
        .route(
            "/voice-cloning/records/:record_id/audio",
            get(handlers::get_voice_cloning_record_audio),
        )
}
