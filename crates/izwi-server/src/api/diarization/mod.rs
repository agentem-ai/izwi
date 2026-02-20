//! First-party diarization history routes for the desktop UI.

mod handlers;

use axum::{routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/diarization/records",
            get(handlers::list_records).post(handlers::create_record),
        )
        .route(
            "/diarization/records/:record_id",
            get(handlers::get_record).delete(handlers::delete_record),
        )
        .route(
            "/diarization/records/:record_id/audio",
            get(handlers::get_record_audio),
        )
}
