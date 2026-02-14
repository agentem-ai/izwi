//! Runtime and engine telemetry endpoints.

use axum::{body::Body, extract::State, http::header, response::Response, Json};
use izwi_core::RuntimeTelemetrySnapshot;

use crate::state::AppState;

pub async fn metrics_json(State(state): State<AppState>) -> Json<RuntimeTelemetrySnapshot> {
    Json(state.runtime.telemetry_snapshot().await)
}

pub async fn metrics_prometheus(State(state): State<AppState>) -> Response<Body> {
    let payload = state.runtime.telemetry_prometheus().await;
    Response::builder()
        .header(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )
        .body(Body::from(payload))
        .unwrap()
}
