//! User preferences routes.

mod handlers;

use axum::Router;

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/preferences",
            axum::routing::get(handlers::get_preferences),
        )
        .route(
            "/preferences/analytics",
            axum::routing::put(handlers::update_analytics_preference),
        )
}
