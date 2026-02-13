use axum::Router;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::state::AppState;

/// Create the main API router.
pub fn create_router(state: AppState) -> Router {
    let v1_routes = Router::new()
        .merge(crate::api::internal::router())
        .merge(crate::api::openai::router())
        .merge(crate::api::admin::router());

    Router::new()
        .nest("/v1", v1_routes)
        // Serve static files for UI
        .fallback_service(
            tower_http::services::ServeDir::new("ui/dist")
                .fallback(tower_http::services::ServeFile::new("ui/dist/index.html")),
        )
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state)
}
