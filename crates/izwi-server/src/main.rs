//! Izwi TTS Server - HTTP API for Qwen3-TTS inference

use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod api;
mod state;
mod error;

use state::AppState;
use izwi_core::{EngineConfig, InferenceEngine};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "izwi_server=debug,izwi_core=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting Izwi TTS Server");

    // Load configuration
    let config = EngineConfig::default();
    info!("Models directory: {:?}", config.models_dir);

    // Create inference engine
    let engine = InferenceEngine::new(config)?;
    let state = AppState::new(engine);

    // Build router
    let app = api::create_router(state);

    // Start server
    let addr = "0.0.0.0:8080";
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("Server listening on http://{}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}
