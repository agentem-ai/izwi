//! Izwi TTS Server - HTTP API for Qwen3-TTS inference

use tokio::signal;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod api;
mod chat_store;
mod error;
mod speech_history_store;
mod state;
mod transcription_store;

use izwi_core::{EngineConfig, RuntimeService};
use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "izwi_server=warn,izwi_core=warn,tower_http=warn".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting Izwi TTS Server");

    // Load configuration
    let config = EngineConfig::default();
    info!("Models directory: {:?}", config.models_dir);

    // Create runtime service
    let runtime = RuntimeService::new(config)?;
    let state = AppState::new(runtime)?;

    info!("Runtime service initialized");

    // Build router
    let app = api::create_router(state.clone());

    // Start server
    let host = std::env::var("IZWI_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = match std::env::var("IZWI_PORT") {
        Ok(raw) => match raw.parse::<u16>() {
            Ok(parsed) => parsed,
            Err(_) => {
                warn!("Invalid IZWI_PORT='{}', falling back to 8080", raw);
                8080
            }
        },
        Err(_) => 8080,
    };
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Server listening on http://{}", addr);

    // Clone state for shutdown handler
    let shutdown_state = state.clone();

    // Spawn server with graceful shutdown
    let server = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal(shutdown_state));

    info!("Server ready. Press Ctrl+C to stop.");
    server.await?;

    Ok(())
}

/// Wait for shutdown signal and cleanup
async fn shutdown_signal(state: AppState) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, shutting down...");
        },
        _ = terminate => {
            info!("Received SIGTERM, shutting down...");
        },
    }
    drop(state);
}
