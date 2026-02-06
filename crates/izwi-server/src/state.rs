//! Application state management with high-concurrency optimizations

use izwi_core::InferenceEngine;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Shared application state with fine-grained locking and backpressure
#[derive(Clone)]
pub struct AppState {
    /// Engine reference - using Arc for cheap clones
    pub engine: Arc<InferenceEngine>,
    /// Concurrency limiter to prevent resource exhaustion
    pub request_semaphore: Arc<Semaphore>,
    /// Request timeout configuration (seconds)
    pub request_timeout_secs: u64,
}

impl AppState {
    pub fn new(engine: InferenceEngine) -> Self {
        // Limit concurrent requests to prevent overwhelming the system
        // Default: 100 concurrent requests (tunable based on hardware)
        let max_concurrent = std::env::var("MAX_CONCURRENT_REQUESTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);

        let timeout = std::env::var("REQUEST_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300); // 5 minutes default

        Self {
            engine: Arc::new(engine),
            request_semaphore: Arc::new(Semaphore::new(max_concurrent)),
            request_timeout_secs: timeout,
        }
    }

    /// Acquire a permit for concurrent request processing
    pub async fn acquire_permit(&self) -> tokio::sync::SemaphorePermit<'_> {
        self.request_semaphore
            .acquire()
            .await
            .expect("Semaphore should never be closed")
    }
}
