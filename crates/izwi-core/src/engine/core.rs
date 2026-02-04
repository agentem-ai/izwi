//! Engine core - the central orchestrator for inference.
//!
//! The engine core coordinates:
//! - Request scheduling
//! - Model execution
//! - KV cache management
//! - Output processing

use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info};

use super::config::EngineCoreConfig;
use super::executor::{UnifiedExecutor, WorkerConfig};
use super::kv_cache::{KVCacheConfig, KVCacheManager};
use super::output::OutputProcessor;
use super::request::{EngineCoreRequest, RequestStatus};
use super::scheduler::{Scheduler, SchedulerConfig};
use super::types::{EngineOutput, RequestId, SequenceId};
use crate::error::{Error, Result};

/// The engine core - manages the inference loop.
pub struct EngineCore {
    /// Configuration
    config: EngineCoreConfig,
    /// Request scheduler
    scheduler: Scheduler,
    /// KV cache manager
    kv_cache: KVCacheManager,
    /// Model executor
    executor: UnifiedExecutor,
    /// Output processor
    output_processor: OutputProcessor,
    /// Active requests (by ID)
    requests: HashMap<RequestId, EngineCoreRequest>,
    /// Request start times (for timing)
    request_start_times: HashMap<RequestId, Instant>,
    /// Sequence ID counter
    next_sequence_id: SequenceId,
    /// Whether the engine has been initialized
    initialized: bool,
}

impl EngineCore {
    /// Create a new engine core.
    pub fn new(config: EngineCoreConfig) -> Result<Self> {
        info!("Creating engine core");

        // Create scheduler
        let scheduler_config = SchedulerConfig::from(&config);
        let scheduler = Scheduler::new(scheduler_config);

        // Create KV cache manager
        let kv_config = KVCacheConfig {
            num_layers: 24,
            num_heads: 16,
            head_dim: 64,
            block_size: config.block_size,
            max_blocks: config.max_blocks,
            dtype_bytes: 2,
        };
        let kv_cache = KVCacheManager::new(kv_config);

        // Create executor
        let worker_config = WorkerConfig::from(&config);
        let executor = UnifiedExecutor::new_native(worker_config);

        // Create output processor
        let output_processor =
            OutputProcessor::new(config.sample_rate).with_chunk_size(config.streaming_chunk_size);

        Ok(Self {
            config,
            scheduler,
            kv_cache,
            executor,
            output_processor,
            requests: HashMap::new(),
            request_start_times: HashMap::new(),
            next_sequence_id: 0,
            initialized: false,
        })
    }

    /// Initialize the engine core.
    pub async fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        info!("Initializing engine core");

        // Initialize executor (starts daemon processes)
        self.executor.initialize().await?;

        self.initialized = true;
        info!("Engine core initialized");

        Ok(())
    }

    /// Add a request to the engine.
    pub fn add_request(&mut self, request: EngineCoreRequest) -> Result<()> {
        let request_id = request.id.clone();

        if self.requests.contains_key(&request_id) {
            return Err(Error::InvalidInput(format!(
                "Request {} already exists",
                request_id
            )));
        }

        // Add to scheduler
        self.scheduler.add_request(&request);

        // Track request
        self.requests.insert(request_id.clone(), request);
        self.request_start_times
            .insert(request_id.clone(), Instant::now());

        debug!("Added request {} to engine core", request_id);

        Ok(())
    }

    /// Execute one step of the inference loop.
    ///
    /// The step consists of:
    /// 1. Schedule - select requests to process
    /// 2. Execute - run forward pass
    /// 3. Process - handle outputs, check stop conditions
    pub async fn step(&mut self) -> Result<Vec<EngineOutput>> {
        // Ensure initialized
        if !self.initialized {
            self.initialize().await?;
        }

        // Phase 1: Schedule
        let schedule_result = self.scheduler.schedule(&mut self.kv_cache);

        if !schedule_result.has_work() {
            return Ok(Vec::new());
        }

        debug!(
            "Scheduled {} prefill, {} decode requests",
            schedule_result.prefill_requests.len(),
            schedule_result.decode_requests.len()
        );

        // Collect requests for execution
        let all_scheduled: Vec<_> = schedule_result
            .prefill_requests
            .iter()
            .chain(schedule_result.decode_requests.iter())
            .collect();

        let request_refs: Vec<&EngineCoreRequest> = all_scheduled
            .iter()
            .filter_map(|s| self.requests.get(&s.request_id))
            .collect();

        if request_refs.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 2: Execute
        let scheduled_refs: Vec<_> = all_scheduled.iter().map(|s| (*s).clone()).collect();
        let executor_outputs = self
            .executor
            .execute(&request_refs, &scheduled_refs)
            .await?;

        // Phase 3: Process outputs
        let mut outputs = Vec::new();

        for exec_output in executor_outputs {
            let request_id = exec_output.request_id.clone();

            // Get timing info
            let generation_time = self
                .request_start_times
                .get(&request_id)
                .map(|t| t.elapsed())
                .unwrap_or_default();

            // Get sequence ID from scheduler
            let sequence_id = self
                .scheduler
                .get_running_info(&request_id)
                .map(|(_, _)| self.next_sequence_id)
                .unwrap_or(0);

            // Process output
            let engine_output =
                self.output_processor
                    .process(exec_output.clone(), sequence_id, generation_time);

            // Update scheduler state
            if exec_output.finished {
                self.scheduler
                    .finish_request(&request_id, &mut self.kv_cache);
                self.requests.remove(&request_id);
                self.request_start_times.remove(&request_id);
                debug!("Finished request {}", request_id);
            } else {
                // Update for next step
                self.scheduler.update_after_step(
                    &request_id,
                    exec_output.tokens_processed,
                    exec_output.tokens_generated,
                    Vec::new(),
                );
            }

            outputs.push(engine_output);
        }

        Ok(outputs)
    }

    /// Check if there's pending work.
    pub fn has_pending_work(&self) -> bool {
        self.scheduler.has_pending_work()
    }

    /// Check if a request exists.
    pub fn has_request(&self, request_id: &RequestId) -> bool {
        self.requests.contains_key(request_id)
    }

    /// Get request status.
    pub fn get_request_status(&self, request_id: &RequestId) -> Option<RequestStatus> {
        self.scheduler.get_status(request_id)
    }

    /// Abort a request.
    pub fn abort_request(&mut self, request_id: &RequestId) -> bool {
        if self.scheduler.abort_request(request_id, &mut self.kv_cache) {
            self.requests.remove(request_id);
            self.request_start_times.remove(request_id);
            debug!("Aborted request {}", request_id);
            true
        } else {
            false
        }
    }

    /// Get number of pending (waiting) requests.
    pub fn pending_request_count(&self) -> usize {
        self.scheduler.waiting_count()
    }

    /// Get number of running requests.
    pub fn running_request_count(&self) -> usize {
        self.scheduler.running_count()
    }

    /// Get KV cache statistics.
    pub fn kv_cache_stats(&self) -> super::kv_cache::KVCacheStats {
        self.kv_cache.stats()
    }

    /// Get configuration.
    pub fn config(&self) -> &EngineCoreConfig {
        &self.config
    }

    /// Shutdown the engine core.
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down engine core");

        // Abort all pending requests
        let request_ids: Vec<_> = self.requests.keys().cloned().collect();
        for id in request_ids {
            self.abort_request(&id);
        }

        // Shutdown executor
        self.executor.shutdown().await?;

        self.initialized = false;
        info!("Engine core shutdown complete");

        Ok(())
    }
}

impl Drop for EngineCore {
    fn drop(&mut self) {
        // Note: We can't do async cleanup in drop, so we just log
        if self.initialized {
            debug!("EngineCore dropped while still initialized");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_core_creation() {
        let config = EngineCoreConfig::default();
        let core = EngineCore::new(config);
        assert!(core.is_ok());
    }

    #[tokio::test]
    async fn test_add_request() {
        let config = EngineCoreConfig::default();
        let mut core = EngineCore::new(config).unwrap();

        let request = EngineCoreRequest::tts("Hello, world!");
        let result = core.add_request(request);
        assert!(result.is_ok());
        assert_eq!(core.pending_request_count(), 1);
    }
}
