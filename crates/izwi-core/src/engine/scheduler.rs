//! Request scheduler with support for FCFS and priority-based scheduling.
//!
//! The scheduler manages request queues and decides which requests to process
//! in each engine step. It handles:
//! - Waiting queue (new requests awaiting processing)
//! - Running queue (requests currently being processed)
//! - Token budget management
//! - KV cache allocation coordination

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::time::Instant;
use tracing::debug;

use super::config::EngineCoreConfig;
use super::kv_cache::KVCacheManager;
use super::request::{EngineCoreRequest, RequestStatus};
use super::types::{BlockId, Priority, RequestId, SequenceId, TaskType};

/// Scheduling policy for the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SchedulingPolicy {
    /// First-come, first-served (default)
    #[default]
    FCFS,
    /// Priority-based scheduling (higher priority first)
    Priority,
}

/// Configuration for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum tokens per step (token budget)
    pub max_tokens_per_step: usize,
    /// Scheduling policy
    pub policy: SchedulingPolicy,
    /// Enable chunked prefill
    pub enable_chunked_prefill: bool,
    /// Threshold for chunked prefill
    pub chunked_prefill_threshold: usize,
    /// Enable preemption when KV cache is full
    pub enable_preemption: bool,
    /// Enable VAD-triggered preemption (for audio interruption handling)
    pub enable_vad_preemption: bool,
    /// Enable adaptive, latency-aware batching heuristics.
    pub enable_adaptive_batching: bool,
    /// Minimum token budget for adaptive scheduling.
    pub min_tokens_per_step: usize,
    /// Target time-to-first-token.
    pub target_ttft_ms: f64,
    /// Target decode time per output token.
    pub target_decode_tpot_ms: f64,
    /// Wait time interval used for priority aging.
    pub priority_aging_ms: u64,
}

/// Preemption reason - why a request was preempted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionReason {
    /// Memory pressure - KV cache is full
    MemoryPressure,
    /// VAD detected user speech during AI output (interruption)
    VadInterruption,
    /// Manual abort by user
    UserAbort,
    /// Timeout
    Timeout,
}

/// VAD preemption event - signals that user started speaking.
#[derive(Debug, Clone)]
pub struct VadPreemptionEvent {
    /// Timestamp of the VAD detection
    pub timestamp: Instant,
    /// Speech probability from VAD
    pub speech_probability: f32,
    /// Request IDs that should be preempted (currently generating requests)
    pub requests_to_preempt: Vec<RequestId>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_tokens_per_step: 512,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: true,
            chunked_prefill_threshold: 256,
            enable_preemption: true,
            enable_vad_preemption: true,
            enable_adaptive_batching: true,
            min_tokens_per_step: 128,
            target_ttft_ms: 250.0,
            target_decode_tpot_ms: 40.0,
            priority_aging_ms: 1_000,
        }
    }
}

impl From<&EngineCoreConfig> for SchedulerConfig {
    fn from(config: &EngineCoreConfig) -> Self {
        Self {
            max_batch_size: config.max_batch_size,
            max_tokens_per_step: config.max_tokens_per_step,
            policy: config.scheduling_policy,
            enable_chunked_prefill: config.enable_chunked_prefill,
            chunked_prefill_threshold: config.chunked_prefill_threshold,
            enable_preemption: config.enable_preemption,
            enable_vad_preemption: true, // Default to enabled for audio apps
            enable_adaptive_batching: config.enable_adaptive_batching,
            min_tokens_per_step: config.min_tokens_per_step,
            target_ttft_ms: config.target_ttft_ms,
            target_decode_tpot_ms: config.target_decode_tpot_ms,
            priority_aging_ms: config.priority_aging_ms,
        }
    }
}

/// A request wrapper for priority queue ordering.
#[derive(Debug, Clone)]
struct PriorityRequest {
    request_id: RequestId,
    priority: Priority,
    arrival_time: Instant,
}

impl PartialEq for PriorityRequest {
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
    }
}

impl Eq for PriorityRequest {}

impl PartialOrd for PriorityRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then earlier arrival time
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => other.arrival_time.cmp(&self.arrival_time), // Earlier is greater
            ord => ord,
        }
    }
}

/// Result of scheduling a step.
#[derive(Debug, Clone)]
pub struct ScheduleResult {
    /// Requests scheduled for decode (already running)
    pub decode_requests: Vec<ScheduledRequest>,
    /// Requests scheduled for prefill (new requests)
    pub prefill_requests: Vec<ScheduledRequest>,
    /// Requests that were preempted to make room
    pub preempted_requests: Vec<RequestId>,
    /// Total tokens to process this step
    pub total_tokens: usize,
    /// Number of blocks allocated
    pub blocks_allocated: usize,
}

impl ScheduleResult {
    pub fn empty() -> Self {
        Self {
            decode_requests: Vec::new(),
            prefill_requests: Vec::new(),
            preempted_requests: Vec::new(),
            total_tokens: 0,
            blocks_allocated: 0,
        }
    }

    /// Check if there's any work to do
    pub fn has_work(&self) -> bool {
        !self.decode_requests.is_empty() || !self.prefill_requests.is_empty()
    }

    /// Get all scheduled request IDs
    pub fn all_request_ids(&self) -> Vec<RequestId> {
        let mut ids: Vec<_> = self
            .decode_requests
            .iter()
            .chain(self.prefill_requests.iter())
            .map(|r| r.request_id.clone())
            .collect();
        ids.dedup();
        ids
    }
}

/// A request that has been scheduled for processing.
#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    /// Request ID
    pub request_id: RequestId,
    /// Sequence ID
    pub sequence_id: SequenceId,
    /// Number of tokens to process this step
    pub num_tokens: usize,
    /// Whether this is a prefill (first pass) or decode (continuation)
    pub is_prefill: bool,
    /// KV cache blocks allocated to this request
    pub block_ids: Vec<BlockId>,
    /// Number of tokens already computed (for chunked prefill)
    pub num_computed_tokens: usize,
}

/// Runtime telemetry used by adaptive scheduling.
#[derive(Debug, Clone)]
pub struct SchedulerTelemetry {
    /// Exponential moving average of time-to-first-token.
    pub avg_ttft_ms: f64,
    /// Exponential moving average of decode time per generated token.
    pub avg_decode_tpot_ms: f64,
    /// Exponential moving average of waiting queue age.
    pub avg_queue_age_ms: f64,
    /// Current adaptive token budget.
    pub dynamic_tokens_per_step: usize,
}

impl SchedulerTelemetry {
    fn new(default_budget: usize) -> Self {
        Self {
            avg_ttft_ms: 0.0,
            avg_decode_tpot_ms: 0.0,
            avg_queue_age_ms: 0.0,
            dynamic_tokens_per_step: default_budget.max(1),
        }
    }

    fn update_ewma(current: &mut f64, sample: f64, alpha: f64) {
        if sample <= 0.0 {
            return;
        }
        if *current <= 0.0 {
            *current = sample;
        } else {
            *current = (*current * (1.0 - alpha)) + (sample * alpha);
        }
    }
}

/// Request scheduler.
pub struct Scheduler {
    config: SchedulerConfig,
    /// Waiting queue (FCFS mode)
    waiting_fcfs: VecDeque<RequestId>,
    /// Waiting queue (Priority mode)
    waiting_priority: BinaryHeap<PriorityRequest>,
    /// Running requests (by request ID)
    running: HashMap<RequestId, RunningRequest>,
    /// Request metadata
    requests: HashMap<RequestId, RequestMetadata>,
    /// Next sequence ID
    next_sequence_id: SequenceId,
    /// Adaptive scheduling telemetry.
    telemetry: SchedulerTelemetry,
}

/// Metadata for a request in the scheduler.
#[derive(Debug, Clone)]
struct RequestMetadata {
    request_id: RequestId,
    sequence_id: SequenceId,
    priority: Priority,
    arrival_time: Instant,
    total_prompt_tokens: usize,
    max_tokens: usize,
    prompt_prefix_tokens: Vec<u32>,
}

/// State for a running request.
#[derive(Debug, Clone)]
struct RunningRequest {
    request_id: RequestId,
    sequence_id: SequenceId,
    /// Number of tokens processed so far (prompt + generated)
    num_tokens_processed: usize,
    /// Number of tokens generated so far
    num_tokens_generated: usize,
    /// KV cache blocks allocated
    block_ids: Vec<BlockId>,
    /// Whether prefill is complete
    prefill_complete: bool,
    /// Priority of this request
    priority: Priority,
    /// Whether this request has produced its first output token.
    first_token_emitted: bool,
    /// Whether this request is temporarily paused due to preemption.
    paused: bool,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(config: SchedulerConfig) -> Self {
        let telemetry = SchedulerTelemetry::new(config.max_tokens_per_step);
        Self {
            config,
            waiting_fcfs: VecDeque::new(),
            waiting_priority: BinaryHeap::new(),
            running: HashMap::new(),
            requests: HashMap::new(),
            next_sequence_id: 0,
            telemetry,
        }
    }

    /// Add a request to the waiting queue.
    pub fn add_request(&mut self, request: &EngineCoreRequest) {
        let sequence_id = self.next_sequence_id;
        self.next_sequence_id += 1;

        let max_tokens = match (request.task_type, request.params.max_tokens) {
            (TaskType::TTS, 0) => usize::MAX,
            (_, 0) => 2048,
            (_, value) => value,
        };

        let metadata = RequestMetadata {
            request_id: request.id.clone(),
            sequence_id,
            priority: request.priority,
            arrival_time: Instant::now(),
            total_prompt_tokens: request.num_prompt_tokens(),
            // TTS uses max_tokens=0 to indicate "auto". Keep scheduler decode budget
            // effectively unbounded so model-level stop criteria can terminate naturally.
            // For other task types, guard against zero-budget stalls if upstream
            // validation is ever bypassed.
            max_tokens,
            prompt_prefix_tokens: request.prompt_tokens.clone(),
        };

        self.requests.insert(request.id.clone(), metadata);

        match self.config.policy {
            SchedulingPolicy::FCFS => {
                self.waiting_fcfs.push_back(request.id.clone());
            }
            SchedulingPolicy::Priority => {
                self.waiting_priority.push(PriorityRequest {
                    request_id: request.id.clone(),
                    priority: request.priority,
                    arrival_time: Instant::now(),
                });
            }
        }

        debug!(
            "Added request {} to waiting queue (sequence_id={}, prompt_tokens={})",
            request.id,
            sequence_id,
            request.num_prompt_tokens()
        );
    }

    /// Schedule requests for the next step.
    pub fn schedule(&mut self, kv_cache: &mut KVCacheManager) -> ScheduleResult {
        let mut result = ScheduleResult::empty();
        let mut remaining_batch = self.config.max_batch_size;
        self.refresh_queue_age_sample();
        self.update_dynamic_budget();

        let mut total_budget = self.current_token_budget();
        let kv_stats = kv_cache.stats();
        let kv_utilization = if kv_stats.soft_max_blocks > 0 {
            kv_stats.allocated_blocks as f64 / kv_stats.soft_max_blocks as f64
        } else {
            0.0
        };
        if kv_utilization > 0.95 {
            total_budget = (total_budget.saturating_mul(45) / 100).max(1);
        } else if kv_utilization > 0.90 {
            total_budget = (total_budget.saturating_mul(65) / 100).max(1);
        } else if kv_utilization > 0.80 {
            total_budget = (total_budget.saturating_mul(80) / 100).max(1);
        }
        let mut decode_budget = total_budget;
        let mut reserved_prefill_budget = 0;
        if self.config.enable_adaptive_batching && total_budget > 0 {
            let prefill_share = if self.telemetry.avg_ttft_ms > self.config.target_ttft_ms {
                0.55
            } else if self.telemetry.avg_ttft_ms > self.config.target_ttft_ms * 0.8 {
                0.40
            } else {
                0.25
            };
            reserved_prefill_budget = ((total_budget as f64) * prefill_share) as usize;
            reserved_prefill_budget = reserved_prefill_budget.clamp(1, total_budget);
            decode_budget = total_budget.saturating_sub(reserved_prefill_budget);
        }
        let mut remaining_decode_budget = decode_budget;

        // Phase 1: schedule decode requests (already running prefill-complete requests).
        let mut decode_candidates: Vec<_> = self
            .running
            .iter()
            .filter(|(_, r)| r.prefill_complete)
            .filter_map(|(id, r)| {
                let metadata = self.requests.get(id)?;
                let remaining_decode_tokens =
                    metadata.max_tokens.saturating_sub(r.num_tokens_generated);
                if remaining_decode_tokens == 0 {
                    return None;
                }

                let total_tokens = r.num_tokens_processed + 1;
                let blocks_needed = kv_cache.blocks_for_tokens(total_tokens);
                let additional_blocks = if blocks_needed > r.block_ids.len() {
                    blocks_needed - r.block_ids.len()
                } else {
                    0
                };
                Some((
                    id.clone(),
                    r.sequence_id,
                    r.priority,
                    r.block_ids.clone(),
                    r.num_tokens_processed,
                    additional_blocks,
                    remaining_decode_tokens,
                    r.num_tokens_generated,
                    r.paused,
                ))
            })
            .collect();

        if self.config.enable_adaptive_batching {
            // Favor requests close to completion to reduce tail latency, then priority.
            decode_candidates.sort_by(|a, b| {
                a.6.cmp(&b.6)
                    .then_with(|| b.2.cmp(&a.2))
                    .then_with(|| a.7.cmp(&b.7))
            });
        }
        let has_decode_demand = !decode_candidates.is_empty();
        let effective_prefill_chunk_threshold =
            self.effective_prefill_chunk_threshold(kv_utilization, has_decode_demand);

        for (
            request_id,
            sequence_id,
            priority,
            mut block_ids,
            num_computed,
            additional_blocks,
            _remaining_decode_tokens,
            _generated_tokens,
            _paused,
        ) in decode_candidates
        {
            if remaining_batch == 0 || remaining_decode_budget == 0 {
                break;
            }

            let num_tokens = 1;

            // Check if we need to allocate more blocks
            if additional_blocks > 0 {
                if !kv_cache.can_allocate(additional_blocks) {
                    // Try preemption if enabled
                    if self.config.enable_preemption {
                        let preempted =
                            self.try_preempt_for_blocks(additional_blocks, priority, kv_cache);
                        if !preempted.is_empty() {
                            result.preempted_requests.extend(preempted);
                            // Re-check if we can allocate now
                            if !kv_cache.can_allocate(additional_blocks) {
                                debug!("Still cannot allocate after preemption for {}", request_id);
                                continue;
                            }
                        } else {
                            debug!("No suitable requests to preempt for {}", request_id);
                            continue;
                        }
                    } else {
                        continue;
                    }
                }

                let extended_blocks = kv_cache.extend(&request_id, additional_blocks);
                if extended_blocks.len() < additional_blocks {
                    kv_cache.free(&request_id);
                    continue;
                }
                block_ids.extend(extended_blocks);
                result.blocks_allocated += additional_blocks;
            }

            // Shared-prefix blocks must be detached before appending decode tokens.
            if !block_ids.is_empty() && kv_cache.ensure_writable_last_block(&request_id).is_none() {
                if self.config.enable_preemption {
                    let preempted = self.try_preempt_for_blocks(1, priority, kv_cache);
                    if !preempted.is_empty() {
                        result.preempted_requests.extend(preempted);
                    }
                }
                if kv_cache.ensure_writable_last_block(&request_id).is_none() {
                    continue;
                }
            }

            if let Some(updated_blocks) = kv_cache.get_block_table(&request_id) {
                block_ids = updated_blocks.to_vec();
            }

            if let Some(running) = self.running.get_mut(&request_id) {
                running.paused = false;
                running.block_ids = block_ids.clone();
            }

            result.decode_requests.push(ScheduledRequest {
                request_id: request_id.clone(),
                sequence_id,
                num_tokens,
                is_prefill: false,
                block_ids,
                num_computed_tokens: num_computed,
            });

            remaining_decode_budget = remaining_decode_budget.saturating_sub(num_tokens);
            remaining_batch -= 1;
            result.total_tokens += num_tokens;
        }

        // Phase 2: schedule prefill requests.
        let mut remaining_prefill_budget = if self.config.enable_adaptive_batching {
            reserved_prefill_budget.saturating_add(remaining_decode_budget)
        } else {
            remaining_decode_budget
        };
        let prefill_admission_cap = if has_decode_demand && kv_utilization > 0.90 {
            1
        } else if has_decode_demand && kv_utilization > 0.80 {
            2
        } else {
            usize::MAX
        };
        let mut prefill_admissions = 0usize;

        // Phase 2a: resume preempted prefill requests before admitting new waiting requests.
        let mut paused_prefill_candidates: Vec<_> = self
            .running
            .iter()
            .filter(|(_, r)| r.paused && !r.prefill_complete)
            .map(|(id, r)| {
                (
                    id.clone(),
                    r.priority,
                    r.sequence_id,
                    r.num_tokens_processed,
                )
            })
            .collect();
        paused_prefill_candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.3.cmp(&b.3)));

        for (request_id, priority, sequence_id, num_computed) in paused_prefill_candidates {
            if remaining_batch == 0 || remaining_prefill_budget == 0 {
                break;
            }

            let metadata = match self.requests.get(&request_id) {
                Some(m) => m.clone(),
                None => continue,
            };

            let remaining_prompt = metadata.total_prompt_tokens.saturating_sub(num_computed);
            if remaining_prompt == 0 {
                if let Some(running) = self.running.get_mut(&request_id) {
                    running.prefill_complete = true;
                    running.paused = false;
                }
                continue;
            }

            let mut num_tokens = remaining_prompt;
            if self.config.enable_chunked_prefill && num_tokens > effective_prefill_chunk_threshold
            {
                num_tokens = effective_prefill_chunk_threshold;
            }
            num_tokens = num_tokens.min(remaining_prefill_budget);
            if num_tokens == 0 {
                continue;
            }

            let existing_blocks = self
                .running
                .get(&request_id)
                .map(|r| r.block_ids.len())
                .unwrap_or(0);
            let total_tokens_after = num_computed.saturating_add(num_tokens);
            let blocks_needed = kv_cache.blocks_for_tokens(total_tokens_after);
            let additional_blocks = blocks_needed.saturating_sub(existing_blocks);

            if additional_blocks > 0 {
                if !kv_cache.can_allocate(additional_blocks) {
                    if self.config.enable_preemption {
                        let preempted =
                            self.try_preempt_for_blocks(additional_blocks, priority, kv_cache);
                        if !preempted.is_empty() {
                            result.preempted_requests.extend(preempted);
                            if !kv_cache.can_allocate(additional_blocks) {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }

                let extended_blocks = kv_cache.extend(&request_id, additional_blocks);
                if extended_blocks.len() < additional_blocks {
                    kv_cache.free(&request_id);
                    continue;
                }
                result.blocks_allocated += additional_blocks;
            }

            let block_ids = kv_cache
                .get_block_table(&request_id)
                .map(|ids| ids.to_vec())
                .unwrap_or_default();

            if let Some(running) = self.running.get_mut(&request_id) {
                running.paused = false;
                running.block_ids = block_ids.clone();
            }

            result.prefill_requests.push(ScheduledRequest {
                request_id: request_id.clone(),
                sequence_id,
                num_tokens,
                is_prefill: true,
                block_ids,
                num_computed_tokens: num_computed,
            });

            remaining_prefill_budget = remaining_prefill_budget.saturating_sub(num_tokens);
            remaining_batch -= 1;
            result.total_tokens += num_tokens;
        }

        while remaining_batch > 0
            && remaining_prefill_budget > 0
            && prefill_admissions < prefill_admission_cap
        {
            let next_request_id = self.select_next_waiting_request();

            let request_id = match next_request_id {
                Some(id) => id,
                None => break,
            };

            let metadata = match self.requests.get(&request_id) {
                Some(m) => m.clone(),
                None => {
                    self.remove_from_waiting(&request_id);
                    continue;
                }
            };

            // Check if already running (shouldn't happen, but safety check)
            if self.running.contains_key(&request_id) {
                self.remove_from_waiting(&request_id);
                continue;
            }

            // Calculate tokens for this prefill
            let mut num_tokens = metadata.total_prompt_tokens;

            // Apply chunked prefill if enabled and prompt is long
            if self.config.enable_chunked_prefill && num_tokens > effective_prefill_chunk_threshold
            {
                num_tokens = effective_prefill_chunk_threshold;
            }

            // Limit by remaining budget
            num_tokens = num_tokens.min(remaining_prefill_budget);
            if num_tokens == 0 {
                break;
            }

            // Allocate KV cache blocks
            let blocks_needed = kv_cache.blocks_for_tokens(num_tokens);
            if !kv_cache.can_allocate(blocks_needed) {
                // Can't fit this request, try preemption or skip
                if self.config.enable_preemption {
                    let preempted =
                        self.try_preempt_for_blocks(blocks_needed, metadata.priority, kv_cache);
                    if !preempted.is_empty() {
                        result.preempted_requests.extend(preempted);
                        // Re-check if we can allocate now
                        if !kv_cache.can_allocate(blocks_needed) {
                            debug!(
                                "Still cannot allocate after preemption for prefill {}",
                                request_id
                            );
                            break;
                        }
                    } else {
                        debug!("No suitable requests to preempt for prefill {}", request_id);
                        break;
                    }
                } else {
                    break;
                }
            }

            let block_ids = kv_cache.allocate_with_prefix_tokens(
                &request_id,
                blocks_needed,
                &metadata.prompt_prefix_tokens,
            );
            if block_ids.len() < blocks_needed {
                debug!("Failed to allocate required blocks for {}", request_id);
                kv_cache.free(&request_id);
                break;
            }
            result.blocks_allocated += block_ids.len();

            // Create running state
            let running = RunningRequest {
                request_id: request_id.clone(),
                sequence_id: metadata.sequence_id,
                num_tokens_processed: 0,
                num_tokens_generated: 0,
                block_ids: block_ids.clone(),
                prefill_complete: num_tokens >= metadata.total_prompt_tokens,
                priority: metadata.priority,
                first_token_emitted: false,
                paused: false,
            };

            result.prefill_requests.push(ScheduledRequest {
                request_id: request_id.clone(),
                sequence_id: metadata.sequence_id,
                num_tokens,
                is_prefill: true,
                block_ids,
                num_computed_tokens: 0,
            });

            self.running.insert(request_id.clone(), running);
            self.remove_from_waiting(&request_id);

            remaining_prefill_budget = remaining_prefill_budget.saturating_sub(num_tokens);
            remaining_batch -= 1;
            prefill_admissions = prefill_admissions.saturating_add(1);
            result.total_tokens += num_tokens;
        }

        result
    }

    /// Update request state after a step.
    pub fn update_after_step(
        &mut self,
        request_id: &RequestId,
        tokens_processed: usize,
        tokens_generated: usize,
        new_block_ids: Vec<BlockId>,
        step_time_ms: f64,
    ) {
        if let Some(running) = self.running.get_mut(request_id) {
            running.paused = false;
            running.num_tokens_processed += tokens_processed;
            running.num_tokens_generated += tokens_generated;
            running.block_ids.extend(new_block_ids);

            // Check if prefill is now complete
            if let Some(metadata) = self.requests.get(request_id) {
                if running.num_tokens_processed >= metadata.total_prompt_tokens {
                    running.prefill_complete = true;
                }

                if !running.first_token_emitted && tokens_generated > 0 {
                    running.first_token_emitted = true;
                    let ttft_ms = metadata.arrival_time.elapsed().as_secs_f64() * 1000.0;
                    SchedulerTelemetry::update_ewma(&mut self.telemetry.avg_ttft_ms, ttft_ms, 0.20);
                }
            }

            if tokens_generated > 0 && step_time_ms > 0.0 {
                let tpot_ms = step_time_ms / tokens_generated as f64;
                SchedulerTelemetry::update_ewma(
                    &mut self.telemetry.avg_decode_tpot_ms,
                    tpot_ms,
                    0.15,
                );
            }
        }
        self.update_dynamic_budget();
    }

    /// Mark a request as finished and remove it.
    pub fn finish_request(&mut self, request_id: &RequestId, kv_cache: &mut KVCacheManager) {
        if let Some(running) = self.running.remove(request_id) {
            // Free KV cache blocks
            kv_cache.free(&running.request_id);
            debug!(
                "Finished request {}, freed {} blocks",
                request_id,
                running.block_ids.len()
            );
        }
        self.requests.remove(request_id);
    }

    /// Abort a request.
    pub fn abort_request(&mut self, request_id: &RequestId, kv_cache: &mut KVCacheManager) -> bool {
        // Remove from waiting queue
        self.waiting_fcfs.retain(|id| id != request_id);
        self.waiting_priority
            .retain(|r| &r.request_id != request_id);

        // Remove from running
        if let Some(running) = self.running.remove(request_id) {
            kv_cache.free(&running.request_id);
            self.requests.remove(request_id);
            return true;
        }

        self.requests.remove(request_id);
        false
    }

    /// Check if a request exists in the scheduler.
    pub fn has_request(&self, request_id: &RequestId) -> bool {
        self.requests.contains_key(request_id)
    }

    /// Get request status.
    pub fn get_status(&self, request_id: &RequestId) -> Option<RequestStatus> {
        if self.running.contains_key(request_id) {
            Some(RequestStatus::Running)
        } else if self.requests.contains_key(request_id) {
            Some(RequestStatus::Waiting)
        } else {
            None
        }
    }

    /// Get number of waiting requests.
    pub fn waiting_count(&self) -> usize {
        match self.config.policy {
            SchedulingPolicy::FCFS => self.waiting_fcfs.len(),
            SchedulingPolicy::Priority => self.waiting_priority.len(),
        }
    }

    /// Get number of running requests.
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Check if there's pending work.
    pub fn has_pending_work(&self) -> bool {
        self.waiting_count() > 0 || self.running_count() > 0
    }

    /// Get running request info.
    pub fn get_running_info(&self, request_id: &RequestId) -> Option<(usize, usize)> {
        self.running
            .get(request_id)
            .map(|r| (r.num_tokens_processed, r.num_tokens_generated))
    }

    /// Get sequence ID for a request.
    pub fn get_sequence_id(&self, request_id: &RequestId) -> Option<SequenceId> {
        self.requests.get(request_id).map(|m| m.sequence_id)
    }

    /// Adaptive scheduler telemetry.
    pub fn telemetry(&self) -> SchedulerTelemetry {
        self.telemetry.clone()
    }

    // Helper methods

    fn select_next_waiting_request(&self) -> Option<RequestId> {
        if !self.config.enable_adaptive_batching {
            return match self.config.policy {
                SchedulingPolicy::FCFS => self.waiting_fcfs.front().cloned(),
                SchedulingPolicy::Priority => {
                    self.waiting_priority.peek().map(|r| r.request_id.clone())
                }
            };
        }

        let candidates: Vec<RequestId> = match self.config.policy {
            SchedulingPolicy::FCFS => self.waiting_fcfs.iter().cloned().collect(),
            SchedulingPolicy::Priority => self
                .waiting_priority
                .iter()
                .map(|r| r.request_id.clone())
                .collect(),
        };

        candidates.into_iter().max_by(|a, b| {
            let score_a = self.adaptive_waiting_score(a);
            let score_b = self.adaptive_waiting_score(b);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn remove_from_waiting(&mut self, request_id: &RequestId) {
        self.waiting_fcfs.retain(|id| id != request_id);
        self.waiting_priority
            .retain(|r| &r.request_id != request_id);
    }

    fn adaptive_waiting_score(&self, request_id: &RequestId) -> f64 {
        let Some(metadata) = self.requests.get(request_id) else {
            return 0.0;
        };
        let base_priority = metadata.priority as i32 as f64;
        let age_ms = metadata.arrival_time.elapsed().as_millis() as f64;
        let age_boost = age_ms / self.config.priority_aging_ms.max(1) as f64;
        let prompt_bonus = 1.0
            / (1.0
                + (metadata.total_prompt_tokens as f64
                    / self.config.chunked_prefill_threshold.max(1) as f64));
        base_priority + age_boost + (prompt_bonus * 0.2)
    }

    fn refresh_queue_age_sample(&mut self) {
        let (sum_ms, count) = self
            .requests
            .values()
            .fold((0.0, 0usize), |(sum, n), metadata| {
                if self.running.contains_key(&metadata.request_id) {
                    (sum, n)
                } else {
                    (
                        sum + metadata.arrival_time.elapsed().as_secs_f64() * 1000.0,
                        n + 1,
                    )
                }
            });
        if count > 0 {
            let avg = sum_ms / count as f64;
            SchedulerTelemetry::update_ewma(&mut self.telemetry.avg_queue_age_ms, avg, 0.2);
        }
    }

    fn current_token_budget(&self) -> usize {
        let max_tokens = self.config.max_tokens_per_step.max(1);
        let min_tokens = self.config.min_tokens_per_step.min(max_tokens);
        if self.config.enable_adaptive_batching {
            self.telemetry
                .dynamic_tokens_per_step
                .clamp(min_tokens, max_tokens)
        } else {
            max_tokens
        }
    }

    fn effective_prefill_chunk_threshold(
        &self,
        kv_utilization: f64,
        has_decode_demand: bool,
    ) -> usize {
        let base = self.config.chunked_prefill_threshold.max(32);
        let mut threshold = base;

        // Under memory pressure, shrink prefill chunks to avoid large transient spikes.
        if kv_utilization > 0.95 {
            threshold = threshold.min((base / 4).max(32));
        } else if kv_utilization > 0.85 {
            threshold = threshold.min((base / 2).max(64));
        }

        // If decode is already active, avoid over-investing in prefill this step.
        if has_decode_demand {
            threshold = threshold.min((base / 2).max(64));
        }

        // Favor throughput for single-request, low-pressure execution.
        if self.waiting_count() <= 1 && self.running.len() <= 1 && kv_utilization < 0.50 {
            threshold = threshold.max(base).min(base.saturating_mul(2));
        }

        threshold.max(32)
    }

    fn update_dynamic_budget(&mut self) {
        let max_tokens = self.config.max_tokens_per_step.max(1);
        let min_tokens = self.config.min_tokens_per_step.min(max_tokens);
        if !self.config.enable_adaptive_batching {
            self.telemetry.dynamic_tokens_per_step = max_tokens;
            return;
        }

        let current = self.telemetry.dynamic_tokens_per_step;
        let step = (max_tokens / 10).max(1);
        let mut target = current;

        if self.telemetry.avg_ttft_ms > self.config.target_ttft_ms * 1.15 {
            target = (current + step).min(max_tokens);
        } else if self.telemetry.avg_decode_tpot_ms > self.config.target_decode_tpot_ms * 1.20 {
            target = current.saturating_sub(step).max(min_tokens);
        } else if current < max_tokens {
            target = (current + (step / 2).max(1)).min(max_tokens);
        }

        self.telemetry.dynamic_tokens_per_step = target;
    }

    /// Try to preempt running requests to free up the required number of blocks.
    /// Only preempts requests with lower priority than the requesting priority.
    /// Returns the list of preempted request IDs.
    fn try_preempt_for_blocks(
        &mut self,
        blocks_needed: usize,
        requesting_priority: Priority,
        kv_cache: &mut KVCacheManager,
    ) -> Vec<RequestId> {
        let mut preempted = Vec::new();
        let mut blocks_freed = 0;

        // Collect candidates for preemption (lower priority, sorted by priority then by tokens generated)
        let mut candidates: Vec<_> = self
            .running
            .iter()
            .filter(|(_, r)| {
                r.priority < requesting_priority && !r.paused && !r.block_ids.is_empty()
            })
            .map(|(id, r)| {
                (
                    id.clone(),
                    r.priority,
                    r.block_ids.len(),
                    r.num_tokens_generated,
                )
            })
            .collect();

        // Sort by priority (lowest first), then by tokens generated (lowest first to minimize wasted work)
        candidates.sort_by(|a, b| match a.1.cmp(&b.1) {
            std::cmp::Ordering::Equal => a.3.cmp(&b.3),
            ord => ord,
        });

        // Preempt until we have enough blocks
        for (request_id, _priority, num_blocks, _) in candidates {
            if blocks_freed >= blocks_needed {
                break;
            }

            // Mark as paused and free blocks while keeping decode progress metadata.
            if let Some(running) = self.running.get_mut(&request_id) {
                kv_cache.free(&request_id);
                blocks_freed += num_blocks;
                preempted.push(request_id.clone());
                running.block_ids.clear();
                running.paused = true;

                debug!(
                    "Preempted request {} (freed {} blocks, total freed: {})",
                    request_id, num_blocks, blocks_freed
                );
            }
        }

        if blocks_freed >= blocks_needed {
            debug!(
                "Successfully preempted {} requests, freed {} blocks (needed {})",
                preempted.len(),
                blocks_freed,
                blocks_needed
            );
        } else {
            debug!(
                "Could not free enough blocks: freed {} but needed {}",
                blocks_freed, blocks_needed
            );
        }

        preempted
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::TaskType;
    use super::*;
    use crate::models::chat_types::{ChatMessage, ChatRole};
    use std::time::Duration;

    fn tiny_preemption_scheduler() -> (Scheduler, KVCacheManager) {
        let config = SchedulerConfig {
            max_batch_size: 2,
            max_tokens_per_step: 8,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::Priority,
            enable_chunked_prefill: false,
            enable_preemption: true,
            enable_adaptive_batching: false,
            ..Default::default()
        };
        let scheduler = Scheduler::new(config);
        let kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 1,
            block_size: 2,
            ..Default::default()
        });
        (scheduler, kv_cache)
    }

    fn build_request(task_type: TaskType, id: &str, priority: Priority) -> EngineCoreRequest {
        let mut request = match task_type {
            TaskType::TTS => EngineCoreRequest::tts("hello world"),
            TaskType::ASR => EngineCoreRequest::asr("UklGRg=="),
            TaskType::Chat => EngineCoreRequest::chat(vec![ChatMessage {
                role: ChatRole::User,
                content: "hello world".to_string(),
            }]),
            TaskType::SpeechToSpeech => EngineCoreRequest::speech_to_speech("UklGRg=="),
        }
        .with_priority(priority);

        request.id = id.to_string();
        request.prompt_tokens = vec![1];
        request
    }

    #[test]
    fn test_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = Scheduler::new(config);
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.running_count(), 0);
    }

    #[test]
    fn test_adaptive_aging_can_promote_old_request() {
        let config = SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 32,
            policy: SchedulingPolicy::Priority,
            enable_adaptive_batching: true,
            priority_aging_ms: 100,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 128,
            block_size: 16,
            ..Default::default()
        });

        let old_id = "old-low".to_string();
        let fresh_id = "fresh-high".to_string();
        let old = EngineCoreRequest::tts("old request").with_priority(Priority::Low);
        let fresh = EngineCoreRequest::tts("new request").with_priority(Priority::High);

        let mut old = EngineCoreRequest {
            id: old_id.clone(),
            ..old
        };
        old.arrival_time = Instant::now() - Duration::from_secs(3);
        let fresh = EngineCoreRequest {
            id: fresh_id.clone(),
            ..fresh
        };

        scheduler.add_request(&old);
        scheduler.add_request(&fresh);
        if let Some(meta) = scheduler.requests.get_mut(&old_id) {
            meta.arrival_time = Instant::now() - Duration::from_secs(3);
        }

        let scheduled = scheduler.schedule(&mut kv_cache);
        assert_eq!(scheduled.prefill_requests.len(), 1);
        assert_eq!(scheduled.prefill_requests[0].request_id, old_id);
    }

    #[test]
    fn test_preemption_requeue_across_task_types() {
        let task_types = [
            TaskType::TTS,
            TaskType::ASR,
            TaskType::Chat,
            TaskType::SpeechToSpeech,
        ];

        for task_type in task_types {
            let (mut scheduler, mut kv_cache) = tiny_preemption_scheduler();
            let low_id = format!("low-{task_type:?}");
            let high_id = format!("high-{task_type:?}");
            let low = build_request(task_type, &low_id, Priority::Low);
            scheduler.add_request(&low);

            let first = scheduler.schedule(&mut kv_cache);
            assert_eq!(
                first.prefill_requests.len(),
                1,
                "expected initial prefill for {task_type:?}"
            );
            assert_eq!(first.prefill_requests[0].request_id, low_id);
            scheduler.update_after_step(&low_id, 1, 0, Vec::new(), 1.0);

            let high = build_request(task_type, &high_id, Priority::High);
            scheduler.add_request(&high);

            let second = scheduler.schedule(&mut kv_cache);
            assert!(
                second.preempted_requests.iter().any(|id| id == &low_id),
                "expected low-priority {task_type:?} request to be preempted"
            );
            assert_eq!(
                scheduler.get_status(&low_id),
                Some(RequestStatus::Running),
                "preempted {task_type:?} request should remain tracked for resume"
            );
            assert_eq!(
                scheduler.get_status(&high_id),
                Some(RequestStatus::Running),
                "high-priority {task_type:?} request should run after preemption"
            );

            scheduler.finish_request(&high_id, &mut kv_cache);

            let third = scheduler.schedule(&mut kv_cache);
            assert_eq!(
                third.decode_requests.len(),
                1,
                "preempted {task_type:?} request should resume decode from preserved state"
            );
            assert_eq!(third.decode_requests[0].request_id, low_id);
            assert!(!third.decode_requests[0].is_prefill);
        }
    }

    #[test]
    fn test_abort_running_request_across_task_types() {
        let task_types = [
            TaskType::TTS,
            TaskType::ASR,
            TaskType::Chat,
            TaskType::SpeechToSpeech,
        ];

        for task_type in task_types {
            let (mut scheduler, mut kv_cache) = tiny_preemption_scheduler();
            let request_id = format!("abort-{task_type:?}");
            let request = build_request(task_type, &request_id, Priority::Normal);
            scheduler.add_request(&request);

            let scheduled = scheduler.schedule(&mut kv_cache);
            assert_eq!(
                scheduled.prefill_requests.len(),
                1,
                "expected running request before abort for {task_type:?}"
            );
            assert_eq!(
                scheduler.get_status(&request_id),
                Some(RequestStatus::Running)
            );

            assert!(
                scheduler.abort_request(&request_id, &mut kv_cache),
                "abort should report running request removal for {task_type:?}"
            );
            assert!(
                !scheduler.has_request(&request_id),
                "aborted {task_type:?} request should be removed from scheduler metadata"
            );
            assert_eq!(
                scheduler.get_status(&request_id),
                None,
                "aborted {task_type:?} request must not remain queued/running"
            );

            let after_abort = scheduler.schedule(&mut kv_cache);
            assert!(
                !after_abort.has_work(),
                "no work should remain after aborting sole {task_type:?} request"
            );
        }
    }

    #[test]
    fn test_tts_auto_max_tokens_allows_decode_after_prefill() {
        let config = SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: false,
            enable_preemption: false,
            enable_adaptive_batching: false,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 32,
            block_size: 16,
            ..Default::default()
        });

        let request_id = "tts-auto-max".to_string();
        let mut request = EngineCoreRequest::tts("hello world");
        request.id = request_id.clone();
        request.prompt_tokens = vec![1];
        request.params.max_tokens = 0;
        scheduler.add_request(&request);

        let first = scheduler.schedule(&mut kv_cache);
        assert_eq!(first.prefill_requests.len(), 1);
        assert_eq!(first.prefill_requests[0].request_id, request_id);
        scheduler.update_after_step(&request_id, 1, 1, Vec::new(), 1.0);

        let second = scheduler.schedule(&mut kv_cache);
        assert_eq!(
            second.decode_requests.len(),
            1,
            "TTS auto max_tokens must still schedule decode"
        );
        assert_eq!(second.decode_requests[0].request_id, request_id);
    }

    #[test]
    fn test_non_tts_zero_max_tokens_gets_safe_default_budget() {
        let config = SchedulerConfig {
            max_batch_size: 1,
            max_tokens_per_step: 8,
            min_tokens_per_step: 1,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: false,
            enable_preemption: false,
            enable_adaptive_batching: false,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);
        let mut kv_cache = KVCacheManager::new(super::super::kv_cache::KVCacheConfig {
            max_blocks: 32,
            block_size: 16,
            ..Default::default()
        });

        let request_id = "chat-zero-max".to_string();
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "hello world".to_string(),
        }]);
        request.id = request_id.clone();
        request.prompt_tokens = vec![1];
        request.params.max_tokens = 0;
        scheduler.add_request(&request);

        let first = scheduler.schedule(&mut kv_cache);
        assert_eq!(first.prefill_requests.len(), 1);
        assert_eq!(first.prefill_requests[0].request_id, request_id);
        scheduler.update_after_step(&request_id, 1, 1, Vec::new(), 1.0);

        let second = scheduler.schedule(&mut kv_cache);
        assert_eq!(
            second.decode_requests.len(),
            1,
            "Non-TTS zero max_tokens should be normalized to a safe decode budget"
        );
        assert_eq!(second.decode_requests[0].request_id, request_id);
    }
}
