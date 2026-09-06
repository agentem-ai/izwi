//! Sampling keeps probability rows on the device. Only bounded token/status
//! results cross the device boundary. The caller owns history and RNG commit.
use super::{ChatGenerationConfig, SimpleRng};
use crate::error::{Error, Result};
use crate::kernels::cuda::sampling::{self, SamplingParams};
use crate::models::shared::speculative_sampling::SpeculativeVerification;
use candle_core::{DType, IndexOp, Tensor};
use std::collections::BTreeMap;
use std::ops::ControlFlow;

fn unit(rng: &mut SimpleRng) -> f32 {
    (rng.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
}

pub(super) fn greedy(logits: &Tensor, vocab: usize) -> Result<Vec<u32>> {
    let (_, width) = logits.dims2()?;
    if vocab == 0 || width < vocab {
        return Err(Error::InvalidInput(
            "invalid device sampling vocabulary".into(),
        ));
    }
    let results = sampling::greedy_rows(&logits.narrow(1, 0, vocab)?)?.to_vec2::<u32>()?;
    results
        .into_iter()
        .map(|row| {
            if row[1] != 1 {
                Err(Error::InferenceError(
                    "No finite Qwen3.8 logits to sample".into(),
                ))
            } else {
                Ok(row[0])
            }
        })
        .collect()
}

pub(super) fn distribution(
    logits: &Tensor,
    vocab: usize,
    config: &ChatGenerationConfig,
    history: &[u32],
) -> Result<Tensor> {
    let (_, width) = logits.dims2()?;
    if vocab == 0 || width < vocab {
        return Err(Error::InvalidInput(
            "invalid device sampling vocabulary".into(),
        ));
    }
    let mut frequencies = BTreeMap::<u32, u32>::new();
    for &token in history {
        if (token as usize) < vocab {
            let count = frequencies.entry(token).or_default();
            *count = count.saturating_add(1);
        }
    }
    let mut counts = Tensor::zeros(vocab, DType::U32, logits.device())?;
    if !frequencies.is_empty() {
        let indices = Tensor::from_vec(
            frequencies.keys().copied().collect::<Vec<_>>(),
            frequencies.len(),
            logits.device(),
        )?;
        let values = Tensor::from_vec(
            frequencies.values().copied().collect::<Vec<_>>(),
            frequencies.len(),
            logits.device(),
        )?;
        counts = counts.index_add(&indices, &values, 0)?;
    }
    let params = SamplingParams {
        temperature: config.temperature,
        repetition_penalty: config.repetition_penalty,
        presence_penalty: config.presence_penalty,
        frequency_penalty: 0.0,
        top_k: config.top_k,
        top_p: config.top_p,
        min_p: 0.0,
    };
    sampling::distributions(&logits.narrow(1, 0, vocab)?, &counts.unsqueeze(0)?, &params)
        .map_err(Error::from)
}

struct SamplingFailure {
    error: Error,
    no_finite_logits: bool,
}

impl SamplingFailure {
    fn into_checked<T>(self) -> Result<ControlFlow<Error, T>> {
        if self.no_finite_logits {
            Ok(ControlFlow::Break(self.error))
        } else {
            Err(self.error)
        }
    }
}

fn strict<T>(result: ControlFlow<Error, T>) -> Result<T> {
    match result {
        ControlFlow::Continue(value) => Ok(value),
        ControlFlow::Break(error) => Err(error),
    }
}

fn sampling_failure(phase: &str, logits: &Tensor, vocab: usize) -> SamplingFailure {
    // Failure-only diagnostics: inspect the sampled vocabulary on-device and
    // transfer three scalar counts, never logits or token data. Widen masks
    // before summing so full-vocabulary counts cannot overflow a U8 mask.
    let counts = (|| -> candle_core::Result<Vec<u32>> {
        let logits = logits
            .flatten_all()?
            .narrow(0, 0, vocab)?
            .to_dtype(DType::F32)?;
        let nan = logits.ne(&logits)?.to_dtype(DType::U32)?.sum_all()?;
        let positive_infinity = logits.eq(f32::INFINITY)?.to_dtype(DType::U32)?.sum_all()?;
        let negative_infinity = logits
            .eq(f32::NEG_INFINITY)?
            .to_dtype(DType::U32)?
            .sum_all()?;
        Tensor::stack(&[nan, positive_infinity, negative_infinity], 0)?.to_vec1::<u32>()
    })();
    let (detail, no_finite_logits) = match counts {
        Ok(counts) => {
            let nan = counts[0] as usize;
            let positive_infinity = counts[1] as usize;
            let negative_infinity = counts[2] as usize;
            let nonfinite = nan + positive_infinity + negative_infinity;
            let finite = vocab.saturating_sub(nonfinite);
            (
                format!(
                    "logits={vocab}, finite={finite}, nonfinite={nonfinite}, \
                     nan={nan}, pos_inf={positive_infinity}, neg_inf={negative_infinity}"
                ),
                vocab != 0 && nonfinite == vocab,
            )
        }
        // A diagnostic/backend failure must not replace the original sampling
        // failure or expose arbitrary backend error text in the statistics.
        Err(_) => (format!("logits={vocab}, counts=unavailable"), false),
    };
    SamplingFailure {
        error: Error::InferenceError(format!(
            "No finite Qwen3.8 sampling distribution (phase={phase}, {detail})"
        )),
        no_finite_logits,
    }
}

fn sample_at(
    probabilities: &Tensor,
    uniform: f32,
    phase: &str,
    logits: &Tensor,
    vocab: usize,
) -> Result<u32> {
    strict(sample_at_checked(
        probabilities,
        uniform,
        phase,
        logits,
        vocab,
    )?)
}

fn sample_at_checked(
    probabilities: &Tensor,
    uniform: f32,
    phase: &str,
    logits: &Tensor,
    vocab: usize,
) -> Result<ControlFlow<Error, u32>> {
    let uniforms = Tensor::from_vec(vec![uniform], 1, probabilities.device())?;
    let result = sampling::sample_rows(probabilities, &uniforms)?.to_vec2::<u32>()?;
    if result[0][1] != 1 {
        return sampling_failure(phase, logits, vocab).into_checked();
    }
    Ok(ControlFlow::Continue(result[0][0]))
}

// Keep the strict API for callers that cannot abandon a proposal and for tests.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn propose(
    logits: &Tensor,
    vocab: usize,
    config: &ChatGenerationConfig,
    history: &mut Vec<u32>,
    rng: &mut SimpleRng,
) -> Result<(u32, Tensor)> {
    strict(propose_or_abort(logits, vocab, config, history, rng)?)
}

/// Abort an optional proposal only when invalid status is backed by a
/// successfully counted, nonempty vocabulary containing no finite logits.
pub(super) fn propose_or_abort(
    logits: &Tensor,
    vocab: usize,
    config: &ChatGenerationConfig,
    history: &mut Vec<u32>,
    rng: &mut SimpleRng,
) -> Result<ControlFlow<Error, (u32, Tensor)>> {
    let probabilities = distribution(logits, vocab, config, history)?;
    let mut staged = rng.clone();
    let token = match sample_at_checked(&probabilities, unit(&mut staged), "draft", logits, vocab)?
    {
        ControlFlow::Continue(token) => token,
        ControlFlow::Break(error) => return Ok(ControlFlow::Break(error)),
    };
    history.push(token);
    *rng = staged;
    Ok(ControlFlow::Continue((token, probabilities)))
}

pub(super) fn sample(
    logits: &Tensor,
    vocab: usize,
    config: &ChatGenerationConfig,
    history: &[u32],
    rng: &mut SimpleRng,
) -> Result<u32> {
    strict(sample_or_abort(
        logits, vocab, config, history, rng, "target",
    )?)
}

/// Like `sample`, with failure-only evidence allowing the caller to abandon an
/// optional draft. `Break` and `Err` both leave the RNG unchanged.
pub(super) fn sample_or_abort(
    logits: &Tensor,
    vocab: usize,
    config: &ChatGenerationConfig,
    history: &[u32],
    rng: &mut SimpleRng,
    phase: &str,
) -> Result<ControlFlow<Error, u32>> {
    if config.temperature <= 1e-5
        && config.repetition_penalty <= 1.0
        && config.presence_penalty.abs() <= f32::EPSILON
    {
        let logits = logits.reshape((1, ()))?;
        if vocab == 0 || logits.dim(1)? < vocab {
            return Err(Error::InvalidInput(
                "invalid device sampling vocabulary".into(),
            ));
        }
        let result = sampling::greedy_rows(&logits.narrow(1, 0, vocab)?)?.to_vec2::<u32>()?;
        if result[0][1] != 1 {
            return sampling_failure(phase, &logits, vocab).into_checked();
        }
        return Ok(ControlFlow::Continue(result[0][0]));
    }
    let probabilities = distribution(&logits.reshape((1, ()))?, vocab, config, history)?;
    let mut staged = rng.clone();
    // Greedy distributions are one-hot and must not consume a draw.
    let uniform = if config.temperature <= 1e-5 {
        0.0
    } else {
        staged.next_f32()
    };
    let token = match sample_at_checked(&probabilities, uniform, phase, logits, vocab)? {
        ControlFlow::Continue(token) => token,
        ControlFlow::Break(error) => return Ok(ControlFlow::Break(error)),
    };
    *rng = staged;
    Ok(ControlFlow::Continue(token))
}

pub(super) fn verify_greedy(
    drafts: &[u32],
    target_logits: &Tensor,
    vocab: usize,
    config: &ChatGenerationConfig,
    history: &mut Vec<u32>,
) -> Result<SpeculativeVerification> {
    let mut position_history = history.clone();
    let mut probabilities = Vec::with_capacity(drafts.len() + 1);
    for position in 0..=drafts.len() {
        probabilities.push(distribution(
            &target_logits.i((0, position))?.unsqueeze(0)?,
            vocab,
            config,
            &position_history,
        )?);
        if position < drafts.len() {
            position_history.push(drafts[position]);
        }
    }
    let probabilities = Tensor::cat(&probabilities, 0)?;
    let statuses = sampling::sample_rows(
        &probabilities,
        &Tensor::zeros(drafts.len() + 1, DType::F32, target_logits.device())?,
    )?
    .to_vec2::<u32>()?;
    let tokens = statuses
        .into_iter()
        .map(|row| {
            if row[1] == 1 {
                Ok(row[0])
            } else {
                Err(Error::InferenceError(
                    "No finite Qwen3.8 greedy distribution".into(),
                ))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    crate::models::shared::speculative_sampling::verify_greedy_token_prefix(
        drafts, &tokens, history,
    )
}

pub(super) fn verify(
    drafts: &[u32],
    q: &[Tensor],
    target_logits: &Tensor,
    vocab: usize,
    config: &ChatGenerationConfig,
    history: &mut Vec<u32>,
    rng: &mut SimpleRng,
) -> Result<SpeculativeVerification> {
    if drafts.is_empty() || drafts.len() != q.len() || target_logits.dim(1)? != drafts.len() + 1 {
        return Err(Error::InvalidInput(
            "invalid device speculative block".into(),
        ));
    }
    let mut staged_history = history.clone();
    let mut p = Vec::with_capacity(drafts.len() + 1);
    for position in 0..=drafts.len() {
        p.push(distribution(
            &target_logits.i((0, position))?.unsqueeze(0)?,
            vocab,
            config,
            &staged_history,
        )?);
        if position < drafts.len() {
            staged_history.push(drafts[position]);
        }
    }
    let mut draws_rng = rng.clone();
    let draws = (0..=drafts.len())
        .map(|_| unit(&mut draws_rng))
        .collect::<Vec<_>>();
    let uniforms = draws
        .windows(2)
        .flat_map(|pair| pair.iter().copied())
        .collect::<Vec<_>>();
    let device = target_logits.device();
    let status = sampling::verify_rows(
        &Tensor::cat(&p[..drafts.len()], 0)?,
        &Tensor::cat(q, 0)?,
        &Tensor::from_slice(drafts, drafts.len(), device)?,
        &Tensor::from_vec(uniforms, (drafts.len(), 2), device)?,
    )?
    .to_vec2::<u32>()?;
    let accepted = status
        .iter()
        .position(|row| row[0] == 0)
        .unwrap_or(drafts.len());
    let inspected = (accepted + 1).min(drafts.len());
    if status[..inspected].iter().any(|row| row[2] != 1) {
        return Err(Error::InferenceError(
            "invalid device speculative probabilities".into(),
        ));
    }
    let mut emitted_tokens = drafts[..accepted].to_vec();
    let consumed_draws = if accepted < drafts.len() {
        emitted_tokens.push(status[accepted][1]);
        accepted + 2
    } else {
        emitted_tokens.push(sample_at(
            &p[drafts.len()],
            draws[drafts.len()],
            "bonus",
            &target_logits.i((0, drafts.len()))?,
            vocab,
        )?);
        drafts.len() + 1
    };
    let mut staged_rng = rng.clone();
    for _ in 0..consumed_draws {
        unit(&mut staged_rng);
    }
    history.extend_from_slice(&emitted_tokens);
    *rng = staged_rng;
    Ok(SpeculativeVerification {
        emitted_tokens,
        accepted_draft_tokens: accepted,
        draft_tokens: drafts.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::shared::speculative_sampling::{
        propose_speculative_draft, verify_speculative_proposals,
    };
    use candle_core::Device;

    #[test]
    fn optional_nan_sampling_aborts_without_committing_rng_or_history() {
        let logits = Tensor::full(f32::NAN, (1, 3), &Device::Cpu).unwrap();
        for (temperature, repetition_penalty) in [(0.0, 1.0), (0.0, 1.1), (0.8, 1.0)] {
            let config = ChatGenerationConfig {
                temperature,
                repetition_penalty,
                presence_penalty: 0.0,
                ..Default::default()
            };
            let mut rng = SimpleRng::new(42);
            let before = rng.state;
            let mut history = vec![1];
            let outcome =
                sample_or_abort(&logits, 3, &config, &history, &mut rng, "draft").unwrap();
            let ControlFlow::Break(error) = outcome else {
                panic!("all-NaN optional sample must abort");
            };
            assert!(error.to_string().contains("phase=draft"));
            assert!(error.to_string().contains("finite=0, nonfinite=3, nan=3"));
            assert_eq!(rng.state, before);
            assert_eq!(history, vec![1]);

            assert!(matches!(
                propose_or_abort(&logits, 3, &config, &mut history, &mut rng).unwrap(),
                ControlFlow::Break(_)
            ));
            assert_eq!(rng.state, before);
            assert_eq!(history, vec![1]);

            // The strict target and proposal APIs still reject the same input.
            assert!(sample(&logits, 3, &config, &history, &mut rng).is_err());
            assert!(propose(&logits, 3, &config, &mut history, &mut rng).is_err());
            assert_eq!(rng.state, before);
            assert_eq!(history, vec![1]);
        }
    }

    #[test]
    fn checked_invalid_status_requires_proven_nonempty_all_nonfinite_logits() {
        let probabilities = Tensor::zeros((1, 3), DType::F32, &Device::Cpu).unwrap();
        // Even one finite logit makes invalid probability mass a strict error.
        for values in [[1.0f32, 2.0, 3.0], [1.0, f32::NAN, f32::INFINITY]] {
            let logits = Tensor::from_slice(&values, 3, &Device::Cpu).unwrap();
            assert!(sample_at_checked(&probabilities, 0.5, "draft", &logits, 3).is_err());
        }
        let short_logits = Tensor::full(f32::NAN, 1, &Device::Cpu).unwrap();
        let error = sample_at_checked(&probabilities, 0.5, "draft", &short_logits, 3).unwrap_err();
        assert!(error.to_string().contains("counts=unavailable"));
        assert!(sample_at_checked(&probabilities, 0.5, "draft", &short_logits, 0).is_err());

        let nonfinite = Tensor::from_slice(
            &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            3,
            &Device::Cpu,
        )
        .unwrap();
        assert!(matches!(
            sample_at_checked(&probabilities, 0.5, "draft", &nonfinite, 3).unwrap(),
            ControlFlow::Break(_)
        ));
        // A primitive API error must not become recovery, even for all-NaN input.
        let malformed = probabilities.to_dtype(DType::F64).unwrap();
        assert!(sample_at_checked(&malformed, 0.5, "draft", &nonfinite, 3).is_err());
    }

    #[test]
    fn optional_sampling_api_errors_remain_strict_and_transactional() {
        let logits = Tensor::full(f32::NAN, (1, 3), &Device::Cpu).unwrap();
        let config = ChatGenerationConfig {
            temperature: f32::NAN,
            ..Default::default()
        };
        let mut history = vec![1];
        let mut rng = SimpleRng::new(42);
        let before = rng.state;
        assert!(sample_or_abort(&logits, 3, &config, &history, &mut rng, "draft").is_err());
        assert!(propose_or_abort(&logits, 3, &config, &mut history, &mut rng).is_err());
        assert_eq!(rng.state, before);
        assert_eq!(history, vec![1]);
    }

    #[test]
    fn optional_sampling_preserves_healthy_draw_and_history_contracts() {
        let logits = Tensor::from_slice(&[2.0f32, 2.0, -1.0], (1, 3), &Device::Cpu).unwrap();
        for (temperature, repetition_penalty) in [(0.0, 1.0), (0.0, 1.1), (0.8, 1.0)] {
            let config = ChatGenerationConfig {
                temperature,
                repetition_penalty,
                presence_penalty: 0.0,
                ..Default::default()
            };
            let history = vec![1];
            let mut rng = SimpleRng::new(42);
            let mut expected_rng = rng.clone();
            let expected = if temperature <= 1e-5 && repetition_penalty <= 1.0 {
                greedy(&logits, 3).unwrap()[0]
            } else {
                let probabilities = distribution(&logits, 3, &config, &history).unwrap();
                let draw = if temperature <= 1e-5 {
                    0.0
                } else {
                    expected_rng.next_f32()
                };
                sample_at(&probabilities, draw, "draft", &logits, 3).unwrap()
            };
            let outcome =
                sample_or_abort(&logits, 3, &config, &history, &mut rng, "draft").unwrap();
            assert!(matches!(outcome, ControlFlow::Continue(token) if token == expected));
            assert_eq!(rng.state, expected_rng.state);
            assert_eq!(history, vec![1]);

            let mut history = history;
            let probabilities = distribution(&logits, 3, &config, &history).unwrap();
            let expected =
                sample_at(&probabilities, unit(&mut expected_rng), "draft", &logits, 3).unwrap();
            let outcome = propose_or_abort(&logits, 3, &config, &mut history, &mut rng).unwrap();
            let ControlFlow::Continue((token, actual_probabilities)) = outcome else {
                panic!("healthy proposal must continue");
            };
            assert_eq!(token, expected);
            assert_eq!(history, vec![1, token]);
            assert_eq!(rng.state, expected_rng.state);
            assert_eq!(
                actual_probabilities.to_vec2::<f32>().unwrap(),
                probabilities.to_vec2::<f32>().unwrap()
            );
        }
    }

    #[test]
    fn device_math_matches_shared_proposal_and_pq_verifier_with_penalties_and_rng_commit() {
        let mut accepted_lengths = [false; 3];
        for seed in 1..=192 {
            let config = ChatGenerationConfig {
                temperature: 0.8,
                top_k: 4,
                top_p: 0.92,
                repetition_penalty: 1.15,
                presence_penalty: 0.17,
                seed,
                ..Default::default()
            };
            let initial_history = vec![1, 1, 3];
            let mut device_history = initial_history.clone();
            let mut shared_history = initial_history.clone();
            let mut device_rng = SimpleRng::new(seed);
            let mut shared_rng = device_rng.clone();
            let mut q = Vec::new();
            let mut tokens = Vec::new();
            let mut proposals = Vec::new();
            for logits in [[1.2f32, -0.3, 0.8, 0.1, -2.0], [-0.2, 0.7, 0.9, 1.1, -1.0]] {
                let tensor = Tensor::from_slice(&logits, (1, 5), &Device::Cpu).unwrap();
                let (token, probabilities) =
                    propose(&tensor, 5, &config, &mut device_history, &mut device_rng).unwrap();
                let shared = propose_speculative_draft(
                    &logits,
                    &config,
                    &mut shared_history,
                    &mut shared_rng,
                )
                .unwrap();
                assert_eq!(token, shared.token_id, "proposal seed={seed}");
                assert_eq!(device_rng.state, shared_rng.state);
                tokens.push(token);
                q.push(probabilities);
                proposals.push(shared);
            }
            assert_eq!(device_history, shared_history);
            let rows = vec![
                vec![0.8f32, 1.1, 0.3, -0.5, -2.0],
                vec![1.0, -0.2, 0.7, 0.9, -1.0],
                vec![0.4, 0.8, -0.3, 0.9, -1.0],
            ];
            let target = Tensor::from_vec(
                rows.iter().flatten().copied().collect(),
                (1, 3, 5),
                &Device::Cpu,
            )
            .unwrap();
            let mut device_history = initial_history.clone();
            let mut shared_history = initial_history;
            let actual = verify(
                &tokens,
                &q,
                &target,
                5,
                &config,
                &mut device_history,
                &mut device_rng,
            )
            .unwrap();
            let expected = verify_speculative_proposals(
                &proposals,
                &rows,
                &config,
                &mut shared_history,
                &mut shared_rng,
            )
            .unwrap();
            assert_eq!(actual, expected, "verification seed={seed}");
            assert_eq!(device_history, shared_history);
            assert_eq!(
                device_rng.state, shared_rng.state,
                "only used acceptance/residual/bonus draws commit"
            );
            accepted_lengths[actual.accepted_draft_tokens] = true;
        }
        assert_eq!(accepted_lengths, [true; 3]);
    }

    #[test]
    fn failed_distribution_is_transactional_and_greedy_does_not_consume_rng() {
        let config = ChatGenerationConfig::default();
        let mut rng = SimpleRng::new(42);
        let before = rng.state;
        let logits = Tensor::from_slice(&[2f32, 2.0, -1.0], 3, &Device::Cpu).unwrap();
        assert_eq!(sample(&logits, 3, &config, &[], &mut rng).unwrap(), 0);
        assert_eq!(rng.state, before);
        let bad = Tensor::from_slice(&[f32::NAN, f32::INFINITY], (1, 2), &Device::Cpu).unwrap();
        let mut history = vec![1];
        let error = propose(&bad, 2, &config, &mut history, &mut rng)
            .unwrap_err()
            .to_string();
        assert!(error.contains("No finite Qwen3.8 sampling distribution (phase=draft"));
        assert!(error.contains("finite=0, nonfinite=2, nan=1, pos_inf=1, neg_inf=0"));
        assert_eq!(history, vec![1]);
        assert_eq!(rng.state, before);

        let stochastic = ChatGenerationConfig {
            temperature: 0.8,
            ..config
        };
        let error = sample(&bad, 2, &stochastic, &history, &mut rng)
            .unwrap_err()
            .to_string();
        assert!(error.contains("No finite Qwen3.8 sampling distribution (phase=target"));
        assert!(error.contains("finite=0, nonfinite=2, nan=1, pos_inf=1, neg_inf=0"));
        assert_eq!(history, vec![1]);
        assert_eq!(rng.state, before);
    }

    #[test]
    fn invalid_status_reports_finite_full_vocabulary_without_counting_padding() {
        let vocab = 248_320;
        let probabilities = Tensor::zeros((1, vocab), DType::F32, &Device::Cpu).unwrap();
        let mut values = vec![1.0f32; vocab];
        values.push(f32::NAN);
        let logits = Tensor::from_vec(values, (1, vocab + 1), &Device::Cpu).unwrap();
        let error = sample_at(&probabilities, 0.5, "target", &logits, vocab)
            .unwrap_err()
            .to_string();
        assert!(error.contains("No finite Qwen3.8 sampling distribution (phase=target"));
        assert!(error
            .contains("logits=248320, finite=248320, nonfinite=0, nan=0, pos_inf=0, neg_inf=0"));

        // The actual reduced masks must also hold counts above 255.
        let logits = Tensor::full(f32::NAN, (1, vocab), &Device::Cpu).unwrap();
        let error = sample_at(&probabilities, 0.5, "draft", &logits, vocab)
            .unwrap_err()
            .to_string();
        assert!(error.contains(
            "logits=248320, finite=0, nonfinite=248320, nan=248320, pos_inf=0, neg_inf=0"
        ));
    }

    #[test]
    fn invalid_status_counts_each_nonfinite_kind_and_preserves_failure_if_counts_fail() {
        let probabilities = Tensor::zeros((1, 5), DType::F32, &Device::Cpu).unwrap();
        let logits = Tensor::from_slice(
            &[7.0f32, -2.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            5,
            &Device::Cpu,
        )
        .unwrap();
        let error = sample_at(&probabilities, 0.5, "draft", &logits, 5)
            .unwrap_err()
            .to_string();
        assert!(error.contains("logits=5, finite=2, nonfinite=3, nan=1, pos_inf=1, neg_inf=1"));

        let short_logits = logits.narrow(0, 0, 1).unwrap();
        let error = sample_at(&probabilities, 0.5, "draft", &short_logits, 5)
            .unwrap_err()
            .to_string();
        assert!(error.contains(
            "No finite Qwen3.8 sampling distribution (phase=draft, logits=5, counts=unavailable)"
        ));
    }

    #[test]
    fn invalid_bonus_reports_its_logit_row_without_committing_rng_or_history() {
        let config = ChatGenerationConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            ..Default::default()
        };
        let target = Tensor::from_slice(
            &[0.0f32, -1000.0, f32::NAN, f32::NEG_INFINITY],
            (1, 2, 2),
            &Device::Cpu,
        )
        .unwrap();
        let q = Tensor::from_slice(&[1.0f32, 0.0], (1, 2), &Device::Cpu).unwrap();
        let mut history = vec![1];
        let mut rng = SimpleRng::new(42);
        let before = rng.state;
        let error = verify(&[0], &[q], &target, 2, &config, &mut history, &mut rng)
            .unwrap_err()
            .to_string();
        assert!(error.contains("No finite Qwen3.8 sampling distribution (phase=bonus"));
        assert!(error.contains("logits=2, finite=0, nonfinite=2, nan=1, pos_inf=0, neg_inf=1"));
        assert_eq!(history, vec![1]);
        assert_eq!(rng.state, before);
    }
}
