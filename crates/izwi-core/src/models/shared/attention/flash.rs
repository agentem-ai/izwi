//! Optional fused attention helpers.
//!
//! This module provides a single integration point for:
//! - CUDA FlashAttention2 via `candle-flash-attn` (when compiled with `flash-attn`)
//! - Metal fused SDPA via `candle_nn::ops::sdpa`
//!
//! Callers should treat these paths as opportunistic accelerations and always keep
//! a numerically equivalent fallback path.

use candle_core::{DType, Tensor};

use crate::error::Result;
use crate::models::shared::telemetry::{
    record_fused_attention_attempt, record_fused_attention_fallback,
    record_fused_attention_success, AttentionFallbackReason,
};

/// Runtime opt-in for fused attention paths.
pub fn flash_attention_requested() -> bool {
    std::env::var("IZWI_USE_FLASH_ATTENTION")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Whether the build includes CUDA FlashAttention2 support.
pub const fn flash_attention_compiled() -> bool {
    cfg!(feature = "flash-attn")
}

/// Runtime check used by models that wire Candle's `use_flash_attn` flag.
pub fn should_enable_flash_attention_v2(device: &candle_core::Device) -> bool {
    flash_attention_requested() && device.is_cuda() && flash_attention_compiled()
}

/// Try a fused self-attention kernel and return `None` when unsupported.
///
/// Input/output layout: `[batch, heads, seq, head_dim]`.
pub fn try_fused_self_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    head_dim: usize,
    causal: bool,
) -> Result<Option<Tensor>> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    record_fused_attention_attempt();
    let mut fallback_reason = AttentionFallbackReason::UnsupportedBackend;

    #[cfg(feature = "flash-attn")]
    {
        if should_enable_flash_attention_v2(q.device()) {
            if mask.is_none() {
                if dtype_supported_for_flash(q.dtype()) {
                    if q.dtype() == k.dtype() && k.dtype() == v.dtype() {
                        let q = q.transpose(1, 2)?.contiguous()?;
                        let k = k.transpose(1, 2)?.contiguous()?;
                        let v = v.transpose(1, 2)?.contiguous()?;
                        let flash_result =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                candle_flash_attn::flash_attn(&q, &k, &v, scale, causal)
                            }));
                        match flash_result {
                            Ok(Ok(out)) => {
                                record_fused_attention_success();
                                return Ok(Some(out.transpose(1, 2)?));
                            }
                            Ok(Err(_)) | Err(_) => {
                                fallback_reason = AttentionFallbackReason::FlashRuntimeError;
                            }
                        }
                    } else {
                        fallback_reason = AttentionFallbackReason::FlashDTypeMismatch;
                    }
                } else {
                    fallback_reason = AttentionFallbackReason::FlashDTypeUnsupported;
                }
            } else {
                fallback_reason = AttentionFallbackReason::FlashMaskUnsupported;
            }
        } else if q.device().is_cuda() {
            fallback_reason = AttentionFallbackReason::FlashNotRequested;
        }
    }

    #[cfg(not(feature = "flash-attn"))]
    {
        if q.device().is_cuda() {
            fallback_reason = if flash_attention_requested() {
                AttentionFallbackReason::FlashNotCompiled
            } else {
                AttentionFallbackReason::FlashNotRequested
            };
        }
    }

    if q.device().is_metal() {
        // Conservative Metal gate: avoid dispatching known high-risk SDPA shapes and
        // fall back to unfused attention directly.
        //
        // This prevents runtime kernel-launch failures from poisoning shared Metal locks
        // in long-lived server processes.
        if should_try_metal_sdpa(q, k, v, mask)? {
            let q_seq = q.dim(2)?;
            let use_f16_cast = should_use_metal_sdpa_f16_cast(q.dtype(), q_seq);
            let sdpa_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if use_f16_cast {
                    run_metal_sdpa_with_f16_inputs(q, k, v, mask, causal, scale)
                } else {
                    candle_nn::ops::sdpa(q, k, v, mask, causal, scale, 1.0)
                }
            }));
            match sdpa_result {
                Ok(Ok(out)) => {
                    record_fused_attention_success();
                    return Ok(Some(out));
                }
                Ok(Err(_)) | Err(_) => {
                    fallback_reason = AttentionFallbackReason::MetalSdpaRuntimeError;
                }
            }
        } else {
            fallback_reason = AttentionFallbackReason::UnsupportedBackend;
        }
    }

    record_fused_attention_fallback(fallback_reason);
    Ok(None)
}

/// Conservative preflight for Metal SDPA.
///
/// This gate mirrors Candle's shape support checks, while keeping F32 full-SDPA
/// behind a guarded cast route to avoid known threadgroup-memory failures.
fn should_try_metal_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<bool> {
    if mask.is_some() {
        return Ok(false);
    }

    let q_heads = q.dim(1)?;
    let kv_heads = k.dim(1)?;
    let v_heads = v.dim(1)?;
    let q_seq = q.dim(2)?;
    let k_seq = k.dim(2)?;
    let q_head = q.dim(3)?;
    let k_head = k.dim(3)?;

    if q.dtype() != k.dtype() || k.dtype() != v.dtype() {
        return Ok(false);
    }
    let dtype_supported = matches!(q.dtype(), DType::F16 | DType::BF16 | DType::F32);
    if !dtype_supported {
        return Ok(false);
    }

    if !metal_sdpa_shape_supported(q_heads, kv_heads, v_heads, q_seq, k_seq, q_head, k_head) {
        return Ok(false);
    }

    // F32 + full-SDPA prefill has triggered oversized threadgroup plans on some
    // Apple GPUs. We only enable this shape when the guarded F16-cast route is on.
    if q_seq > 8 && q.dtype() == DType::F32 && !metal_sdpa_f32_prefill_cast_enabled() {
        return Ok(false);
    }

    Ok(true)
}

fn metal_sdpa_shape_supported(
    q_heads: usize,
    kv_heads: usize,
    v_heads: usize,
    q_seq: usize,
    k_seq: usize,
    q_head: usize,
    k_head: usize,
) -> bool {
    if kv_heads == 0 {
        return false;
    }
    if q_head != k_head {
        return false;
    }
    if v_heads != kv_heads {
        return false;
    }
    if q_heads % kv_heads != 0 {
        return false;
    }
    if q_seq > k_seq {
        return false;
    }

    let supported_head_dim = matches!(q_head, 32 | 64 | 72 | 80 | 96 | 128 | 256);
    if !supported_head_dim {
        return false;
    }

    true
}

fn run_metal_sdpa_with_f16_inputs(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    scale: f32,
) -> candle_core::Result<Tensor> {
    let q_f16 = q.to_dtype(DType::F16)?;
    let k_f16 = k.to_dtype(DType::F16)?;
    let v_f16 = v.to_dtype(DType::F16)?;
    let out = candle_nn::ops::sdpa(&q_f16, &k_f16, &v_f16, mask, causal, scale, 1.0)?;
    out.to_dtype(q.dtype())
}

fn should_use_metal_sdpa_f16_cast(dtype: DType, q_seq: usize) -> bool {
    q_seq > 8 && dtype == DType::F32 && metal_sdpa_f32_prefill_cast_enabled()
}

fn metal_sdpa_f32_prefill_cast_enabled() -> bool {
    env_bool("IZWI_METAL_SDPA_F32_PREFILL_F16", true)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

#[cfg(feature = "flash-attn")]
#[inline]
fn dtype_supported_for_flash(dtype: candle_core::DType) -> bool {
    matches!(dtype, candle_core::DType::F16 | candle_core::DType::BF16)
}

#[cfg(test)]
mod tests {
    use super::{metal_sdpa_shape_supported, should_use_metal_sdpa_f16_cast};
    use candle_core::DType;

    #[test]
    fn metal_sdpa_shape_gate_accepts_supported_shapes() {
        assert!(metal_sdpa_shape_supported(32, 8, 8, 1, 128, 128, 128));
        assert!(metal_sdpa_shape_supported(32, 8, 8, 23, 23, 128, 128));
    }

    #[test]
    fn metal_sdpa_shape_gate_rejects_invalid_shapes() {
        assert!(!metal_sdpa_shape_supported(32, 8, 8, 24, 23, 128, 128));
        assert!(!metal_sdpa_shape_supported(30, 8, 8, 8, 8, 128, 128));
        assert!(!metal_sdpa_shape_supported(32, 8, 7, 8, 8, 128, 128));
        assert!(!metal_sdpa_shape_supported(32, 8, 8, 8, 8, 120, 120));
    }

    #[test]
    fn metal_sdpa_f16_cast_policy_only_applies_to_f32_prefill() {
        let _guard = crate::env_test_lock().lock().expect("env lock");

        std::env::remove_var("IZWI_METAL_SDPA_F32_PREFILL_F16");
        assert!(should_use_metal_sdpa_f16_cast(DType::F32, 23));
        assert!(!should_use_metal_sdpa_f16_cast(DType::F16, 23));
        assert!(!should_use_metal_sdpa_f16_cast(DType::F32, 8));

        std::env::set_var("IZWI_METAL_SDPA_F32_PREFILL_F16", "0");
        assert!(!should_use_metal_sdpa_f16_cast(DType::F32, 23));

        std::env::remove_var("IZWI_METAL_SDPA_F32_PREFILL_F16");
    }
}
