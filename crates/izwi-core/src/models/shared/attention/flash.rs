//! Optional fused attention helpers.
//!
//! This module provides a single integration point for:
//! - CUDA FlashAttention2 via `candle-flash-attn` (when compiled with `flash-attn`)
//! - Metal fused SDPA via `candle_nn::ops::sdpa`
//!
//! Callers should treat these paths as opportunistic accelerations and always keep
//! a numerically equivalent fallback path.

use candle_core::Tensor;

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
                        let flash_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                            || candle_flash_attn::flash_attn(&q, &k, &v, scale, causal),
                        ));
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
        if should_try_metal_sdpa(q, mask)? {
            let sdpa_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                candle_nn::ops::sdpa(q, k, v, mask, causal, scale, 1.0)
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
/// We intentionally restrict fused SDPA to short query lengths to avoid known
/// threadgroup-memory blowups on some Apple GPU families.
fn should_try_metal_sdpa(q: &Tensor, mask: Option<&Tensor>) -> Result<bool> {
    if mask.is_some() {
        return Ok(false);
    }

    let head_dim = q.dim(3)?;
    let supported_head_dim = matches!(head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256);
    if !supported_head_dim {
        return Ok(false);
    }

    let q_seq = q.dim(2)?;
    // Safe-by-default policy: keep fused SDPA on decode-like shapes only.
    // Longer prefill sequences use the unfused path for stability.
    Ok(q_seq <= 8)
}

#[cfg(feature = "flash-attn")]
#[inline]
fn dtype_supported_for_flash(dtype: candle_core::DType) -> bool {
    matches!(dtype, candle_core::DType::F16 | candle_core::DType::BF16)
}
