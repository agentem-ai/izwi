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

    #[cfg(feature = "flash-attn")]
    {
        if should_enable_flash_attention_v2(q.device())
            && mask.is_none()
            && dtype_supported_for_flash(q.dtype())
            && q.dtype() == k.dtype()
            && k.dtype() == v.dtype()
        {
            let q = q.transpose(1, 2)?.contiguous()?;
            let k = k.transpose(1, 2)?.contiguous()?;
            let v = v.transpose(1, 2)?.contiguous()?;
            if let Ok(out) = candle_flash_attn::flash_attn(&q, &k, &v, scale, causal) {
                return Ok(Some(out.transpose(1, 2)?));
            }
        }
    }

    if q.device().is_metal() {
        if let Ok(out) = candle_nn::ops::sdpa(q, k, v, mask, causal, scale, 1.0) {
            return Ok(Some(out));
        }
    }

    Ok(None)
}

#[cfg(feature = "flash-attn")]
#[inline]
fn dtype_supported_for_flash(dtype: candle_core::DType) -> bool {
    matches!(dtype, candle_core::DType::F16 | candle_core::DType::BF16)
}
