//! Shared paged-KV helpers for decode-time attention.

use candle_core::{Tensor, D};

use crate::error::{Error, Result};

/// Default KV page size used when model-specific config is unavailable.
pub const DEFAULT_KV_PAGE_SIZE: usize = 64;

/// Resolve default page size, optionally overridden by env.
pub fn default_kv_page_size() -> usize {
    std::env::var("IZWI_KV_PAGE_SIZE")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_KV_PAGE_SIZE)
}

/// Append a `[batch, seq, heads, dim]` tensor into fixed-size pages along `seq`.
///
/// The last existing page is filled first (if not full), then new pages are pushed.
pub fn append_to_pages(page_size: usize, pages: &mut Vec<Tensor>, append: &Tensor) -> Result<()> {
    if page_size == 0 {
        return Err(Error::InvalidInput(
            "KV page size must be greater than zero".to_string(),
        ));
    }
    let seq_len = append.dim(1)?;
    if seq_len == 0 {
        return Ok(());
    }

    let mut cursor = 0usize;
    while cursor < seq_len {
        let mut consumed = false;
        if let Some(last) = pages.last_mut() {
            let last_len = last.dim(1)?;
            if last_len < page_size {
                let take = (page_size - last_len).min(seq_len - cursor);
                let chunk = append.narrow(1, cursor, take)?;
                *last = Tensor::cat(&[&*last, &chunk], 1)?;
                cursor += take;
                consumed = true;
            }
        }

        if consumed {
            continue;
        }

        let take = page_size.min(seq_len - cursor);
        let chunk = append.narrow(1, cursor, take)?;
        pages.push(chunk);
        cursor += take;
    }
    Ok(())
}

/// Materialize paged tensors into a single contiguous `[batch, total_seq, heads, dim]` tensor.
pub fn materialize_pages(pages: &[Tensor]) -> Result<Tensor> {
    if pages.is_empty() {
        return Err(Error::InferenceError(
            "Attempted to materialize empty KV pages".to_string(),
        ));
    }
    if pages.len() == 1 {
        return Ok(pages[0].clone());
    }
    let refs: Vec<&Tensor> = pages.iter().collect();
    Tensor::cat(&refs, 1).map_err(Error::from)
}

/// Compute exact single-token attention over paged K/V without materializing full K/V.
///
/// `q` is `[batch, 1, heads, head_dim]` and page tensors are `[batch, page_seq, heads, head_dim]`.
/// Returns `[batch, 1, heads, head_dim]`.
pub fn paged_decode_attention(
    q: &Tensor,
    k_pages: &[Tensor],
    v_pages: &[Tensor],
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    if k_pages.is_empty() || v_pages.is_empty() || k_pages.len() != v_pages.len() {
        return Err(Error::InferenceError(
            "Paged decode attention received invalid KV pages".to_string(),
        ));
    }
    let bsz = q.dim(0)?;
    let q_len = q.dim(1)?;
    if q_len != 1 {
        return Err(Error::InvalidInput(format!(
            "Paged decode attention expects q_len=1, got {}",
            q_len
        )));
    }

    let q = q
        .transpose(1, 2)?
        .reshape((bsz * num_heads, q_len, head_dim))?;
    let scale = (head_dim as f64).sqrt();
    let scale_t = Tensor::from_vec(vec![scale as f32], (1,), q.device())?.to_dtype(q.dtype())?;

    let mut running_max: Option<Tensor> = None; // [bh, 1, 1]
    let mut running_sum: Option<Tensor> = None; // [bh, 1, 1]
    let mut running_out: Option<Tensor> = None; // [bh, 1, d]

    for (k_page, v_page) in k_pages.iter().zip(v_pages.iter()) {
        let page_len = k_page.dim(1)?;
        if page_len == 0 {
            continue;
        }

        let k = k_page
            .transpose(1, 2)?
            .reshape((bsz * num_heads, page_len, head_dim))?;
        let v = v_page
            .transpose(1, 2)?
            .reshape((bsz * num_heads, page_len, head_dim))?;

        let mut scores = q.matmul(&k.transpose(1, 2)?)?;
        scores = scores.broadcast_div(&scale_t)?;

        let page_max = scores.max_keepdim(D::Minus1)?;
        let exp_scores = scores.broadcast_sub(&page_max)?.exp()?;
        let page_sum = exp_scores.sum_keepdim(D::Minus1)?;
        let page_out = exp_scores.matmul(&v)?;

        match (&running_max, &running_sum, &running_out) {
            (None, None, None) => {
                running_max = Some(page_max);
                running_sum = Some(page_sum);
                running_out = Some(page_out);
            }
            (Some(cur_max), Some(cur_sum), Some(cur_out)) => {
                let new_max = cur_max.broadcast_maximum(&page_max)?;
                let cur_scale = cur_max.broadcast_sub(&new_max)?.exp()?;
                let page_scale = page_max.broadcast_sub(&new_max)?.exp()?;

                let new_sum = cur_sum
                    .broadcast_mul(&cur_scale)?
                    .broadcast_add(&page_sum.broadcast_mul(&page_scale)?)?;
                let new_out = cur_out
                    .broadcast_mul(&cur_scale)?
                    .broadcast_add(&page_out.broadcast_mul(&page_scale)?)?;

                running_max = Some(new_max);
                running_sum = Some(new_sum);
                running_out = Some(new_out);
            }
            _ => {
                return Err(Error::InferenceError(
                    "Paged decode attention entered inconsistent running state".to_string(),
                ));
            }
        }
    }

    let running_sum = running_sum.ok_or_else(|| {
        Error::InferenceError("Paged decode attention produced no valid page sum".to_string())
    })?;
    let running_out = running_out.ok_or_else(|| {
        Error::InferenceError("Paged decode attention produced no valid page output".to_string())
    })?;

    let out = running_out.broadcast_div(&running_sum)?;
    out.reshape((bsz, num_heads, q_len, head_dim))?
        .transpose(1, 2)
        .map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::ops;

    #[test]
    fn test_append_to_pages_respects_page_size() {
        let device = Device::Cpu;
        let mut pages = Vec::new();
        let tensor = Tensor::randn(0.0f32, 1.0f32, (1, 10, 2, 4), &device).unwrap();
        append_to_pages(4, &mut pages, &tensor).unwrap();
        assert_eq!(pages.len(), 3);
        assert_eq!(pages[0].dim(1).unwrap(), 4);
        assert_eq!(pages[1].dim(1).unwrap(), 4);
        assert_eq!(pages[2].dim(1).unwrap(), 2);

        let next = Tensor::randn(0.0f32, 1.0f32, (1, 3, 2, 4), &device).unwrap();
        append_to_pages(4, &mut pages, &next).unwrap();
        assert_eq!(pages.len(), 4);
        assert_eq!(pages[2].dim(1).unwrap(), 4);
        assert_eq!(pages[3].dim(1).unwrap(), 1);
    }

    #[test]
    fn test_paged_decode_matches_dense_single_token() {
        let device = Device::Cpu;
        let bsz = 2usize;
        let num_heads = 4usize;
        let head_dim = 8usize;
        let total_len = 11usize;

        let q = Tensor::randn(0.0f32, 1.0f32, (bsz, 1, num_heads, head_dim), &device).unwrap();
        let k_full = Tensor::randn(
            0.0f32,
            1.0f32,
            (bsz, total_len, num_heads, head_dim),
            &device,
        )
        .unwrap();
        let v_full = Tensor::randn(
            0.0f32,
            1.0f32,
            (bsz, total_len, num_heads, head_dim),
            &device,
        )
        .unwrap();

        let mut k_pages = Vec::new();
        let mut v_pages = Vec::new();
        append_to_pages(3, &mut k_pages, &k_full).unwrap();
        append_to_pages(3, &mut v_pages, &v_full).unwrap();

        let paged = paged_decode_attention(&q, &k_pages, &v_pages, num_heads, head_dim).unwrap();

        // Dense reference implementation.
        let q_ref = q
            .transpose(1, 2)
            .unwrap()
            .reshape((bsz * num_heads, 1, head_dim))
            .unwrap();
        let k_ref = k_full
            .transpose(1, 2)
            .unwrap()
            .reshape((bsz * num_heads, total_len, head_dim))
            .unwrap();
        let v_ref = v_full
            .transpose(1, 2)
            .unwrap()
            .reshape((bsz * num_heads, total_len, head_dim))
            .unwrap();
        let scale = (head_dim as f64).sqrt();
        let mut scores = q_ref.matmul(&k_ref.transpose(1, 2).unwrap()).unwrap();
        let scale_t = Tensor::from_vec(vec![scale as f32], (1,), &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        scores = scores.broadcast_div(&scale_t).unwrap();
        let weights = ops::softmax(&scores, D::Minus1).unwrap();
        let dense = weights
            .matmul(&v_ref)
            .unwrap()
            .reshape((bsz, num_heads, 1, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        let paged_vals = paged
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let dense_vals = dense
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(paged_vals.len(), dense_vals.len());

        let mut max_abs_diff = 0.0f32;
        for (a, b) in paged_vals.iter().zip(dense_vals.iter()) {
            max_abs_diff = max_abs_diff.max((a - b).abs());
        }
        assert!(max_abs_diff < 1e-4, "max abs diff was {}", max_abs_diff);
    }
}
