use std::f32::consts::PI;

use candle_core::{DType, Tensor};
use candle_nn::{
    ops, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder,
};
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

use crate::error::{Error, Result};

use super::config::{KokoroConfig, KokoroIstftNetConfig};
use super::prosody::{
    load_plain_conv1d, load_weight_norm_conv1d, load_weight_norm_conv_transpose1d, AdaIN1d,
    AdainResBlk1d,
};

#[derive(Debug)]
pub struct KokoroDecoder {
    f0_conv: Conv1d,
    n_conv: Conv1d,
    asr_res: Conv1d,
    encode: AdainResBlk1d,
    decode: Vec<AdainResBlk1d>,
    generator: KokoroIstftGenerator,
}

impl KokoroDecoder {
    pub fn load(cfg: &KokoroConfig, vb: VarBuilder) -> Result<Self> {
        let root = vb.pp("module");
        let style_dim = cfg.style_dim;
        let encode = AdainResBlk1d::load(cfg.hidden_dim + 2, 1024, style_dim, false, root.pp("encode"))?;

        let mut decode = Vec::with_capacity(4);
        decode.push(AdainResBlk1d::load(1024 + 2 + 64, 1024, style_dim, false, root.pp("decode.0"))?);
        decode.push(AdainResBlk1d::load(1024 + 2 + 64, 1024, style_dim, false, root.pp("decode.1"))?);
        decode.push(AdainResBlk1d::load(1024 + 2 + 64, 1024, style_dim, false, root.pp("decode.2"))?);
        decode.push(AdainResBlk1d::load(1024 + 2 + 64, 512, style_dim, true, root.pp("decode.3"))?);

        let conv_stride2_cfg = Conv1dConfig {
            padding: 1,
            stride: 2,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let f0_conv = load_weight_norm_conv1d(root.pp("F0_conv"), conv_stride2_cfg)?;
        let n_conv = load_weight_norm_conv1d(root.pp("N_conv"), conv_stride2_cfg)?;
        let asr_res = load_weight_norm_conv1d(root.pp("asr_res.0"), Conv1dConfig::default())?;
        let generator = KokoroIstftGenerator::load(&cfg.istftnet, style_dim, root.pp("generator"))?;

        Ok(Self {
            f0_conv,
            n_conv,
            asr_res,
            encode,
            decode,
            generator,
        })
    }

    pub fn forward(
        &self,
        asr: &Tensor,      // [B, 512, T]
        f0_curve: &Tensor, // [B, 2T]
        n_curve: &Tensor,  // [B, 2T]
        style: &Tensor,    // [B, 128]
    ) -> Result<Vec<f32>> {
        let f0 = self
            .f0_conv
            .forward(&f0_curve.unsqueeze(1).map_err(Error::from)?)
            .map_err(Error::from)?;
        let n = self
            .n_conv
            .forward(&n_curve.unsqueeze(1).map_err(Error::from)?)
            .map_err(Error::from)?;

        let x = Tensor::cat(&[asr.clone(), f0.clone(), n.clone()], 1).map_err(Error::from)?;
        let mut x = self.encode.forward(&x, style)?;
        let asr_res = self.asr_res.forward(asr).map_err(Error::from)?;

        let mut still_concat_res = true;
        for block in &self.decode {
            if still_concat_res {
                x = Tensor::cat(&[x, asr_res.clone(), f0.clone(), n.clone()], 1).map_err(Error::from)?;
            }
            x = block.forward(&x, style)?;
            let (_b, _c, t) = x.dims3().map_err(Error::from)?;
            let (_, _, asr_t) = asr_res.dims3().map_err(Error::from)?;
            if t > asr_t {
                still_concat_res = false;
            }
        }

        self.generator.forward(&x, style, f0_curve)
    }
}

#[derive(Debug)]
struct KokoroIstftGenerator {
    cfg: KokoroIstftNetConfig,
    num_kernels: usize,
    num_upsamples: usize,
    total_scale: usize,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<AdaInResBlock1>,
    noise_convs: Vec<Conv1d>,
    noise_res: Vec<AdaInResBlock1>,
    conv_post: Conv1d,
    source_linear_w: [f32; 9],
    source_linear_b: f32,
}

impl KokoroIstftGenerator {
    fn load(cfg: &KokoroIstftNetConfig, style_dim: usize, vb: VarBuilder) -> Result<Self> {
        if cfg.resblock_kernel_sizes.is_empty() || cfg.upsample_rates.is_empty() {
            return Err(Error::ModelLoadError(
                "Kokoro ISTFTNet config missing kernels/upsample rates".to_string(),
            ));
        }
        let num_kernels = cfg.resblock_kernel_sizes.len();
        let num_upsamples = cfg.upsample_rates.len();
        let total_scale = cfg
            .upsample_rates
            .iter()
            .copied()
            .product::<usize>()
            .saturating_mul(cfg.gen_istft_hop_size);

        let mut ups = Vec::with_capacity(num_upsamples);
        for (i, (&u, &k)) in cfg
            .upsample_rates
            .iter()
            .zip(cfg.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let padding = (k.saturating_sub(u)) / 2;
            let ct = load_weight_norm_conv_transpose1d(
                vb.pp(format!("ups.{i}")),
                ConvTranspose1dConfig {
                    padding,
                    output_padding: 0,
                    stride: u,
                    dilation: 1,
                    groups: 1,
                },
            )?;
            ups.push(ct);
        }

        let mut resblocks = Vec::with_capacity(num_upsamples * num_kernels);
        for i in 0..num_upsamples {
            let ch = cfg.upsample_initial_channel / (1usize << (i + 1));
            for (j, (&kernel, dils)) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(cfg.resblock_dilation_sizes.iter())
                .enumerate()
            {
                resblocks.push(AdaInResBlock1::load(
                    ch,
                    kernel,
                    dils,
                    style_dim,
                    vb.pp(format!("resblocks.{}", i * num_kernels + j)),
                )?);
            }
        }

        let mut noise_convs = Vec::with_capacity(num_upsamples);
        let mut noise_res = Vec::with_capacity(num_upsamples);
        for i in 0..num_upsamples {
            let c_cur = cfg.upsample_initial_channel / (1usize << (i + 1));
            if i + 1 < num_upsamples {
                let stride_f0 = cfg.upsample_rates[i + 1..].iter().copied().product::<usize>();
                let padding = (stride_f0 + 1) / 2;
                noise_convs.push(load_plain_conv1d(
                    vb.pp(format!("noise_convs.{i}")),
                    Conv1dConfig {
                        padding,
                        stride: stride_f0,
                        dilation: 1,
                        groups: 1,
                        cudnn_fwd_algo: None,
                    },
                )?);
                noise_res.push(AdaInResBlock1::load(
                    c_cur,
                    7,
                    &[1, 3, 5],
                    style_dim,
                    vb.pp(format!("noise_res.{i}")),
                )?);
            } else {
                noise_convs.push(load_plain_conv1d(
                    vb.pp(format!("noise_convs.{i}")),
                    Conv1dConfig::default(),
                )?);
                noise_res.push(AdaInResBlock1::load(
                    c_cur,
                    11,
                    &[1, 3, 5],
                    style_dim,
                    vb.pp(format!("noise_res.{i}")),
                )?);
            }
        }

        let conv_post = load_weight_norm_conv1d(
            vb.pp("conv_post"),
            Conv1dConfig {
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        )?;

        let source_linear_w_t = vb
            .pp("m_source.l_linear")
            .get_unchecked_dtype("weight", DType::F32)
            .map_err(Error::from)?;
        let source_linear_b_t = vb
            .pp("m_source.l_linear")
            .get_unchecked_dtype("bias", DType::F32)
            .map_err(Error::from)?;
        let source_linear_w_v = source_linear_w_t.to_vec2::<f32>().map_err(Error::from)?;
        let source_linear_b_v = source_linear_b_t.to_vec1::<f32>().map_err(Error::from)?;
        if source_linear_w_v.len() != 1 || source_linear_w_v[0].len() != 9 || source_linear_b_v.len() != 1 {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Kokoro source linear shapes: weight={:?}, bias={:?}",
                source_linear_w_t.shape().dims(),
                source_linear_b_t.shape().dims(),
            )));
        }
        let mut source_linear_w = [0.0f32; 9];
        source_linear_w.copy_from_slice(&source_linear_w_v[0]);
        let source_linear_b = source_linear_b_v[0];

        Ok(Self {
            cfg: cfg.clone(),
            num_kernels,
            num_upsamples,
            total_scale,
            ups,
            resblocks,
            noise_convs,
            noise_res,
            conv_post,
            source_linear_w,
            source_linear_b,
        })
    }

    fn forward(&self, x: &Tensor, style: &Tensor, f0_curve: &Tensor) -> Result<Vec<f32>> {
        let har = self.harmonic_features(f0_curve, x.device())?; // [B, n_fft+2, T_har]

        let mut x = x.clone();
        for i in 0..self.num_upsamples {
            x = ops::leaky_relu(&x, 0.1).map_err(Error::from)?;
            let mut x_source = self.noise_convs[i].forward(&har).map_err(Error::from)?;
            x_source = self.noise_res[i].forward(&x_source, style)?;
            x = self.ups[i].forward(&x).map_err(Error::from)?;
            if i + 1 == self.num_upsamples {
                x = reflection_pad_left1(&x)?;
            }
            x = match_time_add(&x, &x_source)?;

            let base = i * self.num_kernels;
            let mut xs = self.resblocks[base].forward(&x, style)?;
            for j in 1..self.num_kernels {
                let y = self.resblocks[base + j].forward(&x, style)?;
                xs = (xs + y).map_err(Error::from)?;
            }
            x = (xs / self.num_kernels as f64).map_err(Error::from)?;
        }

        x = ops::leaky_relu(&x, 0.01).map_err(Error::from)?;
        x = self.conv_post.forward(&x).map_err(Error::from)?;
        let (_b, c, _t) = x.dims3().map_err(Error::from)?;
        let n_bins = self.cfg.gen_istft_n_fft / 2 + 1;
        if c < n_bins * 2 {
            return Err(Error::InferenceError(format!(
                "Kokoro generator conv_post output channels {} < required {}",
                c,
                n_bins * 2
            )));
        }
        let spec = x.narrow(1, 0, n_bins).map_err(Error::from)?.exp().map_err(Error::from)?;
        let phase = x
            .narrow(1, n_bins, n_bins)
            .map_err(Error::from)?
            .sin()
            .map_err(Error::from)?;
        let stft = KokoroStft::new(self.cfg.gen_istft_n_fft, self.cfg.gen_istft_hop_size);
        stft.inverse(&spec, &phase)
    }

    fn harmonic_features(&self, f0_curve: &Tensor, device: &candle_core::Device) -> Result<Tensor> {
        let f0_rows = f0_curve.to_vec2::<f32>().map_err(Error::from)?;
        if f0_rows.len() != 1 {
            return Err(Error::InferenceError(
                "Kokoro generator currently supports batch size 1 for harmonic source".to_string(),
            ));
        }
        let f0 = &f0_rows[0];
        if f0.is_empty() {
            return Err(Error::InferenceError(
                "Kokoro generator received empty F0 curve".to_string(),
            ));
        }
        let upsampled_f0 = repeat_nearest(f0, self.total_scale);
        let har_source = synth_harmonic_source(
            &upsampled_f0,
            &self.source_linear_w,
            self.source_linear_b,
            KokoroConfig::TARGET_SAMPLE_RATE as f32,
        );
        let stft = KokoroStft::new(self.cfg.gen_istft_n_fft, self.cfg.gen_istft_hop_size);
        let (mag, phase) = stft.transform(&har_source)?;
        let n_bins = self.cfg.gen_istft_n_fft / 2 + 1;
        let frames = if n_bins == 0 { 0 } else { mag.len() / n_bins };
        let mut har = vec![0.0f32; n_bins * 2 * frames];
        for k in 0..n_bins {
            for t in 0..frames {
                har[k * frames + t] = mag[t * n_bins + k];
                har[(n_bins + k) * frames + t] = phase[t * n_bins + k];
            }
        }
        Tensor::from_vec(har, (1, n_bins * 2, frames), device).map_err(Error::from)
    }
}

#[derive(Debug)]
struct AdaInResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    adain1: Vec<AdaIN1d>,
    adain2: Vec<AdaIN1d>,
    alpha1: Vec<Tensor>,
    alpha2: Vec<Tensor>,
}

impl AdaInResBlock1 {
    fn load(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        style_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        if dilations.len() != 3 {
            return Err(Error::ModelLoadError(format!(
                "Kokoro AdaInResBlock1 expects 3 dilations, got {}",
                dilations.len()
            )));
        }
        let mut convs1 = Vec::with_capacity(3);
        let mut convs2 = Vec::with_capacity(3);
        let mut adain1 = Vec::with_capacity(3);
        let mut adain2 = Vec::with_capacity(3);
        let mut alpha1 = Vec::with_capacity(3);
        let mut alpha2 = Vec::with_capacity(3);
        for j in 0..3 {
            let d1 = dilations[j];
            convs1.push(load_weight_norm_conv1d(
                vb.pp(format!("convs1.{j}")),
                Conv1dConfig {
                    padding: get_padding(kernel_size, d1),
                    stride: 1,
                    dilation: d1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            )?);
            convs2.push(load_weight_norm_conv1d(
                vb.pp(format!("convs2.{j}")),
                Conv1dConfig {
                    padding: get_padding(kernel_size, 1),
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            )?);
            adain1.push(AdaIN1d::load(style_dim, channels, vb.pp(format!("adain1.{j}")))?);
            adain2.push(AdaIN1d::load(style_dim, channels, vb.pp(format!("adain2.{j}")))?);
            alpha1.push(
                vb.get_unchecked_dtype(&format!("alpha1.{j}"), DType::F32)
                    .map_err(Error::from)?,
            );
            alpha2.push(
                vb.get_unchecked_dtype(&format!("alpha2.{j}"), DType::F32)
                    .map_err(Error::from)?,
            );
        }
        Ok(Self {
            convs1,
            convs2,
            adain1,
            adain2,
            alpha1,
            alpha2,
        })
    }

    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for j in 0..3 {
            let mut xt = self.adain1[j].forward(&x, style)?;
            xt = snake1d(&xt, &self.alpha1[j])?;
            xt = self.convs1[j].forward(&xt).map_err(Error::from)?;
            xt = self.adain2[j].forward(&xt, style)?;
            xt = snake1d(&xt, &self.alpha2[j])?;
            xt = self.convs2[j].forward(&xt).map_err(Error::from)?;
            x = (xt + x).map_err(Error::from)?;
        }
        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct KokoroStft {
    n_fft: usize,
    hop: usize,
    window: Vec<f32>,
}

impl KokoroStft {
    fn new(n_fft: usize, hop: usize) -> Self {
        Self {
            n_fft,
            hop,
            window: hann_window_periodic(n_fft),
        }
    }

    fn transform(&self, input: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if input.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        if self.n_fft == 0 || self.hop == 0 {
            return Err(Error::InferenceError(
                "Invalid Kokoro STFT n_fft/hop".to_string(),
            ));
        }
        let pad = self.n_fft / 2;
        let padded = reflect_pad_1d_center(input, pad)?;
        if padded.len() < self.n_fft {
            return Ok((Vec::new(), Vec::new()));
        }
        let n_bins = self.n_fft / 2 + 1;
        let frames = (padded.len() - self.n_fft) / self.hop + 1;
        let mut mag = vec![0.0f32; frames * n_bins];
        let mut phase = vec![0.0f32; frames * n_bins];
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.n_fft);
        let mut buf = vec![Complex32::new(0.0, 0.0); self.n_fft];

        for frame_idx in 0..frames {
            let start = frame_idx * self.hop;
            for i in 0..self.n_fft {
                buf[i] = Complex32::new(padded[start + i] * self.window[i], 0.0);
            }
            fft.process(&mut buf);
            for k in 0..n_bins {
                let c = buf[k];
                mag[frame_idx * n_bins + k] = c.norm();
                phase[frame_idx * n_bins + k] = c.arg();
            }
        }
        Ok((mag, phase))
    }

    fn inverse(&self, magnitude: &Tensor, phase: &Tensor) -> Result<Vec<f32>> {
        let (b, n_bins, frames) = magnitude.dims3().map_err(Error::from)?;
        let (b2, n_bins2, frames2) = phase.dims3().map_err(Error::from)?;
        if b != 1 || b2 != 1 || n_bins != n_bins2 || frames != frames2 {
            return Err(Error::InferenceError(format!(
                "Kokoro iSTFT expects matching [1,n_bins,frames] tensors, got mag={:?}, phase={:?}",
                magnitude.shape().dims(),
                phase.shape().dims(),
            )));
        }
        let mag_v = magnitude.to_vec3::<f32>().map_err(Error::from)?;
        let phase_v = phase.to_vec3::<f32>().map_err(Error::from)?;
        let mag = &mag_v[0];
        let ph = &phase_v[0];

        if frames == 0 {
            return Ok(Vec::new());
        }
        let output_len = (frames - 1) * self.hop + self.n_fft;
        let mut output = vec![0.0f32; output_len];
        let mut envelope = vec![0.0f32; output_len];
        let mut planner = FftPlanner::<f32>::new();
        let ifft = planner.plan_fft_inverse(self.n_fft);
        let mut spectrum = vec![Complex32::new(0.0, 0.0); self.n_fft];

        for frame_idx in 0..frames {
            spectrum.fill(Complex32::new(0.0, 0.0));
            for k in 0..n_bins {
                let m = mag[k][frame_idx].max(0.0);
                let p = ph[k][frame_idx];
                spectrum[k] = Complex32::from_polar(m, p);
            }
            for k in 1..(n_bins.saturating_sub(1)) {
                spectrum[self.n_fft - k] = spectrum[k].conj();
            }
            ifft.process(&mut spectrum);
            let start = frame_idx * self.hop;
            for n in 0..self.n_fft {
                let sample = (spectrum[n].re / self.n_fft as f32) * self.window[n];
                let idx = start + n;
                output[idx] += sample;
                envelope[idx] += self.window[n] * self.window[n];
            }
        }

        for (y, env) in output.iter_mut().zip(envelope.iter()) {
            if *env > 1e-8 {
                *y /= *env;
            }
            if !y.is_finite() {
                *y = 0.0;
            }
        }

        let pad = self.n_fft / 2;
        let mut trimmed = if output.len() > pad * 2 {
            output[pad..output.len() - pad].to_vec()
        } else {
            output
        };
        for s in &mut trimmed {
            *s = s.clamp(-1.0, 1.0);
        }
        Ok(trimmed)
    }
}

fn get_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size.saturating_mul(dilation).saturating_sub(dilation)) / 2
}

fn snake1d(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let ax = x.broadcast_mul(alpha).map_err(Error::from)?;
    let sin_sq = ax.sin().map_err(Error::from)?.sqr().map_err(Error::from)?;
    let scaled = sin_sq.broadcast_div(alpha).map_err(Error::from)?;
    x.broadcast_add(&scaled).map_err(Error::from)
}

fn upsample_nearest_1d(x: &Tensor, factor: usize) -> Result<Tensor> {
    if factor <= 1 {
        return Ok(x.clone());
    }
    let (b, c, t) = x.dims3().map_err(Error::from)?;
    x.unsqueeze(3)
        .map_err(Error::from)?
        .broadcast_as((b, c, t, factor))
        .map_err(Error::from)?
        .reshape((b, c, t * factor))
        .map_err(Error::from)
}

fn reflection_pad_left1(x: &Tensor) -> Result<Tensor> {
    let (_b, _c, t) = x.dims3().map_err(Error::from)?;
    if t < 2 {
        return Err(Error::InferenceError(
            "Kokoro reflection pad requires time length >= 2".to_string(),
        ));
    }
    let left = x.narrow(2, 1, 1).map_err(Error::from)?;
    Tensor::cat(&[left, x.clone()], 2).map_err(Error::from)
}

fn match_time_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (ba, ca, ta) = a.dims3().map_err(Error::from)?;
    let (bb, cb, tb) = b.dims3().map_err(Error::from)?;
    if ba != bb || ca != cb {
        return Err(Error::InferenceError(format!(
            "Kokoro generator add shape mismatch {:?} vs {:?}",
            a.shape().dims(),
            b.shape().dims()
        )));
    }
    if ta == tb {
        return (a + b).map_err(Error::from);
    }
    let t = ta.min(tb);
    let a2 = a.narrow(2, 0, t).map_err(Error::from)?;
    let b2 = b.narrow(2, 0, t).map_err(Error::from)?;
    (a2 + b2).map_err(Error::from)
}

fn repeat_nearest(input: &[f32], factor: usize) -> Vec<f32> {
    if factor <= 1 {
        return input.to_vec();
    }
    let mut out = Vec::with_capacity(input.len().saturating_mul(factor));
    for &v in input {
        for _ in 0..factor {
            out.push(v);
        }
    }
    out
}

fn synth_harmonic_source(
    upsampled_f0: &[f32],
    linear_w: &[f32; 9],
    linear_b: f32,
    sample_rate: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; upsampled_f0.len()];
    let mut base_phase = 0.0f32;
    for (i, &f0) in upsampled_f0.iter().enumerate() {
        if !f0.is_finite() || f0 <= 10.0 {
            out[i] = 0.0;
            continue;
        }
        let delta = 2.0 * PI * (f0 / sample_rate);
        base_phase += delta;
        if base_phase > 2.0 * PI {
            base_phase %= 2.0 * PI;
        }
        let mut acc = linear_b;
        for h in 0..9 {
            let phase = base_phase * (h as f32 + 1.0);
            let s = phase.sin() * 0.1;
            acc += linear_w[h] * s;
        }
        out[i] = acc.tanh();
    }
    out
}

fn reflect_pad_1d_center(input: &[f32], pad: usize) -> Result<Vec<f32>> {
    if pad == 0 {
        return Ok(input.to_vec());
    }
    if input.len() <= 1 || pad >= input.len() {
        return Err(Error::InferenceError(format!(
            "Kokoro STFT reflect pad invalid for len={} pad={}",
            input.len(),
            pad
        )));
    }
    let mut out = Vec::with_capacity(input.len() + pad * 2);
    for i in 0..pad {
        out.push(input[pad - i]);
    }
    out.extend_from_slice(input);
    for i in 0..pad {
        out.push(input[input.len() - 2 - i]);
    }
    Ok(out)
}

fn hann_window_periodic(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kokoro_hann_window_periodic_smoke() {
        let w = hann_window_periodic(20);
        assert_eq!(w.len(), 20);
        assert!(w.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn kokoro_repeat_nearest_scales_len() {
        let x = vec![1.0f32, 2.0, 3.0];
        let y = repeat_nearest(&x, 4);
        assert_eq!(y.len(), 12);
        assert_eq!(y[0], 1.0);
        assert_eq!(y[4], 2.0);
        assert_eq!(y[8], 3.0);
    }

    #[test]
    fn kokoro_stft_roundtrip_smoke() {
        let stft = KokoroStft::new(20, 5);
        let mut x = Vec::with_capacity(600);
        for i in 0..600 {
            x.push((i as f32 * 0.01).sin() * 0.2);
        }
        let (mag, phase) = stft.transform(&x).expect("stft transform");
        let n_bins = 20 / 2 + 1;
        let frames = mag.len() / n_bins;
        let device = candle_core::Device::Cpu;
        let mut mag_ch = vec![0.0f32; n_bins * frames];
        let mut phase_ch = vec![0.0f32; n_bins * frames];
        for t in 0..frames {
            for k in 0..n_bins {
                mag_ch[k * frames + t] = mag[t * n_bins + k];
                phase_ch[k * frames + t] = phase[t * n_bins + k];
            }
        }
        let mag_t = Tensor::from_vec(mag_ch, (1, n_bins, frames), &device).expect("mag tensor");
        let phase_t = Tensor::from_vec(phase_ch, (1, n_bins, frames), &device).expect("phase tensor");
        let y = stft.inverse(&mag_t, &phase_t).expect("istft inverse");
        assert!(!y.is_empty());
        assert!(y.iter().all(|v| v.is_finite()));
    }
}
