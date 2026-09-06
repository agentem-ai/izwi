# LFM CUDA implementation and qualification

LFM uses Candle 0.11.0. The text backbone and audio depthformer retain quantized
projections; audio encoder/detokenizer components are dense. Current backbone
activations, recurrent rings and physical KV state are F32. A generic configured
BF16 preference does not convert this model. CUDA support remains source-reviewed
and unverified until device evidence is collected.

The implementation fixes the text executor/sequence-adapter mismatch, uses
Candle's native noninterleaved RoPE (including different positions per decode
row), softmax, SiLU and tanh GELU, and packs greedy token IDs before host readback.
Retained singleton audio prefill uses the existing full-span path. ASR declares
its existing full-prompt prefill workspace estimate as well as decode workspace,
including when the execution adapter is rebound. ShortConv only
constructs ring updates for the suffix that survives. Nonstreaming audio-chat
decodes the final waveform without repeatedly decoding intermediate prefixes.
Relative-position Conformer attention and GLU semantics are preserved.

## Optional depthwise convolution

`IZWI_LFM_CUDA_DEPTHWISE_CONV=1` enables an experimental Candle
unfold/multiply/reduce implementation for the encoder's F32 kernel-9, padding-4,
stride-1 depthwise Conv1d. It reuses folded affine weights and limits each expanded
product to 1 MiB. Padded input, reduced chunks, output concatenation and allocator
retention consume additional memory; the limit is not a total workspace bound.
Unsupported geometry uses the existing convolution. Unset the variable or set it
to `0` to retain the default path. Two-dimensional subsampling is unchanged.

Keep this option off until end-to-end quality, peak memory and latency improve
on the deployment GPU. No additional custom CUDA kernel was introduced.

## Validation on NVIDIA hardware

Build/link checks, device numerical checks and model performance evidence are
separate requirements. The portable tests alone satisfy neither CUDA compilation
nor GPU qualification. With the CUDA toolkit and an NVIDIA device, run the
explicit kernel tests (these fail if the device is unavailable):

```bash
cargo test --locked -p izwi-core --lib --features cuda \
  cuda_lfm2_batched_rotary_matches_scalar_at_ragged_positions -- --ignored
cargo test --locked -p izwi-core --lib --features cuda \
  cuda_candle_conformer_operations_and_depthwise_match_reference -- --ignored
cargo test --locked -p izwi-core --lib --features cuda \
  lfm25_cuda_packed_frame_tokens_preserve_greedy_and_mixed_rows -- --ignored
```

Run the normal LFM regressions with the deployment features, including
`flash-attn` and `cudnn` if deployed. Do not infer a provider from enabled features;
retain the provider actually observed by runtime telemetry.

Against a server built from the same clean commit, with all models provisioned:

```bash
scripts/bench/run-model-evidence.sh --backend cuda \
  --manifest benchmarks/manifests/lfm-cuda.toml \
  --server http://127.0.0.1:8080 --output target/lfm-cuda-evidence
```

The manifest covers both text variants and audio ASR/TTS at concurrency 1/2/4/8,
with streaming and nonstreaming requests. Compare the baseline and changed builds
on the same GPU, weights, inputs and limits. Retain TTFT, prefill/decode latency,
tokens or audio seconds per second, peak VRAM, workspace, kernel launches and
host synchronization counts. Check transcript quality, audio quality, finite
outputs, cancellation, EOS and rollback. Include long prompts and unequal prompt
lengths in a separate stress run. Audio-chat has no benchmark command: explicitly
exercise its streaming and nonstreaming API, checking final transcript/audio and
stream output boundaries. The manifest does not cover that route.

## Remaining evidence-dependent work

- Automatic retained audio context remains capped at 4,096; explicit context can
  exceed this. Request generation is already clamped to effective context minus
  prompt length. Raising automatic capacity requires fitting every physical state
  domain and request workspace, not merely changing the output-token policy.
- Measure multirow token-wave prefill before adding packed/span batching. Preserve
  ragged positions, recurrent state, transaction rollback and cancellation.
- Qualify BF16 attention/KV separately from F16. Update physical contracts,
  allocation byte counts, queries and outputs together; keep recurrence F32.
  Test raw KV finiteness and long-generation quality. F16's narrower exponent
  range needs separate evidence. Dense audio precision is another experiment.
- Batch audio heads only across requests at the same dependent codebook step.
  Preserve stochastic sampling and first-codebook EOS. Chat generation currently
  remains greedy; sampling-policy changes need their own API correctness work.
- Share ASR/response encoder preparation only when preprocessing and long-form
  chunking agree. Establish detokenizer dependencies before adding incremental
  streaming state. Profile existing graph/provider paths before new kernels.
