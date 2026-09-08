# Fish S2 Pro validation

The native Candle implementation targets `fishaudio/s2-pro` revision
`1de9996b6be38b745688de084d87a5633f714e4e`, compared with Fish Speech revision
`befe4001745417f8c42131739d862b8a6fdbd15a`. New downloads use the pinned revision.
Existing local bundles remain usable; compare their configuration and tensor
inventory before treating them as equivalent to the reviewed checkpoint.

| Backend | Default transformer dtype | Codec dtype |
| --- | --- | --- |
| CPU | F32 | F32 |
| Metal | F16 | F32 |
| CUDA with observed SM 8.0+ BF16 support | checkpoint BF16 | F32 |
| CUDA without BF16 support | F32 | F32 |

`IZWI_FISH_S2_DTYPE` accepts `f32`, `f16`, or `bf16` only when the backend
supports that representation. CPU half and Metal BF16 are rejected before
neural weights are allocated. An explicit supported CUDA F16 override trades
BF16's exponent range for a different representation; compare audio before
using it in deployment.

## Portable regression checks

```sh
cargo test --locked -p izwi-core --features accelerate --lib fish_s2
cargo test --locked -p izwi-cli -p izwi-server --features izwi-core/accelerate fish_s2
```

Use the platform's supported BLAS feature where Accelerate is unavailable.
These checks cover small synthetic tensors, reference mathematical fixtures,
resource arithmetic and serving contracts. They do not measure speech quality
or certify whole-model accelerator performance.

The metadata-only gate can check an installed bundle without allocating its neural
weights:

```sh
IZWI_FISH_S2_MODEL_DIR='/path/to/models/FishAudio-S2-Pro' \
cargo test --locked -p izwi-core --features accelerate --lib \
  fish_s2_real_artifact_metadata_contracts -- --ignored --nocapture
```

Explicit small device probes are also available under the `metal`/`cuda`
features: `metal_rotation_matches_frozen_upstream_oracles`,
`metal_compact_head_matches_full_projection`,
`metal_sampling_matches_cpu_policy_and_rejects_raw_nan`,
`metal_codec_attention_matches_dense_oracle`, and their `cuda_` counterparts.
Run them with `--ignored --nocapture`; device creation must succeed. A missing
device is an error, never an implicit CPU pass.

## Real model through the runtime

The ignored smoke test uses `RuntimeService`, including reference preparation,
retained slow/fast state, scheduled generation and waveform finalization. It
requires enough memory for weights, caches, workspace and the operating system.
Run one loaded model at a time on memory-constrained hosts.

```sh
IZWI_FISH_S2_MODEL_DIR='/path/to/models/FishAudio-S2-Pro' \
IZWI_FISH_S2_BACKEND=cuda \
IZWI_FISH_S2_SMOKE_OUTPUT_WAV=/tmp/fish-s2-cuda.wav \
cargo test --locked -p izwi-core --features cuda,cudnn,flash-attn \
  --lib fish_s2_real_model_smoke_generates_finite_audio -- --ignored --nocapture
```

For Metal, select `metal` for both the backend and build feature; for CPU select
`cpu` and the appropriate BLAS build feature. CUDA product builds using the CLI
or server `cuda` feature already include Candle FlashAttention. A core-only
`cuda` build needs the explicit `flash-attn` feature to include that provider.

The default conditioning pair is `data/fox.wav` and its exact transcript in
`data/fox.md`; the initial target text is also `data/fox.md`. Then repeat with
`IZWI_FISH_S2_TARGET_TEXT` set to a held-out sentence. Custom reference audio
requires `IZWI_FISH_S2_REFERENCE_WAV` and its matching
`IZWI_FISH_S2_REFERENCE_TEXT`. The test defaults to 512 output frames and an
explicit 4096-token context; override `IZWI_FISH_S2_SMOKE_MAX_FRAMES` and
`IZWI_FISH_S2_SMOKE_CONTEXT` for longer probes. These test bounds do not change
production's native-context policy.

Retain the output WAV, exact source commit, model revision, build features,
dtype, device/driver/CUDA/cuDNN versions and logged timings. Inspect intelligibility,
reference voice similarity, duration, clipping, repeated speech, EOS behavior
and non-finite values. Use listening plus transcription against the target;
finite and non-silent output alone is insufficient.

## Backend performance evidence

Use `benchmarks/manifests/cuda-family-api.toml` for family coverage and the
`{cpu,metal,cuda}-audio-concurrency.toml` manifests for concurrency 1, 2, 4 and 8.
Every Fish case provides the same matching reference voice/transcript. Fish now decodes committed codebook frames incrementally through a resumable
codec stage. Native tensor batching and AR/codec overlap remain separate,
unimplemented optimizations.

Measure warmed and cold preparation, prefill, semantic/codebook decode and codec
finalization separately. Record time to first audio, real-time factor, peak
host/device memory and cancellation latency for short and long reference/target
pairs, including unspaced multilingual text. Verify the actual selected Candle
attention provider on the deployed GPU. Compare numerics and audio before
accepting a faster dtype, convolution algorithm or attention provider.

The codec currently keeps F32 compute on all backends. CUDA convolution uses
Candle's cuDNN `ImplicitGemm` algorithm to keep scratch bounded; faster algorithm
selection requires measured workspace limits and an audio-parity comparison on
the target GPU. This is a memory-safe baseline, not a claim of the fastest
cuDNN algorithm. The slow decoder retains the existing Candle paged-attention
routing, including FlashAttention when the build/device/dtype are supported.
Reference input rates are bounded at 384 kHz and prepared audio at 4096 codec
frames; out-of-contract inputs fail before neural codec allocation.

## Verification recorded on 2026-09-05

On the local macOS arm64 host:

- CPU/Accelerate Fish suite: 84 passed; three real-artifact tests excluded from
  the ordinary run. The metadata-only artifact test was also run and passed.
- Full core suite: 2,404 passed, eight ignored, three pre-existing failures:
  `global_fused_kernel_availability_tracks_known_backends`,
  `predictor_batch_matches_scalar_codebooks_and_isolates_receipts`, and
  `repeated_mrope_positions_match_standard_rope_when_axes_equal`. These match
  the baseline audit; they are outside the Fish changes.
- CLI: all 63 tests passed. Server: all 390 tests passed.
- CPU and Metal feature builds compiled. Actual Metal device probes passed for
  rotary fixtures, compact-vs-full projection, sampling with dense and sparse
  masks, raw-NaN rejection, and blocked codec attention against the dense oracle.
  GPU probes need device access; the restricted sandbox hides Metal devices.

No whole-model speech generation, CUDA build, CUDA device execution or end-to-end
CUDA performance measurement was performed. This 16 GiB host cannot comfortably
hold the F32 model plus its inference state. Use the protocol above on an
appropriately sized device before certifying audible quality, peak memory or
throughput. No custom kernel or dependency upgrade was introduced.

## Preparation reservation contract

Fish preparation now reserves host scratch and the maximum retained prompt
before codec/tokenizer work. The physical batch runner exclusively owns codec
tensor workspace; the request does not reserve a second copy of that workspace.
CUDA keeps these host and device budgets separate. CPU and Metal map the same
ownership to host and unified memory respectively.

The dense conditioning prompt is bounded by the smaller of loaded-model and
fitted context, leaving a generation position. Text is tokenized once, and the
complete prompt length is checked before allocating codebook rows and the mask.
The retained observation counts actual buffer capacities, the outer row vector
and the shared artifact allocation once. Execution admission carries that exact
host charge independently of generic request/output allowances.

Each scalar audio-decode quantum derives its workspace from its bounded frame
count and the sealed request budget at the exact bound stage. Terminal flush
reserves no full-waveform decode workspace. Preparation leaves
native tensor-stage costs empty: those costs require an already-bound batchable
stage and cannot represent Fish's scalar codec. The finalization regression
covers artifact installation before binding, request validation after binding,
and workspace admission for CPU, CUDA and Metal:

```sh
cargo test -p izwi-core --lib fish_s2_finalization_tests
```

Decoded inputs stay covered while row sealing and dispatch run, and failed
handoffs free owned buffers before releasing their reservation. Resource errors
retain their strict guard and report reservation ID/class, operation, authorized
and observed domains; preparation observations add bounded request correlation,
backend and commit stage without logging speech text or reference audio.

Host scratch includes decoded audio capacity, resampling/FFT buffers, VQ
readback and tokenization. Tokenizer scratch uses a conservative envelope of
256 bytes per rendered UTF-8 byte plus a 16 MiB small-buffer allowance; it is
not a measured allocator peak. Real CUDA peak memory and speech quality remain
part of device qualification below.

Portable regression entry points:

```sh
cargo test -p izwi-core --lib fish_s2_preparation_tests
cargo test -p izwi-core --lib models::architectures::fish_s2::
cargo test -p izwi-core --lib materialization_diagnostics
```

## Reservation incident API replay

`scripts/bench/run-fish-cuda-replay.py` exercises the saved-voice preparation
boundary implicated in the September 7 CUDA reservation incident. Run it only
against an authorized, reviewed deployment with the saved voice available. The
script creates persistent speech-history records and retains reference audio,
transcripts and generated audio in a new private evidence directory. It does not
deploy, delete records or certify CUDA merely from a successful HTTP response.

Prepare a JSON metadata file from **the deployed build and runtime logs**, with:

```json
{
  "deployed_sha": "FULL_40_CHARACTER_DEPLOYED_GIT_SHA",
  "gpu": "Actual GPU model and memory",
  "backend": "cuda",
  "dtype": "Actual runtime dtype(s), including codec",
  "effective_context": 4096,
  "service_concurrency": 2,
  "runtime_versions": "Driver, CUDA, cuDNN, build features and model revision"
}
```

Replace all examples with observed values. Metadata is explicitly labeled
operator-supplied; the replay does not infer the remote SHA from a local checkout.
From the repository root, run:

```sh
python3 scripts/bench/run-fish-cuda-replay.py \
  --server https://brizdigital--izwi-cuda-serve.modal.run \
  --metadata /tmp/fish-deployed-metadata.json \
  --output /tmp/fish-cuda-reservation-evidence \
  --sequential 3 --concurrency 2 --cancel
```

The default voice is `55fc6c39-ed9d-495e-93fb-87ea76d3a955` and the default target
is `Tell me something funny, I stay laughing`; override with `--saved-voice-id`
and `--text`. The runner fetches that voice's exact stored audio and transcript
for the direct-reference case, so the two paths use the same conditioning pair.
It warms the saved-voice path, repeats sequentially, runs the direct-reference
case and submits one bounded concurrent wave. Set `--concurrency` to the actual
configured service concurrency to qualify its limit. Each case retains the
created record, polls, final record, HTTP headers, errors and WAV. Successful
speech requires completed status, nonempty fully readable PCM WAV frames, a
finite positive recorded duration and agreement with the decoded duration.
This does not measure audible quality or establish simultaneous GPU execution.

`--cancel` also creates and immediately cancels a pending job. The API currently
represents cancellation as `failed` with `Cancelled by speech history request`;
the runner requires both the cancellation acknowledgement and that final reason.
A completion race fails the cancellation check; rerun on an appropriate workload
and retain both traces. This pending-job check does **not** prove cancellation
after conditioning has materialized. Separately exercise cancellation during
preparation and execution under logs, checking release after work unwinds.

If a metrics endpoint is exposed, add `--metrics-path /ACTUAL/METRICS/PATH` to
retain snapshots after warmup, after completion and after cancellation. The
runner does not assume metrics expose per-request ownership or that an immediate
snapshot proves quiescence. On an otherwise idle service, wait for cleanup and
compare coordinator request/artifact reservation counts, authorized vectors and
materialized vectors with the warmed baseline in logs/metrics. Attach correlation
IDs, operation/stage and host/device values for any failures. Do not confuse
persistent model allocations with per-request ownership; exclude unrelated
traffic. If these fields are not exposed, collect internal runtime logs. Missing
lease evidence remains an open qualification gate.

Exit zero means the automatic API/audio checks passed. `hardware_qualified`
remains false in `summary.json`: deployment identity, actual CUDA backend/device,
lease return to baseline (including in-flight cancellation), peak memory and
listening quality require reviewed hardware evidence. No live CUDA qualification
is implied by the offline fixtures:

```sh
python3 scripts/bench/test-fish-cuda-replay.py
```


## Incremental streaming implementation — 2026-09-08

The scheduler yields after the first four committed semantic frames, then at
most sixteen frames per codec quantum (2,048 mono samples per frame at 44.1 kHz).
Causal convolution, transposed convolution and bounded-attention histories are
retained between calls. The full decoder remains the numerical oracle. Flush
emits no extra samples: the oracle trims transposed-convolution tails.

Codec candidates and PCM are published only after the engine commits the
quantum. Aborts restore the previous history and sequence number. Visible audio
prevents sequence recomputation. Output ceilings still allow a pending codec
quantum and the final flush; codec steps do not consume semantic-token budget.
The 224-token prompt / 192-token first-prefill regression remains covered.

Retained codec history reserves both committed and candidate snapshots in the
backend's memory domain. Scalar transient workspace has one owner. Host output
allows retained PCM and pending/transport copies. Reference code reuse has a
separate 4 MiB model-owned host budget, with at most 32 entries, content-based
keys and coalesced misses. It caches codes, never target-specific prompts. Model
replacement invalidates the cache. Warm reference hits improve preparation and
request latency; they do not by themselves improve execution-only RTF.

The speech-history route streams PCM16 over SSE and spools those exact bytes to
a temporary WAV. Success repairs the header and stores the file; failures and
cancellation remove the spool. The spool limit follows the model/frame contract
and planned long-form constituents; `IZWI_TTS_STREAM_MAX_PCM_BYTES` can tighten
it. The current storage interface still requires a bounded final byte vector.
Browser playback belongs to the route, survives modal closure, unlocks on the
Generate gesture and applies backpressure at 30 seconds or 1,024 queued buffers.
Navigation and cancellation stop playback and the stream.

The OpenAI-compatible `/v1/audio/speech` streaming mode uses SSE
(`stream: true` or `stream_format: "sse"`). Its WAV format encodes each chunk as
a complete WAV; raw `pcm_i16`/`pcm_f32` chunks are separate options. This differs
from a single progressively written WAV response.

Terminal runtime metadata carries cumulative semantic-frame counts even when
its PCM is empty. Server SSE timing distinguishes first/last PCM, request wall
time and execution RTF. Model tracing reports reference encoding, prompt build,
prefill, slow/fast AR, sampling and codec **host wall intervals**. Sampling is a
subset of AR time; these are not additive GPU kernel timings. Use a CUDA profiler
for synchronization, launch gaps and provider/kernel attribution.

### Portable and device checks

```sh
cargo test --locked -p izwi-core --lib fish_s2
cargo test --locked -p izwi-core --lib fish_pending
cargo test --locked -p izwi-server --lib
python3 scripts/bench/test-fish-streaming-benchmark.py
```

Explicit synthetic device parity and rollback checks use:

```sh
cargo test --locked -p izwi-core --features metal --lib \
  metal_incremental_codec_matches_full_decode_and_rollback -- --ignored --nocapture
cargo test --locked -p izwi-core --features cuda,cudnn,flash-attn --lib \
  cuda_incremental_codec_matches_full_decode_and_rollback -- --ignored --nocapture
```

For the actual checkpoint, provide JSON containing all ten codebook rows, with
more than 128 real frames and at most 4,096 frames. The CUDA test compares full
decode with chunk sizes 1, 4, 16 and 32 (maximum absolute difference 1e-4):

```sh
IZWI_FISH_CODEC_PATH=/models/FishAudio-S2-Pro/codec.pth \
IZWI_FISH_CODEBOOK_FIXTURE=/fixtures/fish-codes.json \
cargo test --locked -p izwi-core --features cuda,cudnn,flash-attn --lib \
  cuda_real_codec_streaming_matches_frozen_codebooks -- --ignored --nocapture
```

The real runtime test requires multiple nonempty PCM chunks before the runtime
future completes, ordered sequences, frame/sample equality and terminal stats.
It records TTFA, gaps and RTF; it does not certify audible quality or independently
observe the internal AR completion frontier.

```sh
IZWI_FISH_S2_MODEL_DIR=/models/FishAudio-S2-Pro \
IZWI_FISH_S2_BACKEND=cuda \
IZWI_FISH_S2_REFERENCE_WAV=/fixtures/reference.wav \
IZWI_FISH_S2_REFERENCE_TEXT='Exact matching transcript' \
IZWI_FISH_S2_SMOKE_OUTPUT_WAV=/tmp/fish-stream.wav \
cargo test --locked -p izwi-core --features cuda,cudnn,flash-attn --lib \
  fish_s2_real_runtime_streaming_emits_pcm_before_completion -- --ignored --nocapture
```

### Deployment measurement gate

Run against an authorized test deployment: the POST creates speech history.
The request file is the exact speech-history body with `stream: true`. Metadata must name
`deployed_sha`, `gpu`, `dtype`, `runtime_versions`, `checkpoint`, `build_features`,
`attention_provider`, `effective_context`, `concurrency`, and `cache_state`.

```sh
python3 scripts/bench/run-fish-streaming-benchmark.py \
  --url https://YOUR-TEST-SERVER/v1/text-to-speech \
  --request /tmp/fish-request.json --metadata /tmp/fish-deployment.json \
  --output /tmp/fish-stream-evidence
```

Use the deployed route's actual URL. The runner measures nonempty PCM arrival,
validates sequence/sample accounting and retains audio and server timing. Run
paired warm and cold cases with fixed content/sampling, short and long references,
and supported concurrency. Compare against the baseline build on the same GPU;
report failures, output duration and quality, not only successful-run averages.

Production qualification remains required for every advertised backend. Before
release, run actual checkpoint parity, end-to-end SSE and browser playback,
listening/transcription, slow-consumer and in-flight cancellation checks, and
peak-memory/lease-return measurements. A portable pass or successful build does
not certify CUDA/Metal audio or speed.

CUDA graphs, device sampling, codec precision/kernel changes, AR/codec overlap,
multimodal prefix sharing and tensor batching remain gated on these profiles.
The report's H200 RTF 0.195 is not a result achieved by this implementation. No
throughput or TTFA improvement should be claimed without paired device evidence.


### Local implementation verification

On macOS arm64, the core suite passed 2,567 tests (nine ignored; two optional
local LFM weight-loading tests excluded). All 405 server tests passed with the host physical-memory query available.
The selected browser playback, route,
creation-dialog and API suites passed 67 tests; TypeScript and targeted ESLint
also passed. The six offline streaming benchmark fixtures passed. The small
Metal incremental-codec parity and rollback probe passed on the actual device.
Repository hygiene passed: workspace Clippy with warnings denied, all-target
compilation, guarded formatting and shell/evidence/concurrency fixtures.

These results do not include full-checkpoint CUDA codec parity, real-weight
runtime streaming, live browser audio, production peak-memory measurement or a
paired performance benchmark. Those remain deployment qualification gates.
