#!/usr/bin/env python3
"""Measure real PCM receipt over speech-history SSE, without timing metadata as audio.

Requires --metadata JSON describing deployed_sha, gpu, dtype, runtime_versions,
checkpoint, build_features, attention_provider, effective_context, concurrency,
and cache_state. This is an opt-in hardware benchmark; fixtures make no speed claim.
"""
import argparse
import base64
import concurrent.futures
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request
import wave


def events(lines):
    """SSE supports comments, CRLF and multiline data fields."""
    data = []
    for raw in lines:
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                yield json.loads('\n'.join(data))
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
    if data:
        yield json.loads('\n'.join(data))


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lo = math.floor(index)
    hi = math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


class Measurement:
    def __init__(self, pcm_sink):
        self.sink = pcm_sink
        self.arrivals = []
        self.samples = 0
        self.rate = None
        self.terminal = None
        self.next_sequence = 0

    def accept(self, event, elapsed_ms):
        kind = event.get('event')
        if kind == 'error':
            raise ValueError(event.get('error', 'stream failed'))
        if self.terminal is not None:
            if kind == 'done':
                return
            raise ValueError('event after terminal result')
        if kind == 'chunk':
            raw = base64.b64decode(event['audio_base64'], validate=True)
            count = event['sample_count']
            rate = event['sample_rate']
            if type(count) is not int or count <= 0 or len(raw) != count * 2:
                raise ValueError('invalid PCM sample count')
            if event.get('audio_format') != 'pcm_i16' or type(rate) is not int or rate <= 0:
                raise ValueError('invalid PCM format or rate')
            if event.get('sequence') != self.next_sequence:
                raise ValueError('non-monotonic chunk sequence')
            if self.rate is not None and rate != self.rate:
                raise ValueError('sample rate changed')
            self.rate = rate
            self.next_sequence += 1
            self.samples += count
            self.arrivals.append(elapsed_ms)
            self.sink.write(raw)
        elif kind == 'final':
            self.terminal = event

    def result(self):
        if self.terminal is None or not self.samples:
            raise ValueError('stream ended without final metadata or PCM')
        duration = self.samples / self.rate
        reported = self.terminal.get('audio_duration_secs')
        if not isinstance(reported, (int, float)) or not math.isfinite(reported) or abs(reported - duration) > max(1 / self.rate, duration * 1e-5):
            raise ValueError('terminal duration does not match PCM')
        tokens = self.terminal.get('tokens_generated')
        if type(tokens) is not int or tokens < 0:
            raise ValueError('missing committed terminal token count')
        gaps = [b - a for a, b in zip(self.arrivals, self.arrivals[1:])]
        return {
            'client_receipt_ttfa_ms': self.arrivals[0],
            'client_request_to_last_pcm_ms': self.arrivals[-1],
            'client_request_to_last_pcm_rtf': self.arrivals[-1] / 1000 / duration,
            'audio_duration_secs': duration,
            'pcm_samples': self.samples,
            'sample_rate': self.rate,
            'pcm_chunks': len(self.arrivals),
            'interchunk_gap_ms': {f'p{int(q * 100)}': percentile(gaps, q) for q in (.5, .95, .99)},
            'tokens_generated': tokens,
            'server_timing': self.terminal.get('timing'),
            'terminal': self.terminal,
            'first_audible_playback_ms': None,
            'playback_underruns': None,
        }


def run_request(url, body, output, timeout, scheduled_at=None):
    """One independent connection and PCM sink; retain partial failures as evidence."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'passed': False, 'outcome': 'failed',
              'generator_delay_ms': max(0, (started - (scheduled_at or started)) * 1000)}
    measurement = None
    try:
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
        with (output / 'audio.pcm').open('wb') as pcm, (output / 'events.jsonl').open('w') as log:
            measurement = Measurement(pcm)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result['http_status'] = response.status
                if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                    raise ValueError('response is not SSE')
                for event in events(response):
                    elapsed = (time.monotonic() - started) * 1000
                    # PCM is already saved separately; do not duplicate it in JSON logs.
                    log.write(json.dumps({'elapsed_ms': elapsed, 'event': {
                        key: value for key, value in event.items() if key != 'audio_base64'}}) + '\n')
                    measurement.accept(event, elapsed)
            result.update(measurement.result())
        with wave.open(str(output / 'audio.wav'), 'wb') as wav, (output / 'audio.pcm').open('rb') as pcm:
            wav.setparams((1, 2, measurement.rate, 0, 'NONE', 'not compressed'))
            while block := pcm.read(65536):
                wav.writeframesraw(block)
        result.update(passed=True, outcome='completed')
    except urllib.error.HTTPError as error:
        result.update(http_status=error.code, error=str(error),
                      outcome='rejected' if error.code in (429, 503) else 'failed')
        error.close()
    except Exception as error:
        result['error'] = str(error)
    if measurement is not None:
        result['received_pcm_samples'] = measurement.samples
    result['request_wall_ms'] = (time.monotonic() - started) * 1000
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def summarize(results, wall_seconds, max_inflight, mode):
    completed = [row for row in results if row['outcome'] == 'completed']
    counts = {kind: sum(row['outcome'] == kind for row in results)
              for kind in ('completed', 'rejected', 'failed', 'generator_dropped')}
    distributions = {}
    for field in ('client_receipt_ttfa_ms', 'request_wall_ms',
                  'client_request_to_last_pcm_rtf', 'generator_delay_ms'):
        values = [row[field] for row in completed if field in row]
        distributions[field] = {f'p{int(q * 100)}': percentile(values, q) for q in (.5, .95, .99)}
    audio = sum(row['audio_duration_secs'] for row in completed)
    return {'passed': counts['completed'] == len(results), 'mode': mode,
            'offered_requests': len(results), 'sent_requests': len(results) - counts['generator_dropped'],
            'outcomes': counts, 'wall_seconds': wall_seconds, 'peak_client_outstanding': max_inflight,
            'completed_requests_per_second': len(completed) / wall_seconds if wall_seconds else 0,
            'generated_audio_seconds': audio,
            'audio_seconds_per_wall_second': audio / wall_seconds if wall_seconds else 0,
            'distributions': distributions,
            'percentile_sample_count': len(completed),
            'qualification': 'measurement only; requires workload SLOs, server traces and device qualification',
            'first_audible_playback_ms': None, 'playback_underruns': None}


def run_load(url, bodies, output, timeout, count, concurrency, arrival_rate=None):
    """Bound outstanding work, including executor queue; open-loop never waits for capacity.

    At saturation an arrival is recorded as generator_dropped, not a server rejection.
    Closed-loop replaces completed work immediately up to concurrency. Open-loop uses
    absolute monotonic arrival times so service time cannot lower the offered rate.
    """
    if count < 1 or concurrency < 1 or not bodies:
        raise ValueError('requests, concurrency and workload must be positive')
    if arrival_rate is not None and (not math.isfinite(arrival_rate) or arrival_rate <= 0):
        raise ValueError('arrival rate must be finite and positive')
    results = []
    pending = {}
    start = time.monotonic()
    peak = 0

    def collect(done):
        for future in done:
            index = pending.pop(future)
            row = future.result()
            row['request_index'] = index
            results.append(row)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for index in range(count):
            scheduled = start + index / arrival_rate if arrival_rate else time.monotonic()
            if arrival_rate:
                time.sleep(max(0, scheduled - time.monotonic()))
                collect([future for future in pending if future.done()])
                if len(pending) == concurrency:
                    results.append({'request_index': index, 'passed': False,
                                    'outcome': 'generator_dropped', 'scheduled_offset_ms': (scheduled - start) * 1000})
                    continue
            elif len(pending) == concurrency:
                done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                collect(done)
            future = pool.submit(run_request, url, bodies[index % len(bodies)],
                                 output / f'request-{index:06d}', timeout, scheduled)
            pending[future] = index
            peak = max(peak, len(pending))
        collect(list(pending))
    wall = time.monotonic() - start
    results.sort(key=lambda row: row['request_index'])
    (output / 'requests.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in results))
    summary = summarize(results, wall, peak, 'open_loop' if arrival_rate else 'closed_loop')
    summary.update(concurrency_limit=concurrency, arrival_rate_per_second=arrival_rate)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Full speech-history streaming POST URL')
    parser.add_argument('--request', required=True, type=Path, help='Request object or array of workload objects, cycled across arrivals')
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timeout', type=float, default=300, help='Per-socket operation timeout in seconds')
    parser.add_argument('--requests', type=int, default=1)
    parser.add_argument('--concurrency', type=int, help='Actual simultaneous HTTP requests; defaults to metadata concurrency')
    parser.add_argument('--arrival-rate', type=float, help='Open-loop arrivals/second; omitted means closed-loop')
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    required = ('deployed_sha', 'gpu', 'dtype', 'runtime_versions', 'checkpoint',
                'build_features', 'attention_provider', 'effective_context', 'concurrency', 'cache_state')
    for key in required:
        if key not in metadata:
            parser.error(f'metadata missing {key}')
    concurrency = args.concurrency if args.concurrency is not None else metadata['concurrency']
    if type(concurrency) is not int or concurrency < 1 or args.requests < 1:
        parser.error('concurrency and requests must be positive integers')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be finite and positive')
    if args.arrival_rate is not None and (not math.isfinite(args.arrival_rate) or args.arrival_rate <= 0):
        parser.error('arrival-rate must be finite and positive')
    body = json.loads(args.request.read_text())
    bodies = body if isinstance(body, list) else [body]
    if not bodies or any(not isinstance(row, dict) for row in bodies):
        parser.error('request must be an object or a nonempty array of objects')
    args.output.mkdir(parents=True, exist_ok=False)
    metadata.update(concurrency=concurrency, requests=args.requests, arrival_rate_per_second=args.arrival_rate)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (args.output / 'request.json').write_text(json.dumps(body, indent=2))
    summary = run_load(args.url, bodies, args.output, args.timeout,
                       args.requests, concurrency, args.arrival_rate)
    print(json.dumps(summary, indent=2))
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
