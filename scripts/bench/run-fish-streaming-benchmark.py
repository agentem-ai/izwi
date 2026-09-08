#!/usr/bin/env python3
"""Measure real PCM receipt over speech-history SSE, without timing metadata as audio.

Requires --metadata JSON describing deployed_sha, gpu, dtype, runtime_versions,
checkpoint, build_features, attention_provider, effective_context, concurrency,
and cache_state. This is an opt-in hardware benchmark; fixtures make no speed claim.
"""
import argparse
import base64
import json
import math
from pathlib import Path
import time
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Full speech-history streaming POST URL')
    parser.add_argument('--request', required=True, type=Path, help='Exact request body JSON')
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    required = ('deployed_sha', 'gpu', 'dtype', 'runtime_versions', 'checkpoint',
                'build_features', 'attention_provider', 'effective_context', 'concurrency', 'cache_state')
    for key in required:
        if key not in metadata:
            parser.error(f'metadata missing {key}')
    body = json.loads(args.request.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (args.output / 'request.json').write_text(json.dumps(body, indent=2))
    started = time.monotonic()
    result = {'passed': False}
    try:
        request = urllib.request.Request(args.url, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
        with (args.output / 'audio.pcm').open('wb') as pcm, (args.output / 'events.jsonl').open('w') as log:
            measurement = Measurement(pcm)
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                    raise ValueError('response is not SSE')
                for event in events(response):
                    elapsed = (time.monotonic() - started) * 1000
                    log.write(json.dumps({'elapsed_ms': elapsed, 'event': event}) + '\n')
                    measurement.accept(event, elapsed)
            result.update(measurement.result())
        with wave.open(str(args.output / 'audio.wav'), 'wb') as wav, (args.output / 'audio.pcm').open('rb') as pcm:
            wav.setparams((1, 2, measurement.rate, 0, 'NONE', 'not compressed'))
            while block := pcm.read(65536):
                wav.writeframesraw(block)
        result['passed'] = True
    except Exception as error:
        result['error'] = str(error)
    result['request_wall_ms'] = (time.monotonic() - started) * 1000
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        raise SystemExit(result['error'])


if __name__ == '__main__':
    main()
