#!/usr/bin/env python3
"""Opt-in Fish CUDA API replay. Portable checks do not certify hardware or quality."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
import math
import pathlib
import re
import time
import urllib.parse
import wave

# Share the existing bounded HTTP transport (raw responses and HTTP headers are
# retained even on failure). No requests are made until main() is called.
_spec = importlib.util.spec_from_file_location(
    'lfm_transport', pathlib.Path(__file__).with_name('run-lfm-cuda-replay.py'))
_transport = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_transport)
Client = _transport.Client
MODEL = 'FishAudio-S2-Pro'
TEXT = 'Tell me something funny, I stay laughing'
VOICE = '55fc6c39-ed9d-495e-93fb-87ea76d3a955'
TERMINAL = {'completed', 'failed'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def validate_metadata(metadata):
    required = ('deployed_sha', 'gpu', 'backend', 'dtype', 'effective_context',
                'service_concurrency', 'runtime_versions')
    for key in required:
        if not metadata.get(key):
            raise ValueError(f'metadata requires {key}')
    if not re.fullmatch(r'[0-9a-fA-F]{40}', metadata['deployed_sha']):
        raise ValueError('deployed_sha must be the full deployed Git SHA')
    if metadata['backend'].lower() != 'cuda':
        raise ValueError('metadata backend must be CUDA')
    for key in ('effective_context', 'service_concurrency'):
        if type(metadata[key]) is not int or metadata[key] < 1:
            raise ValueError(f'{key} must be a positive integer')


def validate_audio(raw, reported_duration):
    with wave.open(io.BytesIO(raw), 'rb') as audio:
        frames, rate = audio.getnframes(), audio.getframerate()
        width, channels = audio.getsampwidth(), audio.getnchannels()
        decoded = audio.readframes(frames)
        if frames <= 0 or rate <= 0 or len(decoded) != frames * channels * width:
            raise ValueError('WAV has empty or truncated PCM audio')
        duration = frames / rate
    if (isinstance(reported_duration, bool) or not isinstance(reported_duration, (int, float))
            or not math.isfinite(reported_duration) or reported_duration <= 0):
        raise ValueError('record duration must be finite and positive')
    if abs(duration - reported_duration) > max(0.1, duration * 0.02):
        raise ValueError('record duration does not match decoded WAV')
    return {'duration_secs': duration, 'frames': frames, 'sample_rate': rate,
            'channels': channels, 'sha256': hashlib.sha256(raw).hexdigest()}


def run_case(client, body, output, timeout, poll_interval, cancel=False):
    output.mkdir()
    result = {'passed': False, 'cancel_requested': cancel}
    try:
        record = json.loads(client.request('/v1/text-to-speech', output / 'created.json', body))
        result['record_id'] = record['id']
        member = '/v1/text-to-speech/' + urllib.parse.quote(record['id'], safe='')
        if cancel:
            cancellation = json.loads(client.request(member + '/cancel', output / 'cancel.json', {}))
            if cancellation.get('cancelled') is not True:
                raise RuntimeError('cancellation was not acknowledged')
        deadline = time.monotonic() + timeout
        attempt = 0
        while record.get('processing_status') not in TERMINAL:
            if time.monotonic() >= deadline:
                raise TimeoutError('record did not reach terminal status before polling deadline')
            record = json.loads(client.request(member, output / f'poll-{attempt:04d}.json'))
            attempt += 1
            if record.get('processing_status') not in TERMINAL:
                time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
        result['terminal_status'] = record['processing_status']
        write_json(output / 'terminal.json', record)
        # The speech-history API represents an acknowledged cancellation as
        # failed with this reason (there is no cancelled record status).
        expected = 'failed' if cancel else 'completed'
        if record['processing_status'] != expected:
            raise RuntimeError(f"expected {expected}: {record.get('processing_error') or record['processing_status']}")
        if cancel and record.get('processing_error') != 'Cancelled by speech history request':
            raise RuntimeError('failed record is not the acknowledged cancellation')
        if not cancel:
            raw = client.request(member + '/audio', output / 'audio.wav')
            result['audio'] = validate_audio(raw, record.get('audio_duration_secs'))
        result['passed'] = True
    except Exception as error:
        result['error'] = str(error)
    write_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, help='Origin only, e.g. http://127.0.0.1:8080')
    parser.add_argument('--metadata', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path, help='New private evidence directory')
    parser.add_argument('--saved-voice-id', default=VOICE)
    parser.add_argument('--text', default=TEXT)
    parser.add_argument('--sequential', type=int, default=3)
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--poll-interval', type=float, default=1)
    parser.add_argument('--cancel', action='store_true', help='Also require an acknowledged cancellation')
    parser.add_argument('--metrics-path', help='Optional exposed metrics endpoint; snapshots require manual lease review')
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    validate_metadata(metadata)
    url = urllib.parse.urlsplit(args.server)
    if url.scheme not in ('http', 'https') or not url.hostname or url.path not in ('', '/') or url.query or url.fragment or url.username:
        parser.error('--server must be an HTTP(S) origin without credentials')
    if args.sequential < 2 or not 1 <= args.concurrency <= metadata['service_concurrency']:
        parser.error('use at least two sequential requests and concurrency within service_concurrency')
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.poll_interval) or args.poll_interval <= 0:
        parser.error('timeout and poll interval must be finite and positive')
    if args.metrics_path and (not args.metrics_path.startswith('/') or args.metrics_path.startswith('//')):
        parser.error('--metrics-path must be an absolute endpoint path')
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(args.output / 'metadata.operator-supplied.json', metadata)
    client = Client(args.server, args.timeout)
    summary = {'automatic_checks_passed': False, 'cases': [], 'hardware_qualified': False,
               'manual_gates': ['Verify metadata against deployed build/runtime logs',
                                'Compare request/artifact ownership baseline after completion and cancellation',
                                'Listen for intelligibility, voice similarity, repetition and clipping'],
               'cancellation_tested': args.cancel}
    try:
        voice_path = '/v1/voices/' + urllib.parse.quote(args.saved_voice_id, safe='')
        voice = json.loads(client.request(voice_path, args.output / 'saved-voice.json'))
        reference = client.request(voice_path + '/audio', args.output / 'reference.audio')
        if not voice.get('reference_text') or not reference:
            raise ValueError('saved voice needs matching reference text and nonempty audio')
        summary['reference_audio_sha256'] = hashlib.sha256(reference).hexdigest()
        saved = {'model_id': MODEL, 'text': args.text, 'saved_voice_id': args.saved_voice_id}
        direct = {'model_id': MODEL, 'text': args.text, 'reference_text': voice['reference_text'],
                  'reference_audio': base64.b64encode(reference).decode('ascii')}
        # Warm before baseline so persistent model residency is not mistaken for
        # a leaked per-request reservation.
        summary['cases'].append(run_case(client, saved, args.output / 'warmup', args.timeout, args.poll_interval))
        if args.metrics_path:
            client.request(args.metrics_path, args.output / 'metrics-before.raw')
        for index in range(args.sequential):
            summary['cases'].append(run_case(client, saved, args.output / f'sequential-{index}', args.timeout, args.poll_interval))
        summary['cases'].append(run_case(client, direct, args.output / 'direct-reference', args.timeout, args.poll_interval))
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(run_case, client, saved, args.output / f'concurrent-{index}',
                                       args.timeout, args.poll_interval) for index in range(args.concurrency)]
            summary['cases'].extend(future.result() for future in futures)
        if args.metrics_path:
            client.request(args.metrics_path, args.output / 'metrics-after-completion.raw')
        if args.cancel:
            summary['cases'].append(run_case(client, saved, args.output / 'cancellation', args.timeout, args.poll_interval, cancel=True))
            if args.metrics_path:
                client.request(args.metrics_path, args.output / 'metrics-after-cancellation.raw')
        summary['automatic_checks_passed'] = all(case['passed'] for case in summary['cases'])
    except Exception as error:
        summary['error'] = str(error)
    write_json(args.output / 'summary.json', summary)
    return 0 if summary['automatic_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
