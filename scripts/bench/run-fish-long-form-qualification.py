#!/usr/bin/env python3
"""Opt-in long speech transport qualification. Creates real jobs; never certifies quality.

Build approximate narration workloads, run mixed short/long concurrent SSE requests,
then compare each successful stream with THAT job's saved WAV in bounded blocks.
No comparison is made between independently sampled requests.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import urllib.parse
import urllib.request
import wave

spec = importlib.util.spec_from_file_location('fish_stream_bench', Path(__file__).with_name('run-fish-streaming-benchmark.py'))
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

PASSAGES = {
    'en': 'We walked through the quiet garden and listened to the birds. Each story has a beginning, a middle, and an ending. Please remember this sentence as the narration continues.',
    'zh': '我们走过安静的花园，听见了鸟儿的歌声。每个故事都有开始、过程和结尾。请记住这句话，然后继续听下面的内容。',
}


def workloads(template, minutes=(1, 10, 30, 60, 120)):
    """Duration labels are text-size estimates, never expected exact output lengths."""
    result = []
    for language, passage in PASSAGES.items():
        units = len(passage.split()) if language == 'en' else len(passage)
        units_per_minute = 150 if language == 'en' else 300
        for duration in minutes:
            if type(duration) is not int or duration <= 0:
                raise ValueError('minutes must be positive integers')
            repeats = math.ceil(duration * units_per_minute / units)
            # Numbered units make omitted/repeated passages identifiable during ASR review.
            text = '\n\n'.join(f'{index + 1}. {passage}' for index in range(repeats))
            for kind, content in [('long', text), ('short', passage)]:
                body = dict(template, text=content, stream=True)
                # An inherited explicit output cap would invalidate complete-text qualification.
                body.pop('max_tokens', None)
                body.pop('max_output_tokens', None)
                result.append({'language': language, 'kind': kind,
                               'estimated_minutes': duration if kind == 'long' else None,
                               'input_sha256': hashlib.sha256(content.encode()).hexdigest(),
                               'request': body})
    return result


def compare_pcm(saved_wav, streamed_pcm, expected_rate):
    digest = hashlib.sha256()
    samples = 0
    with wave.open(str(saved_wav), 'rb') as wav, Path(streamed_pcm).open('rb') as pcm:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, expected_rate, 'NONE'):
            raise ValueError('saved WAV format differs from streamed mono PCM16')
        while raw := wav.readframes(32768):
            if raw != pcm.read(len(raw)):
                raise ValueError('saved WAV samples differ from this job stream')
            digest.update(raw)
            samples += len(raw) // 2
        if pcm.read(1) or not samples or samples != wav.getnframes():
            raise ValueError('saved WAV and stream sample counts differ or are empty')
    return {'pcm_sha256': digest.hexdigest(), 'samples': samples, 'sample_rate': expected_rate,
            'stream_matches_saved': True}


def record_id(directory):
    found = None
    with (directory / 'events.jsonl').open() as events:
        for line in events:
            event = json.loads(line)['event']
            record = event.get('record')
            if isinstance(record, dict) and record.get('id'):
                if found is not None and found != record['id']:
                    raise ValueError('stream changed record identity')
                found = record['id']
    if found is None:
        raise ValueError('stream lacks durable record identity')
    return found


def verify_saved(url, directory, result, timeout, max_bytes):
    identity = urllib.parse.quote(record_id(directory), safe='')
    record_url = url.rstrip('/') + '/' + identity
    with urllib.request.urlopen(record_url, timeout=timeout) as response:
        record = json.load(response)
    if record.get('processing_status') != 'ready' or record.get('processing_error'):
        raise ValueError('stream succeeded without a ready, error-free saved job')
    (directory / 'saved-record.json').write_text(json.dumps(record, indent=2) + '\n')
    target = directory / 'saved.wav'
    total = 0
    with urllib.request.urlopen(record_url + '/audio', timeout=timeout) as response, target.open('xb') as sink:
        while block := response.read(65536):
            total += len(block)
            if total > max_bytes:
                raise ValueError('saved artifact exceeds qualification download byte limit')
            sink.write(block)
    return compare_pcm(target, directory / 'audio.pcm', result['sample_rate'])


def class_metrics(rows):
    result = {}
    for kind in ('short', 'long'):
        selected = [row for row in rows if row['kind'] == kind]
        completed = [row for row in selected if row['passed']]
        latencies = [row['client_receipt_ttfa_ms'] for row in completed
                     if row.get('client_receipt_ttfa_ms') is not None]
        result[kind] = {'requests': len(selected), 'passed': len(completed),
                        'failed': len(selected) - len(completed),
                        'ttfa_sample_count': len(latencies),
                        'ttfa_p95_ms': benchmark.percentile(latencies, .95),
                        'ttfa_p99_ms': benchmark.percentile(latencies, .99)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', required=True, type=Path, help='Speech-history JSON request with model and authorized saved voice')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--url', help='Omit to generate workloads without contacting any server')
    parser.add_argument('--metadata', type=Path, help='Required for a real run; same metadata as Fish streaming benchmark')
    parser.add_argument('--minutes', type=int, nargs='+', default=[1, 10, 30, 60, 120])
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=300, help='Socket idle timeout, not whole-job deadline')
    parser.add_argument('--max-download-bytes', type=int, default=1024 * 1024 * 1024)
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_download_bytes < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('concurrency, download bytes and timeout must be positive')
    template = json.loads(args.template.read_text())
    if not isinstance(template, dict):
        parser.error('template must be an object')
    matrix = workloads(template, args.minutes)
    metadata = None
    if args.url:
        if not args.metadata:
            parser.error('--metadata is required for live qualification')
        metadata = json.loads(args.metadata.read_text())
        for key in ('deployed_sha', 'gpu', 'dtype', 'runtime_versions', 'checkpoint', 'build_features', 'attention_provider', 'effective_context', 'cache_state'):
            if key not in metadata:
                parser.error(f'metadata missing {key}')
        metadata['concurrency'] = args.concurrency
    args.output.mkdir(parents=True, exist_ok=False)
    bodies = [row['request'] for row in matrix]
    (args.output / 'workloads.json').write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'request.json').write_text(json.dumps(bodies, ensure_ascii=False) + '\n')
    if not args.url:
        print('Workloads generated; no inference performed.')
        return
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    summary = benchmark.run_load(args.url, bodies, args.output, args.timeout, len(bodies), args.concurrency)
    evidence = []
    for index, workload in enumerate(matrix):
        directory = args.output / f'request-{index:06d}'
        result = json.loads((directory / 'result.json').read_text())
        row = {key: value for key, value in workload.items() if key != 'request'}
        row.update(request_index=index, passed=False, outcome=result['outcome'])
        for key in ('client_receipt_ttfa_ms', 'request_wall_ms', 'audio_duration_secs', 'interchunk_gap_ms'):
            row[key] = result.get(key)
        try:
            if not result['passed']:
                raise ValueError(result.get('error', 'generation failed'))
            row.update(verify_saved(args.url, directory, result, args.timeout, args.max_download_bytes))
            row['passed'] = True
        except Exception as error:
            row['error'] = str(error)
        evidence.append(row)
    report = {'transport_passed': summary['passed'] and all(row['passed'] for row in evidence),
              'production_certified': False, 'quality_review': 'required', 'cuda_memory_soak': 'required',
              'restart_and_cancel_fault_injection': 'required', 'requests': evidence,
              'by_request_kind': class_metrics(evidence)}
    (args.output / 'long-form-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['transport_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
