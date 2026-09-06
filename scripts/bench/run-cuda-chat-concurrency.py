#!/usr/bin/env python3
"""Bounded, uncapped-output CUDA chat concurrency evidence (Python stdlib only)."""
import argparse
import http.client
import json
import pathlib
import socket
import subprocess
import threading
import time
import urllib.parse


def delta_text(event):
    if event.get('error') or event.get('event') == 'error':
        raise RuntimeError(f'server stream error: {event}')
    if event.get('event') == 'delta':
        return event.get('delta', '')
    return ''.join(choice.get('delta', {}).get('content', '') or
                   choice.get('delta', {}).get('reasoning_content', '') or ''
                   for choice in event.get('choices', []))


def assess(records, before, after, width):
    """Intervals are observed text deltas, not guessed token counts or HTTP overlap."""
    failures = []
    if len(records) != width:
        failures.append('missing request records')
    if any(r.get('error') or len(r['deltas']) < 2 for r in records):
        failures.append('each stream must deliver at least two nonempty text deltas without error')
    overlap = (min((r['deltas'][-1] for r in records if r['deltas']), default=0) -
               max((r['deltas'][0] for r in records if r['deltas']), default=0))
    if width > 1 and overlap <= 0:
        failures.append('no shared interval of observed text generation')
    calls = after['engine']['model_tensor_multirow_calls_total'] - before['engine']['model_tensor_multirow_calls_total']
    histogram = 'model_tensor_batch_width_counts'
    before_widths = before['engine'].get(histogram, {})
    after_widths = after['engine'].get(histogram, {})
    width_calls = sum(value - before_widths.get(key, 0)
                      for key, value in after_widths.items() if int(key) >= width)
    if width > 1 and (calls < 2 or width_calls < 2):
        failures.append(f'need at least two actual model forwards with width >= {width}')
    return {'passed': not failures, 'failures': failures,
            'shared_text_interval_seconds': max(0, overlap),
            'multirow_calls_delta': calls, 'requested_width_calls_delta': width_calls}


def cleanup_drained(snapshot, baseline):
    engine, original = snapshot['engine'], baseline['engine']
    if engine['scheduler_running_requests'] or engine['scheduler_queue_depth']:
        return False
    current = engine['kv_cache']['totals']
    initial = original['kv_cache']['totals']
    if current['registered_sessions'] != initial['registered_sessions']:
        return False
    # Prefix cache retention is legitimate; active request ownership is not.
    return all(current['coordinator'][key] == initial['coordinator'][key]
               for key in ('admission_claimed_pages', 'admission_claims', 'table_refs',
                           'execution_pins', 'transfer_pins', 'reservations', 'active_transactions'))


class Client:
    def __init__(self, server, timeout):
        self.url = urllib.parse.urlsplit(server)
        self.timeout = timeout

    def connection(self):
        cls = http.client.HTTPSConnection if self.url.scheme == 'https' else http.client.HTTPConnection
        return cls(self.url.hostname, self.url.port, timeout=self.timeout)

    def json(self, path, body=None, method=None):
        conn = self.connection()
        try:
            conn.request(method or ('POST' if body is not None else 'GET'), path,
                         json.dumps(body) if body is not None else None,
                         {'Content-Type': 'application/json'})
            response = conn.getresponse()
            data = response.read()
            if response.status >= 300:
                raise RuntimeError(f'{path}: HTTP {response.status}: {data[:500]!r}')
            return json.loads(data)
        finally:
            conn.close()

    def stream(self, path, body, stop, record):
        conn = self.connection()
        active_socket = [None]
        aborted = threading.Event()
        record.update(deltas=[], started=time.monotonic(), body=body)
        def abort():
            stop.wait(self.timeout)
            aborted.set()
            if active_socket[0]:
                try:
                    active_socket[0].shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        watcher = threading.Thread(target=abort, daemon=True)
        watcher.start()
        try:
            conn.connect()
            active_socket[0] = conn.sock
            if aborted.is_set() or stop.is_set():
                if not stop.is_set():
                    raise TimeoutError('stream deadline expired while connecting')
                record['cancelled'] = True
                return
            conn.request('POST', path, json.dumps(body), {'Content-Type': 'application/json'})
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f'HTTP {response.status}: {response.read()[:500]!r}')
            while not stop.is_set():
                line = response.readline().decode('utf-8').strip()
                if not line:
                    if response.isclosed():
                        break
                    continue
                if not line.startswith('data:'):
                    continue
                payload = line[5:].strip()
                if payload == '[DONE]':
                    record['terminal'] = True
                    break
                event = json.loads(payload)
                if delta_text(event):
                    record['deltas'].append(time.monotonic())
                if event.get('event') == 'done':
                    record['terminal'] = True
            record['cancelled'] = stop.is_set()
            if not record.get('terminal') and not stop.is_set():
                record['error'] = 'stream ended without a terminal event'
        except Exception as error:
            if not stop.is_set():
                record['error'] = str(error)
            else:
                record['cancelled'] = True
        finally:
            record['ended'] = time.monotonic()
            conn.close()


def run_case(client, args, route, width, staggered=False):
    records, threads, conversation_ids = [], [], []
    stops = [threading.Event() for _ in range(width)]
    before = client.json('/v1/metrics')
    samples = []
    deadline = time.monotonic() + args.timeout
    case_error = None
    cancelled_first = survivors_advanced = False
    try:
        for index in range(width):
            if staggered and index == width - 1:
                while not all(len(r['deltas']) >= 2 for r in records):
                    if time.monotonic() >= deadline:
                        raise RuntimeError('existing requests did not decode before late arrival')
                    time.sleep(.05)
            prompt = f'Concurrency evidence request {time.time_ns()}-{index}. Enumerate the integers from 1 to 100000, one per line, without skipping any.'
            body = {'model': args.model, 'stream': True, 'temperature': 0}
            path = '/v1/chat/completions'
            if route == 'conversation':
                created = client.json('/v1/chat/threads', {'title': 'CUDA concurrency evidence', 'model_id': args.model})
                conversation_id = created['id']
                conversation_ids.append(conversation_id)
                path = f'/v1/chat/threads/{conversation_id}/messages'
                body['content'] = prompt
            else:
                body['messages'] = [{'role': 'user', 'content': prompt}]
            record = {'deltas': [], 'path': path}
            records.append(record)
            thread = threading.Thread(target=client.stream, args=(path, body, stops[index], record))
            threads.append(thread)
            thread.start()
        cancelled_first = False
        survivors_advanced = False
        while time.monotonic() < deadline:
            sample = {'at': time.monotonic(), 'metrics': client.json('/v1/metrics')}
            gpu = subprocess.run([args.nvidia_smi, '--query-gpu=uuid,name,memory.used,memory.free,memory.total',
                                  '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            if gpu.returncode or not gpu.stdout.strip():
                raise RuntimeError('nvidia-smi resource sampling failed')
            sample['nvidia_smi'] = gpu.stdout
            samples.append(sample)
            enough = all(len(r['deltas']) >= args.events for r in records)
            if enough and staggered and not cancelled_first:
                if not threads[0].is_alive():
                    raise RuntimeError('first stream finished before cancellation could be tested')
                # Disconnect one active client and prove surviving streams keep making progress.
                stops[0].set()
                survivor_counts = [len(r['deltas']) for r in records[1:]]
                cancelled_first = True
            elif enough and (not staggered or all(len(r['deltas']) >= count + 2
                                                   for r, count in zip(records[1:], survivor_counts))):
                survivors_advanced = True
                break
            if all(not t.is_alive() for t in threads):
                break
            time.sleep(.2)
    except Exception as error:
        case_error = str(error)
    finally:
        for stop in stops:
            stop.set()
        for thread in threads:
            thread.join(args.timeout + 1)
    # Cancellation cleanup must restore the idle baseline before the next case.
    cleanup_deadline = time.monotonic() + args.timeout
    while True:
        after = client.json('/v1/metrics')
        if cleanup_drained(after, before):
            break
        if time.monotonic() >= cleanup_deadline:
            case_error = 'cancellation did not drain requests and release active cache ownership'
            break
        time.sleep(.2)
    for conversation_id in conversation_ids:
        client.json(f'/v1/chat/threads/{conversation_id}', method='DELETE')
    verdict = assess(records, before, after, width)
    if not samples:
        verdict['passed'] = False
        verdict['failures'].append('missing resource samples')
    if case_error:
        verdict['passed'] = False
        verdict['failures'].append(case_error)
    if staggered and not (cancelled_first and survivors_advanced):
        verdict['passed'] = False
        verdict['failures'].append('late-arrival cancellation/survivor progress was not exercised')
    return {'route': route, 'concurrent': width, 'staggered': staggered,
            'records': records, 'before': before, 'after': after, 'samples': samples, **verdict}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='http://127.0.0.1:8080')
    parser.add_argument('--model', default='Qwen3.8-27B-FP8')
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('target/cuda-chat-concurrency'))
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--events', type=int, default=32, help='text deltas per stream before controlled cancellation')
    parser.add_argument('--extended', action='store_true', help='also require c4/c8')
    parser.add_argument('--allow-remote', action='store_true')
    parser.add_argument('--nvidia-smi', default='nvidia-smi', help='must observe the server host GPUs')
    args = parser.parse_args()
    if args.timeout <= 0 or args.events < 4:
        parser.error('timeout must be positive and events >= 4')
    url = urllib.parse.urlsplit(args.server)
    if url.scheme not in ('http', 'https') or url.path not in ('', '/') or url.query or url.fragment:
        parser.error('server must be an HTTP(S) origin without path/query/fragment')
    if urllib.parse.urlsplit(args.server).hostname not in ('localhost', '127.0.0.1', '::1') and not args.allow_remote:
        parser.error('non-loopback server requires --allow-remote')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'izwi.cuda-chat-concurrency.v1', 'status': 'failed', 'cases': []}
    try:
        repo = pathlib.Path(__file__).resolve().parents[2]
        sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        if dirty:
            raise RuntimeError('exact-SHA evidence requires a clean tracked worktree')
        client = Client(args.server, args.timeout)
        health = client.json('/v1/health')
        runtime = health['runtime']
        if (runtime['build_git_sha'] != sha or runtime['selected_backend'] != 'cuda' or
                not runtime.get('compiled_backends', {}).get('cuda') or
                not runtime.get('cuda_runtime', {}).get('device_usable')):
            raise RuntimeError('server must run the checked-out exact SHA on CUDA')
        initial = client.json('/v1/metrics')
        if initial['engine']['scheduler_running_requests'] or initial['engine']['scheduler_queue_depth']:
            raise RuntimeError('evidence requires an idle dedicated server')
        models = [m for m in initial['models'] if m['variant_id'] == args.model and m['actual_device_kind'] == 'cuda']
        if len(models) != 1:
            raise RuntimeError('requested model must already be loaded on CUDA')
        report.update(git_sha=sha, health=health, model=models[0], server=args.server)
        for route in ('stateless', 'conversation'):
            for width in [1, 2, 3] + ([4, 8] if args.extended else []):
                report['cases'].append(run_case(client, args, route, width))
                if not report['cases'][-1]['passed']:
                    raise RuntimeError(f'{route} c{width} failed acceptance')
            report['cases'].append(run_case(client, args, route, 3, staggered=True))
            if not report['cases'][-1]['passed']:
                raise RuntimeError(f'{route} late arrival failed acceptance')
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"{report['status']}: {args.output / 'report.json'}")
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
