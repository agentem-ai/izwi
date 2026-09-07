#!/usr/bin/env python3
"""Replay LFM multi-turn incidents against a running server; retain raw evidence."""
import argparse
import http.client
import json
import pathlib
import re
import shutil
import socket
import threading
import time
import urllib.parse

CASES = [
    {'name': 'thinking', 'model': 'LFM2.5-1.2B-Thinking-GGUF', 'turns': [
        'Hello Qwen', 'How are you', 'What are you talking about', 'huh',
        'Tell me about Harare, ZImbabwe']},
    {'name': 'instruct', 'model': 'LFM2.5-1.2B-Instruct-GGUF', 'turns': [
        'Tell me about victoria falls', 'Where is it',
        'What about South Africa, which major attractions does it have', 'Answer', 'Hi']},
]


def visible_answer(text):
    # Reasoning-only completion is not a visible answer. An unfinished think
    # region is also not evidence of a successfully completed answer.
    return re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.S).strip()


def assess(text, previous):
    answer = visible_answer(text)
    failures = []
    if not answer or not any(char.isalnum() for char in answer):
        failures.append('no visible answer')
    if len(answer) >= 24 and any(answer == visible_answer(old) for old in previous):
        failures.append('exact repeated answer across turns (requires review)')
    words = answer.split()
    if len(words) >= 24 and any(words[-span:] * 4 == words[-4 * span:]
                               for span in range(1, min(32, len(words) // 4) + 1)):
        failures.append('repeated answer suffix (requires review)')
    return failures


def parse_response(raw, stream, route):
    if not stream:
        final = json.loads(raw)
    else:
        events, data = [], []
        for line in raw.decode('utf-8').splitlines() + ['']:
            if not line:
                if data:
                    payload = '\n'.join(data)
                    events.append('[DONE]' if payload == '[DONE]' else json.loads(payload))
                    data = []
            elif line.startswith('data:'):
                data.append(line[5:].lstrip())
        if any(isinstance(event, dict) and (event.get('error') or event.get('event') == 'error')
               for event in events):
            raise RuntimeError('server returned an SSE error; see raw response')
        if route == 'conversation':
            done = [event for event in events if isinstance(event, dict) and event.get('event') == 'done']
            if len(done) != 1:
                raise RuntimeError('stream must have exactly one completed response')
            final = done[0]
        else:
            if not events or events[-1] != '[DONE]':
                raise RuntimeError('stream ended without [DONE]')
            if not any(choice.get('finish_reason') is not None for event in events
                       if isinstance(event, dict) for choice in event.get('choices', [])):
                raise RuntimeError('stream has no finish reason')
            text = ''.join(choice.get('delta', {}).get('content') or '' for event in events
                           if isinstance(event, dict) for choice in event.get('choices', []))
            return text, {'events': events}
    if final.get('error'):
        raise RuntimeError('server returned an error; see raw response')
    text = (final['assistant_message']['content'] if route == 'conversation'
            else final['choices'][0]['message'].get('content') or '')
    return text, final


class Client:
    def __init__(self, server, timeout):
        self.url = urllib.parse.urlsplit(server)
        self.timeout = timeout

    def request(self, path, destination, body=None, method=None):
        cls = http.client.HTTPSConnection if self.url.scheme == 'https' else http.client.HTTPConnection
        conn = cls(self.url.hostname, self.url.port, timeout=self.timeout)
        active = [None]
        expired = threading.Event()
        def abort():
            expired.set()
            if active[0] is not None:
                try:
                    active[0].shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        timer = threading.Timer(self.timeout, abort)
        timer.daemon = True
        timer.start()
        size = 0
        try:
            conn.connect()
            active[0] = conn.sock
            if expired.is_set():
                raise TimeoutError('request deadline expired while connecting')
            conn.request(method or ('POST' if body is not None else 'GET'), path,
                         json.dumps(body) if body is not None else None,
                         {'Content-Type': 'application/json'})
            response = conn.getresponse()
            destination.with_suffix('.headers.json').write_text(json.dumps({
                'status': response.status, 'headers': response.getheaders()}, indent=2))
            with destination.open('wb') as output:
                while True:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    output.write(chunk)
                    size += len(chunk)
                    if size > 8 * 1024 * 1024:
                        raise RuntimeError('response exceeded 8 MiB evidence bound')
            if expired.is_set():
                raise TimeoutError('request deadline expired')
            if not 200 <= response.status < 300:
                raise RuntimeError(f'HTTP {response.status}; see {destination}')
            return destination.read_bytes()
        finally:
            timer.cancel()
            conn.close()

    def snapshot(self, path, destination):
        return json.loads(self.request(path, destination))


def run_case(client, case, stream, route, output, max_tokens, temperature):
    output.mkdir()
    result = {'name': case['name'], 'model': case['model'], 'stream': stream,
              'route': route, 'turns': [], 'failures': []}
    history = case.get('history', []).copy()
    previous = []
    thread_id = None
    try:
        if route == 'conversation':
            if history:
                raise ValueError('saved history requires --route stateless')
            created = json.loads(
                client.request('/v1/chat/threads', output / 'created.json',
                               {'title': 'LFM CUDA diagnostic replay', 'model_id': case['model']}))
            thread_id = created['id']
            result['thread_id'] = thread_id
        for index, prompt in enumerate(case['turns']):
            record = {'index': index, 'prompt': prompt}
            result['turns'].append(record)
            body = {'model': case['model'], 'stream': stream, 'max_tokens': max_tokens,
                    'temperature': temperature}
            history.append({'role': 'user', 'content': prompt})
            path = '/v1/chat/completions'
            if route == 'conversation':
                path = f'/v1/chat/threads/{urllib.parse.quote(thread_id, safe="")}/messages'
                body['content'] = prompt
            else:
                body['messages'] = history.copy()
            (output / f'{index}.request.json').write_text(json.dumps(body, indent=2))
            started = time.monotonic()
            try:
                raw = client.request(path, output / f'{index}.response.raw', body)
                text, final = parse_response(raw, stream, route)
                record.update(text=text, final=final, failures=assess(text, previous))
                history.append({'role': 'assistant', 'content': text})
                previous.append(text)
                result['failures'].extend(f'turn {index}: {item}' for item in record['failures'])
            except Exception as error:
                record['error'] = str(error)
                result['failures'].append(f'turn {index}: {error}')
                break
            finally:
                record['elapsed_seconds'] = time.monotonic() - started
    except Exception as error:
        result['failures'].append(str(error))
    finally:
        if thread_id:
            # Retain persisted final state, then attempt cleanup even if that
            # read fails. Delete only the conversation created by this case.
            path = f'/v1/chat/threads/{urllib.parse.quote(thread_id, safe="")}'
            for filename, method in [('persisted.json', 'GET'), ('deleted.json', 'DELETE')]:
                try:
                    client.request(path, output / filename, method=method)
                except Exception as error:
                    result['failures'].append(f'{method} cleanup failed: {error}')
        result['passed'] = not result['failures']
        (output / 'case.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='http://127.0.0.1:8080')
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--route', choices=['conversation', 'stateless'], default='conversation')
    parser.add_argument('--cases', type=pathlib.Path, help='JSON array: name, model, turns, optional stateless history')
    parser.add_argument('--server-log', type=pathlib.Path, help='same-host server log; copies bytes appended during replay')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0)
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.server)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.path not in ('', '/') or url.query or url.fragment:
        parser.error('server must be an HTTP(S) origin without credentials/path/query/fragment')
    if args.timeout <= 0 or args.max_tokens <= 0:
        parser.error('timeout and max-tokens must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'izwi.lfm-cuda-replay.v1', 'passed': False, 'cases': [],
              'expected_sha': args.expected_sha, 'server': args.server,
              'quality_scope': 'empty/repeated-answer smoke gates; human semantic review required'}
    log_start = None
    try:
        cases = json.loads(args.cases.read_text()) if args.cases else CASES
        (args.output / 'inputs.json').write_text(json.dumps(cases, indent=2))
        if args.server_log:
            log_start = args.server_log.stat().st_size
        client = Client(args.server, args.timeout)
        health = client.snapshot('/v1/health', args.output / 'health.json')
        before = client.snapshot('/v1/metrics', args.output / 'metrics-before.json')
        runtime = health['runtime']
        if (runtime['build_git_sha'] != args.expected_sha or runtime['selected_backend'] != 'cuda'
                or not runtime.get('compiled_backends', {}).get('cuda')
                or not runtime.get('cuda_runtime', {}).get('device_usable')):
            raise RuntimeError('deployment identity/backend differs from requested CUDA build')
        for case in cases:
            loaded = [model for model in before['models'] if model['variant_id'] == case['model']
                      and model['actual_device_kind'] == 'cuda']
            if len(loaded) != 1:
                raise RuntimeError(f'{case["model"]} must already be loaded on CUDA')
        for index, case in enumerate(cases):
            for stream in [False, True]:
                report['cases'].append(run_case(client, case, stream, args.route,
                    args.output / f'{index}-{"stream" if stream else "nonstream"}',
                    args.max_tokens, args.temperature))
        client.snapshot('/v1/metrics', args.output / 'metrics-after.json')
        report['passed'] = bool(report['cases']) and all(case['passed'] for case in report['cases'])
    except Exception as error:
        report['error'] = str(error)
    finally:
        if args.server_log and log_start is not None:
            try:
                with args.server_log.open('rb') as source, (args.output / 'server.log').open('wb') as dest:
                    if source.seek(0, 2) < log_start:
                        raise RuntimeError('server log rotated/truncated during replay')
                    source.seek(log_start)
                    shutil.copyfileobj(source, dest)
            except Exception as error:
                report['log_error'] = str(error)
                report['passed'] = False
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(args.output / 'report.json')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
