#!/usr/bin/env python3
"""Portable fixtures; no model or CUDA certification."""
import importlib.util
import json
import pathlib
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest

spec = importlib.util.spec_from_file_location('replay', pathlib.Path(__file__).with_name('run-lfm-cuda-replay.py'))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ReplayTests(unittest.TestCase):
    def test_empty_reasoning_and_repetition_gates(self):
        for text in ['', '   ', '<think>hello</think>', '<think>unfinished', '!!!']:
            self.assertIn('no visible answer', replay.assess(text, []))
        self.assertEqual(replay.assess('<think>reason</think>Harare is in Zimbabwe.', []), [])
        text = 'This answer is exactly the same as before.'
        self.assertTrue(replay.assess(text, [text]))
        self.assertTrue(replay.assess('hello world ' * 24, []))

    def test_sse_done_and_errors(self):
        final = {'event': 'done', 'assistant_message': {'content': 'Harare'}, 'stats': {'tokens_generated': 2}}
        raw = ('data: {"event":"delta","delta":"Har"}\n\n' +
               'data: ' + json.dumps(final) + '\n\n').encode()
        self.assertEqual(replay.parse_response(raw, True, 'conversation'), ('Harare', final))
        for bad in [b'data: {"event":"delta","delta":"hi"}\n\n',
                    b'data: {"event":"error","error":"CUDA failure"}\n\n']:
            with self.assertRaises(RuntimeError):
                replay.parse_response(bad, True, 'conversation')

    def test_stateless_requires_completion_and_retains_raw_reasoning_events(self):
        events = [{'choices': [{'delta': {'reasoning_content': 'hidden'}}]},
                  {'choices': [{'delta': {'content': 'Answer'}, 'finish_reason': 'stop'}]}]
        raw = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events).encode()
        with self.assertRaises(RuntimeError):
            replay.parse_response(raw, True, 'stateless')
        text, final = replay.parse_response(raw + b'data: [DONE]\n\n', True, 'stateless')
        self.assertEqual(text, 'Answer')
        self.assertEqual(final['events'][0], events[0])

    def test_case_uses_generated_history_and_preserves_raw_empty_failure(self):
        class Client:
            bodies = []
            def request(self, path, destination, body=None, method=None):
                self.bodies.append(body)
                content = 'Victoria Falls is on the Zambezi River.' if len(self.bodies) == 1 else ''
                raw = json.dumps({'choices': [{'message': {'content': content}}]}).encode()
                destination.write_bytes(raw)
                return raw
        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            case = {'name': 'fixture', 'model': 'test', 'turns': ['Tell me about Victoria Falls', 'Where is it']}
            output = pathlib.Path(directory) / 'case'
            result = replay.run_case(client, case, False, 'stateless', output, 100, 0)
            self.assertFalse(result['passed'])
            self.assertEqual(client.bodies[1]['messages'][1]['content'],
                             'Victoria Falls is on the Zambezi River.')
            self.assertTrue((output / '1.response.raw').is_file())
            self.assertIn('no visible answer', result['failures'][0])

    def test_saved_history_and_transport_failure_stop_without_fake_reply(self):
        class Client:
            def request(self, path, destination, body=None, method=None):
                self.body = body
                destination.write_bytes(b'partial response')
                raise TimeoutError('deadline')
        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            history = [{'role': 'user', 'content': 'saved question'},
                       {'role': 'assistant', 'content': 'saved answer'}]
            case = {'name': 'fixture', 'model': 'test', 'turns': ['new question', 'not sent'], 'history': history}
            result = replay.run_case(client, case, True, 'stateless', pathlib.Path(directory) / 'case', 100, 0)
            self.assertFalse(result['passed'])
            self.assertEqual(len(result['turns']), 1)
            self.assertEqual(client.body['messages'][:2], history)


    def test_http_transport_preserves_sse_and_bounds_keepalive_stream(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                try:
                    for _ in range(100 if self.path == '/slow' else 1):
                        self.wfile.write(b'data: {"event":"delta","delta":"hello"}\n\n')
                        self.wfile.flush()
                        if self.path == '/slow':
                            time.sleep(.01)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / 'response.raw'
                origin = f'http://127.0.0.1:{server.server_port}'
                raw = replay.Client(origin, 1).request('/ok', path)
                self.assertEqual(path.read_bytes(), raw)
                self.assertIn(b'hello', raw)
                started = time.monotonic()
                with self.assertRaises((TimeoutError, OSError)):
                    replay.Client(origin, .08).request('/slow', path)
                self.assertLess(time.monotonic() - started, .8)
                self.assertIn(b'hello', path.read_bytes())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
