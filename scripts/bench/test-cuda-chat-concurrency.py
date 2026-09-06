#!/usr/bin/env python3
"""Deterministic evidence gates and mock SSE transport fixtures."""
import importlib.util
import io
import http.client
import json
import pathlib
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('runner', pathlib.Path(__file__).with_name('run-cuda-chat-concurrency.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def metrics(calls=0, widths=None):
    return {'engine': {'model_tensor_multirow_calls_total': calls,
                       'model_tensor_batch_width_counts': widths or {},
                       'kv_cache': {'totals': {'registered_sessions': 0, 'coordinator': {
                           key: 0 for key in ('admission_claimed_pages', 'admission_claims', 'table_refs',
                                             'execution_pins', 'transfer_pins', 'reservations', 'active_transactions')
                       }}}}}


class Gates(unittest.TestCase):
    def test_three_streams_and_real_width_three(self):
        records = [{'deltas': [n / 10, 1 + n / 10]} for n in range(3)]
        verdict = runner.assess(records, metrics(10, {'3': 4}), metrics(14, {'3': 8}), 3)
        self.assertTrue(verdict['passed'])

    def test_queued_http_connections_are_not_overlap(self):
        records = [{'deltas': [0, 1]}, {'deltas': [2, 3]}]
        self.assertFalse(runner.assess(records, metrics(), metrics(5, {'2': 5}), 2)['passed'])

    def test_old_width_high_water_does_not_certify_current_batch(self):
        records = [{'deltas': [0, 1]}] * 3
        self.assertFalse(runner.assess(records, metrics(10, {'3': 8}), metrics(14, {'3': 8, '2': 4}), 3)['passed'])

    def test_two_rows_cannot_certify_three(self):
        self.assertFalse(runner.assess([{'deltas': [0, 1]}] * 3, metrics(), metrics(5, {'2': 5}), 3)['passed'])

    def test_stream_error_fails_even_with_overlap(self):
        records = [{'deltas': [0, 1], 'error': 'OOM'}, {'deltas': [0, 1]}]
        self.assertFalse(runner.assess(records, metrics(), metrics(5, {'2': 5}), 2)['passed'])

    def test_payloads(self):
        self.assertEqual(runner.delta_text({'event': 'delta', 'delta': 'abc'}), 'abc')
        self.assertEqual(runner.delta_text({'choices': [{'delta': {'reasoning_content': 'abc'}}]}), 'abc')
        with self.assertRaises(RuntimeError):
            runner.delta_text({'event': 'error', 'error': 'capacity'})

    def test_both_routes_omit_limits_and_conversations_are_independent(self):
        class Client:
            def __init__(self):
                self.created = 0
                self.calls = 0
            def json(self, path, body=None, method=None):
                if path == '/v1/chat/threads':
                    self.created += 1
                    return {'id': str(self.created)}
                if method == 'DELETE':
                    return {'deleted': True}
                self.calls += 1
                result = metrics(0 if self.calls == 1 else 3,
                                 {} if self.calls == 1 else {'3': 3})
                result['engine'].update(scheduler_queue_depth=0, scheduler_running_requests=0)
                return result
            def stream(self, path, body, stop, record):
                record.update(body=body, deltas=[0, 1, 2, 3], terminal=True)
        args = SimpleNamespace(timeout=2, events=4, model='fixture', nvidia_smi='fixture')
        for route in ('conversation', 'stateless'):
            client = Client()
            with patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='GPU-fixture, 1, 2, 3')):
                case = runner.run_case(client, args, route, 3)
            self.assertTrue(case['passed'], case['failures'])
            self.assertEqual(client.created, 3 if route == 'conversation' else 0)
            if route == 'conversation':
                self.assertEqual(len({r['path'] for r in case['records']}), 3)
            for record in case['records']:
                self.assertNotIn('max_tokens', record['body'])
                self.assertNotIn('max_completion_tokens', record['body'])

    def test_cleanup_rejects_claim_leak_after_scheduler_drains(self):
        baseline, after = metrics(), metrics()
        for snapshot in (baseline, after):
            snapshot['engine'].update(scheduler_queue_depth=0, scheduler_running_requests=0)
        self.assertTrue(runner.cleanup_drained(after, baseline))
        after['engine']['kv_cache']['totals']['coordinator']['admission_claims'] = 1
        self.assertFalse(runner.cleanup_drained(after, baseline))

    def test_http_chunk_boundaries_do_not_split_sse_or_unicode(self):
        payload = 'data: {"event":"delta","delta":"é"}\n\ndata: [DONE]\n\n'.encode()
        chunks = [payload[index:index + 1] for index in range(len(payload))]
        wire = b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: text/event-stream\r\n\r\n'
        wire += b''.join(b'1\r\n' + chunk + b'\r\n' for chunk in chunks) + b'0\r\n\r\n'
        class Socket:
            def makefile(self, mode):
                return io.BufferedReader(io.BytesIO(wire))
            def shutdown(self, how):
                pass
        response = http.client.HTTPResponse(Socket())
        response.begin()
        class Connection:
            sock = Socket()
            def connect(self):
                pass
            def request(self, *args):
                pass
            def getresponse(self):
                return response
            def close(self):
                pass
        client = runner.Client('http://127.0.0.1:8080', 2)
        stop, record = threading.Event(), {}
        with patch.object(client, 'connection', return_value=Connection()):
            client.stream('/v1/chat/completions', {'stream': True}, stop, record)
        stop.set()
        self.assertTrue(record.get('terminal'), record)
        self.assertEqual(len(record['deltas']), 1)
        self.assertNotIn('error', record)

    def test_cancel_before_connection_never_posts_inference(self):
        class Connection:
            sock = None
            def connect(self):
                pass
            def request(self, *args):
                raise AssertionError('cancelled request must not be posted')
            def close(self):
                pass
        client = runner.Client('http://127.0.0.1:8080', 2)
        stop, record = threading.Event(), {}
        stop.set()
        with patch.object(client, 'connection', return_value=Connection()):
            client.stream('/v1/chat/completions', {'stream': True}, stop, record)
        self.assertTrue(record['cancelled'])
        self.assertNotIn('error', record)

    def test_uncapped_stream_and_disconnect(self):
        stop, record, bodies = threading.Event(), {}, []
        class Response:
            status = 200
            count = 0
            def readline(self):
                self.count += 1
                if self.count == 5:
                    stop.set()
                return b'data: {"event":"delta","delta":"x"}\n\n'
        class Connection:
            sock = None
            def connect(self):
                pass
            def request(self, method, path, body, headers):
                bodies.append(json.loads(body))
            def getresponse(self):
                return Response()
            def close(self):
                pass
        client = runner.Client('http://127.0.0.1:8080', 2)
        with patch.object(client, 'connection', return_value=Connection()):
            client.stream('/v1/chat/completions', {'stream': True}, stop, record)
        self.assertGreaterEqual(len(record['deltas']), 4)
        self.assertTrue(record['cancelled'])
        self.assertNotIn('max_tokens', bodies[0])
        self.assertNotIn('max_completion_tokens', bodies[0])


if __name__ == '__main__':
    unittest.main()
