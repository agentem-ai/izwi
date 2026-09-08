import base64
import importlib.util
import io
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('benchmark', Path(__file__).with_name('run-fish-streaming-benchmark.py'))
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def chunk(sequence=0):
    return {'event': 'chunk', 'sequence': sequence, 'audio_base64': base64.b64encode(b'\0' * 4096).decode(),
            'sample_count': 2048, 'sample_rate': 44100, 'audio_format': 'pcm_i16'}


class Tests(unittest.TestCase):
    def test_pcm_only_ttfa_and_real_terminal_counts(self):
        pcm = io.BytesIO()
        measure = benchmark.Measurement(pcm)
        measure.accept({'event': 'created'}, 10)
        measure.accept({'event': 'start'}, 20)
        measure.accept(chunk(), 100)
        measure.accept(chunk(1), 140)
        measure.accept({'event': 'final', 'tokens_generated': 2, 'audio_duration_secs': 4096 / 44100,
                        'timing': {'split_request_count': 1}}, 200)
        measure.accept({'event': 'done'}, 210)
        result = measure.result()
        self.assertEqual(result['client_receipt_ttfa_ms'], 100)
        self.assertEqual(result['client_request_to_last_pcm_ms'], 140)
        self.assertEqual(result['tokens_generated'], 2)
        self.assertEqual(len(pcm.getvalue()), 8192)
        self.assertEqual(result['interchunk_gap_ms']['p95'], 40)
        self.assertEqual(result['server_timing']['split_request_count'], 1)

    def test_sse_comments_crlf_and_multiline(self):
        self.assertEqual(list(benchmark.events([b': hello\r\n', b'data: {\r\n', b'data: "event": "start"}\r\n', b'\r\n'])), [{'event': 'start'}])

    def test_invalid_pcm_or_sequence_rejected(self):
        for field, value in [('sequence', 1), ('sample_count', 1), ('audio_base64', '?'), ('sample_rate', 0)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                benchmark.Measurement(io.BytesIO()).accept(dict(chunk(), **{field: value}), 1)

    def test_missing_terminal_is_failure(self):
        measure = benchmark.Measurement(io.BytesIO())
        measure.accept(chunk(), 1)
        with self.assertRaises(ValueError):
            measure.result()

    def test_error_after_partial_audio_is_failure(self):
        measure = benchmark.Measurement(io.BytesIO())
        measure.accept(chunk(), 1)
        with self.assertRaisesRegex(ValueError, 'cancelled'):
            measure.accept({'event': 'error', 'error': 'cancelled'}, 2)

    def test_duration_mismatch_is_failure(self):
        measure = benchmark.Measurement(io.BytesIO())
        measure.accept(chunk(), 1)
        measure.accept({'event': 'final', 'tokens_generated': 1, 'audio_duration_secs': 100}, 2)
        with self.assertRaises(ValueError):
            measure.result()



class LoadTests(unittest.TestCase):
    def test_closed_loop_runs_real_parallel_connections(self):
        import http.server
        import json
        import tempfile
        import threading
        import time

        state = {'active': 0, 'peak': 0, 'count': 0}
        lock = threading.Lock()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                with lock:
                    state['active'] += 1
                    state['count'] += 1
                    state['peak'] = max(state['peak'], state['active'])
                try:
                    time.sleep(.05)
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers()
                    for event in (chunk(), {'event': 'final', 'tokens_generated': 1,
                                             'audio_duration_secs': 2048 / 44100}):
                        self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
                finally:
                    with lock:
                        state['active'] -= 1

            def log_message(self, *_args):
                pass

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = benchmark.run_load(f'http://127.0.0.1:{server.server_port}', [{}],
                                            Path(tmp), 5, 9, 3)
                self.assertEqual(state['count'], 9)
                self.assertGreater(state['peak'], 1)
                self.assertLessEqual(state['peak'], 3)
                self.assertEqual(result['outcomes']['completed'], 9)
                self.assertTrue(result['passed'])
                self.assertEqual(len(list(Path(tmp).glob('request-*/audio.wav'))), 9)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_open_loop_does_not_hide_saturation_in_executor_queue(self):
        import tempfile
        import time
        from unittest.mock import patch

        def slow(*_args):
            time.sleep(.15)
            return {'passed': True, 'outcome': 'completed', 'audio_duration_secs': 1}

        with tempfile.TemporaryDirectory() as tmp, patch.object(benchmark, 'run_request', slow):
            result = benchmark.run_load('unused', [{}], Path(tmp), 1, 12, 2, 1000)
        self.assertEqual(result['outcomes']['completed'], 2)
        self.assertEqual(result['outcomes']['generator_dropped'], 10)
        self.assertEqual(result['outcomes']['rejected'], 0)
        self.assertEqual(result['sent_requests'], 2)
        self.assertFalse(result['passed'])

    def test_failed_and_rejected_requests_remain_in_denominator(self):
        result = benchmark.summarize([
            {'outcome': 'completed', 'audio_duration_secs': 10, 'client_receipt_ttfa_ms': 50},
            {'outcome': 'failed'}, {'outcome': 'rejected'}, {'outcome': 'generator_dropped'},
        ], 5, 2, 'open_loop')
        self.assertEqual(result['offered_requests'], 4)
        self.assertEqual(result['sent_requests'], 3)
        self.assertEqual(result['audio_seconds_per_wall_second'], 2)
        self.assertEqual(result['percentile_sample_count'], 1)
        self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
