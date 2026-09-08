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


if __name__ == '__main__':
    unittest.main()
