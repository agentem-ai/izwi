import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
import io

spec = importlib.util.spec_from_file_location('long_form', Path(__file__).with_name('run-fish-long-form-qualification.py'))
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


class QualificationTests(unittest.TestCase):
    def test_matrix_covers_both_languages_all_durations_and_interleaves_short_jobs(self):
        rows = qualification.workloads({'model': 'FishAudio-S2-Pro', 'saved_voice_id': 'voice', 'max_tokens': 128})
        self.assertEqual(len(rows), 20)
        for language in ['en', 'zh']:
            long = [row for row in rows if row['language'] == language and row['kind'] == 'long']
            self.assertEqual([row['estimated_minutes'] for row in long], [1, 10, 30, 60, 120])
            self.assertEqual(sorted(len(row['request']['text']) for row in long),
                             [len(row['request']['text']) for row in long])
        for index, row in enumerate(rows):
            self.assertEqual(row['kind'], 'long' if index % 2 == 0 else 'short')
            self.assertTrue(row['request']['stream'])
            self.assertEqual(row['request']['saved_voice_id'], 'voice')
            self.assertNotIn('max_tokens', row['request'])

    def test_rejects_bad_duration(self):
        for value in [0, -1, 1.5, True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                qualification.workloads({}, [value])

    def test_compares_same_job_in_bounded_blocks_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = b'\x00\x01' * 100000
            pcm = root / 'audio.pcm'
            pcm.write_bytes(raw)
            wav = root / 'saved.wav'
            with wave.open(str(wav), 'wb') as sink:
                sink.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
                sink.writeframes(raw)
            result = qualification.compare_pcm(wav, pcm, 44100)
            self.assertTrue(result['stream_matches_saved'])
            self.assertEqual(result['samples'], 100000)
            with self.assertRaisesRegex(ValueError, 'format'):
                qualification.compare_pcm(wav, pcm, 24000)
            for invalid in [raw[:-2], raw + b'\0\0', b'\0\0' + raw[2:]]:
                pcm.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    qualification.compare_pcm(wav, pcm, 44100)

    def test_record_identity_must_not_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log = root / 'events.jsonl'
            log.write_text(json.dumps({'event': {'event': 'created', 'record': {'id': 'one'}}}) + '\n')
            self.assertEqual(qualification.record_id(root), 'one')
            with log.open('a') as stream:
                stream.write(json.dumps({'event': {'record': {'id': 'two'}}}) + '\n')
            with self.assertRaisesRegex(ValueError, 'identity'):
                qualification.record_id(root)

    def test_ready_is_required_and_failed_limited_output_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'events.jsonl').write_text(json.dumps({'event': {'record': {'id': 'one'}}}) + '\n')
            for status in ['failed', 'cancelled', 'running']:
                response = io.BytesIO(json.dumps({'processing_status': status,
                    'processing_error': 'frame_limit'}).encode())
                with patch.object(qualification.urllib.request, 'urlopen', return_value=response):
                    with self.assertRaisesRegex(ValueError, 'ready'):
                        qualification.verify_saved('http://fixture/v1/text-to-speech', root, {}, 1, 100)

    def test_short_latency_summary_keeps_failures_visible(self):
        metrics = qualification.class_metrics([
            {'kind': 'short', 'passed': True, 'client_receipt_ttfa_ms': 100},
            {'kind': 'short', 'passed': False},
            {'kind': 'long', 'passed': True, 'client_receipt_ttfa_ms': 500},
        ])
        self.assertEqual(metrics['short']['ttfa_p99_ms'], 100)
        self.assertEqual(metrics['short']['ttfa_sample_count'], 1)
        self.assertEqual(metrics['short']['failed'], 1)
        self.assertEqual(metrics['long']['ttfa_p99_ms'], 500)

    def test_partial_stream_is_not_a_success(self):
        measurement = qualification.benchmark.Measurement(io.BytesIO())
        with self.assertRaisesRegex(ValueError, 'frame_limit'):
            measurement.accept({'event': 'error', 'error': 'generation incomplete: frame_limit'}, 10)
        with self.assertRaises(ValueError):
            measurement.result()


if __name__ == '__main__':
    unittest.main()
