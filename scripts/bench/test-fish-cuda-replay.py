#!/usr/bin/env python3
"""Offline Fish replay fixtures; no model, service or GPU is accessed."""
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock
import wave

spec = importlib.util.spec_from_file_location('fish_replay', pathlib.Path(__file__).with_name('run-fish-cuda-replay.py'))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def wav():
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b'\x01\x00' * 8000)
    return buffer.getvalue()


class FakeClient:
    def __init__(self, status='completed', cancel=False, audio=None):
        self.status, self.cancel, self.audio = status, cancel, wav() if audio is None else audio
        self.calls = []

    def request(self, path, destination, body=None):
        self.calls.append((path, body))
        if path.endswith('/audio'):
            raw = self.audio
        elif path.endswith('/cancel'):
            raw = json.dumps({'cancelled': self.cancel}).encode()
        elif body is not None:
            raw = b'{"id":"fixture", "processing_status":"pending"}'
        else:
            raw = json.dumps({'id': 'fixture', 'processing_status': self.status,
                              'processing_error': 'Cancelled by speech history request' if self.cancel else 'reservation failure',
                              'audio_duration_secs': 1.0}).encode()
        destination.write_bytes(raw)
        return raw


class ReplayTests(unittest.TestCase):
    def test_audio_validates_decoded_frames_and_duration(self):
        self.assertEqual(replay.validate_audio(wav(), 1.0)['frames'], 8000)
        for raw, duration in [(b'', 1), (wav()[:-2], 1), (wav(), 0), (wav(), float('nan')),
                              (wav(), float('inf')), (wav(), True), (wav(), 5)]:
            with self.assertRaises((ValueError, EOFError, wave.Error)):
                replay.validate_audio(raw, duration)

    def run_fixture(self, client, cancel=False, timeout=1):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / 'case'
            result = replay.run_case(client, {'model_id': replay.MODEL}, output, timeout, .001, cancel)
            self.assertEqual(json.loads((output / 'result.json').read_text()), result)
            return result

    def test_completion_requires_success_and_audio(self):
        client = FakeClient()
        self.assertTrue(self.run_fixture(client)['passed'])
        self.assertEqual([path for path, _ in client.calls],
                         ['/v1/text-to-speech', '/v1/text-to-speech/fixture', '/v1/text-to-speech/fixture/audio'])
        self.assertFalse(self.run_fixture(FakeClient(status='failed'))['passed'])
        self.assertFalse(self.run_fixture(FakeClient(audio=b''))['passed'])

    def test_cancellation_uses_actual_api_failed_record_contract(self):
        self.assertTrue(self.run_fixture(FakeClient(status='failed', cancel=True), cancel=True)['passed'])
        self.assertFalse(self.run_fixture(FakeClient(cancel=False), cancel=True)['passed'])
        self.assertFalse(self.run_fixture(FakeClient(status='completed', cancel=True), cancel=True)['passed'])

    def test_poll_timeout_is_failure(self):
        result = self.run_fixture(FakeClient(status='pending'), timeout=.005)
        self.assertFalse(result['passed'])
        self.assertIn('deadline', result['error'])

    def test_full_matrix_fetches_equivalent_reference_and_keeps_hardware_gate(self):
        class MatrixClient(FakeClient):
            def request(self, path, destination, body=None):
                if path.startswith('/v1/voices/'):
                    raw = wav() if path.endswith('/audio') else b'{"reference_text":"matching transcript"}'
                    destination.write_bytes(raw)
                    return raw
                return super().request(path, destination, body)
        metadata = {'deployed_sha': 'a' * 40, 'gpu': 'fixture', 'backend': 'cuda', 'dtype': 'f32',
                    'effective_context': 4096, 'service_concurrency': 2, 'runtime_versions': 'fixture'}
        client = MatrixClient()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / 'metadata.json').write_text(json.dumps(metadata))
            argv = ['replay', '--server', 'http://fixture', '--metadata', str(root / 'metadata.json'),
                    '--output', str(root / 'evidence'), '--sequential', '2', '--concurrency', '2']
            with mock.patch('sys.argv', argv), mock.patch.object(replay, 'Client', return_value=client):
                self.assertEqual(replay.main(), 0)
            summary = json.loads((root / 'evidence/summary.json').read_text())
            self.assertTrue(summary['automatic_checks_passed'])
            self.assertFalse(summary['hardware_qualified'])
            self.assertEqual(len(summary['cases']), 6)
            direct = [body for _, body in client.calls if body and 'reference_audio' in body]
            self.assertEqual(len(direct), 1)
            self.assertEqual(direct[0]['reference_text'], 'matching transcript')
            self.assertEqual(replay.base64.b64decode(direct[0]['reference_audio']), wav())
            self.assertNotIn('saved_voice_id', direct[0])

    def test_metadata_does_not_accept_local_sha_or_missing_cuda_context(self):
        metadata = {'deployed_sha': 'a' * 40, 'gpu': 'test', 'backend': 'cuda', 'dtype': 'f32',
                    'effective_context': 4096, 'service_concurrency': 2, 'runtime_versions': 'fixture'}
        replay.validate_metadata(metadata)
        for key, bad in [('deployed_sha', 'abc'), ('backend', 'cpu'), ('effective_context', -1),
                         ('service_concurrency', True), ('runtime_versions', '')]:
            with self.assertRaises(ValueError):
                replay.validate_metadata(dict(metadata, **{key: bad}))


if __name__ == '__main__':
    unittest.main()
