import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('capacity', Path(__file__).with_name('report-fish-serving-capacity.py'))
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


class Tests(unittest.TestCase):
    def evaluate(self, **updates):
        summary = {'outcomes': {'completed': 1000, 'failed': 0, 'rejected': 0, 'generator_dropped': 0},
                   'sent_requests': 1000, 'wall_seconds': 3600,
                   'distributions': {'client_receipt_ttfa_ms': {'p99': 100},
                                     'client_request_to_last_pcm_rtf': {'p99': .5}},
                   'completed_requests_per_second': .27, 'audio_seconds_per_wall_second': 2}
        summary.update(updates)
        return capacity.evaluate(summary, min_completed=1000, min_wall_seconds=3600,
                                 ttfa_p99_ms=500, rtf_p99=1, max_failure_fraction=0)

    def test_passes_exact_boundaries(self):
        self.assertTrue(self.evaluate()['load_slo_passed'])

    def test_generator_saturation_is_not_capacity_evidence(self):
        result = self.evaluate(outcomes={'completed': 1000, 'failed': 0, 'rejected': 0, 'generator_dropped': 2})
        self.assertFalse(result['load_slo_passed'])
        self.assertIn('load generator saturated', result['reasons'][0])

    def test_missing_latency_is_not_zero(self):
        result = self.evaluate(distributions={'client_receipt_ttfa_ms': {'p99': None},
                                             'client_request_to_last_pcm_rtf': {'p99': float('nan')}})
        self.assertEqual(len(result['reasons']), 2)

    def test_rejections_count_against_success(self):
        self.assertFalse(self.evaluate(outcomes={'completed': 1000, 'failed': 0, 'rejected': 1,
                                                'generator_dropped': 0})['load_slo_passed'])

    def test_short_run_cannot_qualify(self):
        self.assertFalse(self.evaluate(wall_seconds=1)['load_slo_passed'])


if __name__ == '__main__':
    unittest.main()
