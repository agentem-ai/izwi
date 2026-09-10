#!/usr/bin/env python3
"""Evaluate measured Fish load SLOs. This does not certify device/model correctness."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def evaluate(summary, *, min_completed, min_wall_seconds, ttfa_p99_ms, rtf_p99,
             max_failure_fraction):
    outcomes = summary['outcomes']
    completed = outcomes['completed']
    sent = summary['sent_requests']
    latency = summary['distributions']['client_receipt_ttfa_ms']['p99']
    rtf = summary['distributions']['client_request_to_last_pcm_rtf']['p99']
    failure_fraction = (outcomes['failed'] + outcomes['rejected']) / sent if sent else 1
    reasons = []
    if completed < min_completed:
        reasons.append('insufficient completed requests for configured percentile gate')
    if summary['wall_seconds'] < min_wall_seconds:
        reasons.append('measurement shorter than configured duration')
    if outcomes['generator_dropped']:
        reasons.append('load generator saturated; offered traffic was not delivered')
    if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency > ttfa_p99_ms:
        reasons.append('client receipt TTFA p99 missing or above SLO')
    if not isinstance(rtf, (int, float)) or not math.isfinite(rtf) or rtf > rtf_p99:
        reasons.append('request-to-last-PCM RTF p99 missing or above SLO')
    if failure_fraction > max_failure_fraction:
        reasons.append('failed/rejected fraction above SLO')
    return {'load_slo_passed': not reasons, 'reasons': reasons, 'failure_fraction': failure_fraction,
            'measured_completed_requests_per_second': summary['completed_requests_per_second'],
            'measured_audio_seconds_per_wall_second': summary['audio_seconds_per_wall_second']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--ttfa-p99-ms', required=True, type=float)
    parser.add_argument('--rtf-p99', required=True, type=float)
    parser.add_argument('--min-completed', type=int, default=1000)
    parser.add_argument('--min-wall-seconds', required=True, type=float)
    parser.add_argument('--max-failure-fraction', type=float, default=0)
    parser.add_argument('--operating-headroom', type=float, default=.2)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.ttfa_p99_ms, args.rtf_p99, args.min_wall_seconds)) or args.min_completed < 1:
        parser.error('latency, duration and sample gates must be finite and positive')
    if not 0 <= args.max_failure_fraction < 1 or not 0 <= args.operating_headroom < 1:
        parser.error('failure fraction and headroom must be in [0, 1)')
    identity = None
    rows = []
    for run in args.runs:
        metadata = json.loads((run / 'metadata.json').read_text())
        # Never merge GPUs, source SHAs, runtime/provider builds, cache states or workloads.
        profile = {key: value for key, value in metadata.items()
                   if key not in ('concurrency', 'requests', 'arrival_rate_per_second')}
        profile['workload_sha256'] = hashlib.sha256((run / 'request.json').read_bytes()).hexdigest()
        if identity is not None and identity != profile:
            parser.error('runs have different hardware/build/cache/workload identities; report each profile separately')
        identity = profile
        summary = json.loads((run / 'summary.json').read_text())
        row = evaluate(summary, min_completed=args.min_completed, min_wall_seconds=args.min_wall_seconds,
                       ttfa_p99_ms=args.ttfa_p99_ms, rtf_p99=args.rtf_p99,
                       max_failure_fraction=args.max_failure_fraction)
        row.update(run=str(run.resolve()), mode=summary['mode'],
                   concurrency_limit=summary['concurrency_limit'],
                   arrival_rate_per_second=summary['arrival_rate_per_second'])
        rows.append(row)
    passing = [row for row in rows if row['load_slo_passed'] and row['mode'] == 'open_loop']
    best = max(passing, key=lambda row: row['measured_completed_requests_per_second'], default=None)
    result = {'profile': identity, 'runs': rows, 'best_passing_open_loop_run': best,
              'operating_headroom': args.operating_headroom,
              'candidate_operating_requests_per_second':
                  best['measured_completed_requests_per_second'] * (1 - args.operating_headroom) if best else None,
              'production_certified': False,
              'remaining_release_evidence': ['native batch traces and device/provider correctness',
                  'mixed-workload fairness and overload recovery', 'stream quality and audio underruns',
                  'resource plateau and cancellation/fault ownership',
                  'actual proxy, durable jobs, replica drain/restart and shared persistence'],
              'gates': {key: getattr(args, key) for key in
                  ('ttfa_p99_ms', 'rtf_p99', 'min_completed', 'min_wall_seconds', 'max_failure_fraction')}}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if best is None:
        raise SystemExit('No open-loop run passed the configured load SLOs')


if __name__ == '__main__':
    main()
