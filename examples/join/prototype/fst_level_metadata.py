#!/usr/bin/env python3
"""Compare compact sampled indexes and exact per-depth summaries."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import fst_index_scale as limits

ROOT = limits.ROOT


def summarize(rows):
    result = []
    keys = sorted({(r['backend'], r['shape'], r['mode']) for r in rows if 'backend' in r})
    for backend, shape, mode in keys:
        group = [r for r in rows if (r.get('backend'), r.get('shape'), r.get('mode')) == (backend, shape, mode)]

        def median(phase, key='us_per_edit', warm=False, cold=False):
            values = [r[key] for r in group if r['phase'] == phase
                      and (not warm or r.get('trial', 0) > 0)
                      and (not cold or r.get('trial') == 0)]
            return statistics.median(values) if values else None

        result.append(dict(backend=backend, shape=shape, mode=mode,
                           prepare_us=median('prepare', 'prepare_us'),
                           first_insert_us=median('insert', cold=True),
                           warm_insert_us=median('insert', warm=True),
                           fresh_insert_us=median('fresh_insert'),
                           warm_delete_us=median('delete_base', warm=True),
                           warm_restore_us=median('restore_base', warm=True),
                           base_index_bytes=median('insert', 'base_index_bytes', warm=True),
                           memo_bytes=median('insert', 'memo_bytes', warm=True),
                           total_ram_bytes=median('insert', 'total_ram_bytes', warm=True),
                           first_scans=median('insert', 'base_scans', cold=True),
                           first_summary_rejections=statistics.median([
                               r.get('summary_rejections', 0) for r in group
                               if r['phase'] == 'insert' and r.get('trial') == 0])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, default=ROOT / 'target/automata-tmp/fst-lazy-before-level-metadata')
    parser.add_argument('--after', type=Path, default=ROOT / 'target/automata-prototype/profiling/examples/fst_lazy_perf')
    parser.add_argument('--output', type=Path, default=ROOT / 'examples/join/prototype/results/fst-level-metadata')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--large-trials', type=int, default=1)
    parser.add_argument('--skip-large', action='store_true')
    parser.add_argument('--large-only', action='store_true')
    args = parser.parse_args()
    if args.trials < 1 or args.large_trials < 1:
        raise ValueError('trials must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    binaries = {'before': args.before.resolve(), 'after': args.after.resolve()}
    sources = ['src/fst_update_index.rs', 'src/mutable_fst.rs', 'src/path_fst.rs',
               'src/fst_snapshot.rs', 'examples/fst_lazy_perf.rs',
               'examples/join/prototype/fst_level_metadata.py']
    manifest = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
                    binary_sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in binaries.items()},
                    source_sha256={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources},
                    notes=['Before: saved pre-change binary; see binary hash and per-mode results.',
                           'After: current binary; see source hashes and per-mode results.',
                           'Wall times are sequential and unpinned; two reversed passes per invocation.',
                           'Index/RAM bytes count retained capacity, not RSS or peak preparation workspace.',
                           'Large cases use systemd verified RLIMIT_AS; CPUQuota=100% from the shared runner.'],
                    commands=[])
    if args.large_only and (args.output / 'manifest.json').exists():
        manifest['commands'] = json.loads((args.output / 'manifest.json').read_text())['commands']
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    raw = {}
    summary = json.loads((args.output / 'summary.json').read_text()) if args.large_only else {}
    fixtures = [] if args.large_only else [('small', 4096, 32, 'all', 1), ('medium', 65536, 64, 'heap', args.trials)]
    if not args.skip_large:
        fixtures.append(('large', 524288, 64, 'heap', args.large_trials))
    for size, vertices, edits, backend, trials in fixtures:
        for trial in range(trials):
            for label in (['before', 'after'] if trial % 2 == 0 else ['after', 'before']):
                stem = f'{size}-{label}-{trial}'
                command = [str(binaries[label]), str(vertices), str(edits), '3', backend]
                print(f'{stem}: start', flush=True)
                if size == 'large':
                    record = limits.capped_job(stem, command, args.output, fraction=0.5, max_gib=1)
                    if record['exit_code']:
                        raise RuntimeError(f'{stem}: benchmark failed')
                    rows = record['rows']
                else:
                    run = subprocess.run(command, capture_output=True, text=True, check=True, timeout=600)
                    (args.output / f'{stem}.jsonl').write_text(run.stdout)
                    (args.output / f'{stem}.log').write_text(run.stderr)
                    rows = [json.loads(line) for line in run.stdout.splitlines()]
                raw.setdefault(f'{size}_{label}', []).extend(rows)
                manifest['commands'].append(dict(stem=stem, command=command))
                (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
                summary.update({k: summarize(v) for k, v in raw.items()})
                (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
                print(f'{stem}: complete', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
