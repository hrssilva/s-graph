#!/usr/bin/env python3
"""Isolated parallel construction trials with systemd memory containment."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--binary', type=Path, default=ROOT / 'target/automata-prototype/release/examples/fst_build_perf')
    p.add_argument('--out', type=Path, default=ROOT / 'examples/join/prototype/results/fst-parallel-build')
    p.add_argument('--triples', type=int, default=1_000_000)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--threads', type=int, nargs='+', default=[2, 4, 8])
    p.add_argument('--shapes', nargs='+', default=['functional', 'shared', 'irregular', 'skewed'])
    p.add_argument('--backends', nargs='+', default=['heap'])
    p.add_argument('--memory', default='1500M')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sources = ['Cargo.toml', 'src/path_fst.rs', 'src/fst_parallel_build.rs',
               'examples/fst_build_perf.rs', 'examples/join/prototype/fst_parallel_build.py']
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(binary=str(a.binary.resolve()), binary_sha256=digest(a.binary),
                    sources={f: digest(ROOT / f) for f in sources},
                    arguments=vars(a) | {'binary': str(a.binary), 'out': str(a.out)},
                    rustc=subprocess.check_output(['rustc', '-Vv'], text=True),
                    machine=subprocess.check_output(['uname', '-a'], text=True))
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows = []
    modes = [('sequential', 1)] + [('parallel', t) for t in a.threads]
    for trial in range(a.trials):
        for backend in a.backends:
            for shape in a.shapes:
                for mode, threads in modes if trial % 2 == 0 else reversed(modes):
                    name = f'{backend}-{shape}-{mode}-{threads}-{trial}'
                    unit = f'sgraph-build-{time.time_ns()}'
                    command = ['systemd-run', '--user', '--wait', '--pipe', '--collect', f'--unit={unit}',
                               '-p', f'MemoryMax={a.memory}', '-p', 'MemorySwapMax=0',
                               '/usr/bin/time', '-f', 'peak_rss_kib=%M', str(a.binary.resolve()),
                               mode, shape, str(a.triples), str(threads), backend]
                    proc = subprocess.run(command, text=True, capture_output=True)
                    (a.out / f'{name}.log').write_text(proc.stdout + proc.stderr)
                    if proc.returncode:
                        raise RuntimeError(f'{name}: {proc.stderr}')
                    result = next(line for line in proc.stdout.splitlines() if line.startswith('mode='))
                    row = dict(field.split('=', 1) for field in result.split(','))
                    for key in ['triples', 'threads', 'retained_bytes', 'nodes', 'visited']:
                        row[key] = int(row[key])
                    row['build_ms'] = float(row['build_ms'])
                    row['peak_rss_kib'] = int(next(line.split('=')[1] for line in proc.stderr.splitlines()
                                                   if line.startswith('peak_rss_kib=')))
                    row.update(trial=trial, backend=backend, command=command)
                    rows.append(row)
                    print(name, row['build_ms'], row['peak_rss_kib'], flush=True)
                    (a.out / 'trials.json').write_text(json.dumps(rows, indent=2) + '\n')
    summary = []
    for backend in a.backends:
        for shape in a.shapes:
            group = [r for r in rows if r['backend'] == backend and r['shape'] == shape]
            assert len({r['nodes'] for r in group}) == 1, 'global minimization mismatch'
            assert len({r['visited'] for r in group}) == 1, 'logical trie mismatch'
            for mode, threads in modes:
                trials = [r for r in group if r['mode'] == mode and r['threads'] == threads]
                summary.append(dict(backend=backend, shape=shape, mode=mode, threads=threads,
                                    **{k: statistics.median(r[k] for r in trials)
                                       for k in ['build_ms', 'peak_rss_kib', 'nodes', 'retained_bytes']}))
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
