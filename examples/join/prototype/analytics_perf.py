#!/usr/bin/env python3
"""Profile warmed analytics runs; FIFO gating excludes graph build and warmup."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('binary', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--backends', default='fst,csr')
p.add_argument('--algorithms', default='pagerank,label-propagation')
p.add_argument('--seconds', type=float, default=1.5)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
summary = []
for backend in a.backends.split(','):
    for algorithm in a.algorithms.split(','):
        env = dict(os.environ, LC_ALL='C', ANALYTICS_CASE='uniform')
        command = [str(a.binary), '--profile', backend, algorithm]
        pilot = subprocess.run(command + ['1'], env=env, capture_output=True, text=True, check=True)
        seconds = json.loads(pilot.stdout)['elapsed_s']
        repeats = max(1, int(a.seconds / seconds))
        stem = f'{backend}-{algorithm}'
        records = []
        for trial in range(3):
            with tempfile.TemporaryDirectory(prefix='sgraph-analytics-perf-') as folder:
                folder = Path(folder)
                control, ack = folder/'control', folder/'ack'
                os.mkfifo(control)
                os.mkfifo(ack)
                env.update(PROTO_PERF_CONTROL=str(control), PROTO_PERF_ACK=str(ack))
                events = ['cycles:u', 'instructions:u', 'branches:u', 'branch-misses:u']
                stat = a.output/f'{stem}-{trial}.csv'
                run = subprocess.run(['perf', 'stat', '-x', ';', '--delay=-1', '--control',
                    f'fifo:{control},{ack}', '-e', ','.join(events), '-o', str(stat), '--'] +
                    command + [str(repeats)], env=env, capture_output=True, text=True, check=True, timeout=60)
                (a.output/f'{stem}-{trial}.json').write_text(run.stdout)
                (a.output/f'{stem}-{trial}.log').write_text(run.stderr)
                record = json.loads(run.stdout)
                for parts in csv.reader(stat.open(), delimiter=';'):
                    if len(parts) >= 3 and parts[2] in events:
                        record[parts[2]] = float(parts[0]) / repeats
                        record[parts[2]+'_running_pct'] = float(parts[4])
                records.append(record)
        result = dict(backend=backend, algorithm=algorithm, repeats=repeats)
        for key in records[0]:
            result[key] = statistics.median(r[key] for r in records)
        result['run_ms'] = result['elapsed_s'] * 1000 / repeats
        summary.append(result)
        with tempfile.TemporaryDirectory(prefix='sgraph-analytics-record-') as folder:
            folder = Path(folder)
            control, ack = folder/'control', folder/'ack'
            os.mkfifo(control)
            os.mkfifo(ack)
            env.update(PROTO_PERF_CONTROL=str(control), PROTO_PERF_ACK=str(ack))
            run = subprocess.run(['perf', 'record', '--no-buildid-cache', '--delay=-1', '--control',
                f'fifo:{control},{ack}', '-e', 'cycles:u', '-F', '999', '--call-graph', 'dwarf,4096',
                '-o', str(folder/'perf.data'), '--'] + command + [str(repeats)],
                env=env, capture_output=True, text=True, check=True, timeout=60)
            (a.output/f'{stem}-record.json').write_text(run.stdout)
            (a.output/f'{stem}-record.log').write_text(run.stderr)
            report = subprocess.run(['perf', 'report', '--stdio', '--no-children', '-g', 'none',
                '--percent-limit', '0.5', '-i', str(folder/'perf.data')], capture_output=True, text=True, check=True)
            (a.output/f'{stem}-self.txt').write_text(report.stdout)
        (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(result, flush=True)
