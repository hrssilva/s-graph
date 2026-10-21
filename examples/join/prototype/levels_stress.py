#!/usr/bin/env python3
"""Bounded carry-chain comparison: diagnostic work counts and full traversal."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("binary", type=Path)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--timeout", type=float, default=5)
p.add_argument("--budget", type=int, default=1_000_000)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
records = []
for atoms in [4, 8, 12, 16, 20, 32, 4096]:
    for mode in ["budget", "full"]:
        for trial in range(1 if mode == "budget" else 3):
            for layout in (["pow2", "bfs"] if trial % 2 == 0 else ["bfs", "pow2"]):
                for policy in (["adaptive", "levels"] if trial % 2 == 0 else ["levels", "adaptive"]):
                    if any(r.get("timed_out") and (r["atoms"], r["mode"], r["layout"], r["policy"]) ==
                           (atoms, mode, layout, policy) for r in records):
                        continue
                    stem = f"{atoms}-{mode}-{layout}-{policy}-{trial}"
                    command = [str(a.binary), mode, layout, policy, str(atoms), str(a.budget)]
                    try:
                        run = subprocess.run(command, capture_output=True, text=True, timeout=a.timeout)
                        (a.output / f"{stem}.jsonl").write_text(run.stdout)
                        (a.output / f"{stem}.log").write_text(run.stderr)
                        run.check_returncode()
                        row = json.loads(run.stdout)
                        row["timed_out"] = False
                    except subprocess.TimeoutExpired:
                        row = dict(atoms=atoms, mode=mode, layout=layout, policy=policy,
                                   timed_out=True, process_timeout_s=a.timeout)
                    row["trial"] = trial
                    records.append(row)
                    (a.output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
            print(atoms, mode, trial, flush=True)
summary = []
for atoms in [4, 8, 12, 16, 20, 32, 4096]:
    for mode in ["budget", "full"]:
        for layout in ["bfs", "pow2"]:
            for policy in ["adaptive", "levels"]:
                selected = [r for r in records if (r["atoms"], r["mode"], r["layout"], r["policy"]) ==
                            (atoms, mode, layout, policy)]
                row = dict(selected[0])
                if not row["timed_out"]:
                    for key in ["prepare_us", "visit_us"]:
                        row[key] = statistics.median(r[key] for r in selected)
                    for key in ["nodes", "candidates", "rows", "finished", "cut", "planning_nodes"]:
                        assert all(r[key] == row[key] for r in selected)
                row["trials"] = len(selected)
                summary.append(row)
(a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
