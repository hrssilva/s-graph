#!/usr/bin/env python3
"""Paired old/new adaptive joins; both layouts, reversed version and layout order."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("baseline", type=Path)
p.add_argument("current", type=Path)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--cycles", action="store_true")
p.add_argument("--carry", action="store_true", help="Carry-chain work diagnostics (not query timings)")
p.add_argument("--trials", type=int, default=5)
p.add_argument("--target-ms", type=float, default=50)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
cases = ["uniform-star", "uniform-triangle", "uniform-anchored", "skew", "dense-star", "mixed-triangle",
         "parity-triangle-128", "functional-triangle", "functional-shared", "large-chain-64", "large-chain-1024",
         "large-chain", "large-star-1024", "large-cycle", "large-cycle-4097"]
if a.cycles:
    cases = ["xor-ring:4096:32", "xor-ring:4096:128", "carry-ring:4096:128", "shuffled-ring:4096:128",
             "odd-planted:128:33", "odd-planted:128:129", "odd-frontier:128:129", "xor-padding:4096:128"]
if a.carry:
    assert not a.cycles
    cases = ["20", "32", "4096"]
records = []
for pass_number in range(2):
    for case in cases:
        layouts = (["pow2", "bfs"] if pass_number == 0 else ["bfs", "pow2"]) if a.cycles or a.carry else [None]
        for layout in layouts:
            versions = [("baseline", a.baseline), ("current", a.current)]
            if pass_number:
                versions.reverse()
            for version, binary in versions:
                env = dict(os.environ, LAYOUT_TRIALS=str(a.trials), LAYOUT_TARGET_MS=str(a.target_ms),
                           LAYOUT_POLICIES="adaptive", CYCLE_TRIALS=str(a.trials), CYCLE_TARGET_MS=str(a.target_ms))
                env.pop("LAYOUT_BFS_FIRST", None)
                if pass_number:
                    env["LAYOUT_BFS_FIRST"] = "1"
                command = [str(binary), case]
                if a.cycles:
                    command = [str(binary), *case.split(":"), layout, "adaptive", "5000000"]
                if a.carry:
                    command = [str(binary), "budget", layout, "adaptive", case, "1000000"]
                run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
                stem = f"{case.replace(':', '-')}-{layout or 'both'}-{version}-{pass_number}"
                (a.output / f"{stem}.jsonl").write_text(run.stdout)
                (a.output / f"{stem}.log").write_text(run.stderr)
                run.check_returncode()
                for line in run.stdout.splitlines():
                    row = json.loads(line)
                    row.update(case=case, version=version, pass_number=pass_number)
                    if a.cycles or a.carry:
                        row["layout"] = layout
                    if a.carry:
                        assert row["finished"] and row["rows"] == 16
                        row["diagnostic_us"] = row["visit_us"]
                        row["visit_us"] = None
                        row["total_us"] = None
                    records.append(row)
                (a.output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
                print(case, layout or "both", version, pass_number,
                      [(r["layout"], "diagnostic" if a.carry else (round(r["visit_us"], 3) if r["visit_us"] is not None else "budget"))
                       for r in records if r["case"] == case and r["version"] == version and r["pass_number"] == pass_number], flush=True)
summary = []
for case in cases:
    for layout in ["bfs", "pow2"]:
        pair = {}
        for version in ["baseline", "current"]:
            selected = [r for r in records if (r["case"], r["layout"], r["version"]) == (case, layout, version)]
            assert len(selected) == 2
            row = dict(selected[0])
            for key in ["nodes", "candidates", "rows", "cut", "planning_nodes"]:
                assert all(r[key] == row[key] for r in selected), (case, version, key)
            for key in ["prepare_us", "visit_us", "total_us"]:
                if row[key] is not None:
                    row[key] = statistics.median(r[key] for r in selected)
            row["pass_visit_us"] = [r["visit_us"] for r in selected]
            pair[version] = row
        if not a.cycles:
            assert pair["baseline"]["rows"] == pair["current"]["rows"]
        summary.append(dict(case=case, layout=layout, **pair))
for row in summary:
    other = next(r for r in summary if r["case"] == row["case"] and r["layout"] != row["layout"])
    for version in ["baseline", "current"]:
        for key in ["nodes", "candidates", "rows", "cut", "planning_nodes"]:
            assert row[version][key] == other[version][key]
(a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
