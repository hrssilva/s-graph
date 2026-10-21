#!/usr/bin/env python3
"""Force Levels against Adaptive on all comparison shapes, with paired order."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("binary", type=Path)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--trials", type=int, default=5)
p.add_argument("--target-ms", type=float, default=50)
p.add_argument("--cases", help="Comma-separated fixture names; defaults to the full matrix")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
cases = ["uniform-star", "uniform-triangle", "uniform-anchored", "skew", "dense-star", "mixed-triangle",
         "parity-triangle-128", "functional-triangle", "functional-shared", "large-chain-64", "large-chain-1024",
         "large-chain", "large-star-1024", "large-cycle", "large-cycle-4097"]
if a.cases:
    cases = a.cases.split(",")
records = []
for pass_number in range(2):
    for case in cases:
        env = dict(os.environ, LAYOUT_TRIALS=str(a.trials), LAYOUT_TARGET_MS=str(a.target_ms),
                   LAYOUT_POLICIES="adaptive,levels", LAYOUT_BLOCK_CAPACITY="65536")
        env.pop("LAYOUT_BFS_FIRST", None)
        if pass_number:
            env["LAYOUT_BFS_FIRST"] = "1"
        run = subprocess.run([str(a.binary), case], env=env, capture_output=True, text=True, timeout=90)
        stem = f"{case}-{pass_number}"
        (a.output / f"{stem}.jsonl").write_text(run.stdout)
        (a.output / f"{stem}.log").write_text(run.stderr)
        run.check_returncode()
        rows = [json.loads(line) for line in run.stdout.splitlines()]
        for row in rows:
            row.update(pass_number=pass_number, trials=a.trials)
            records.append(row)
        print(case, pass_number, [(r["layout"], r["policy"], round(r["visit_us"], 3)) for r in rows], flush=True)
        (a.output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
summary = []
for case in cases:
    for layout in ["bfs", "pow2"]:
        pair = {}
        for policy in ["adaptive", "levels"]:
            selected = [r for r in records if (r["case"], r["layout"], r["policy"]) == (case, layout, policy)]
            value = dict(selected[0])
            for key in ["prepare_us", "visit_us", "total_us"]:
                value[key] = statistics.median(r[key] for r in selected)
            value["pass_visit_us"] = [r["visit_us"] for r in selected]
            pair[policy] = value
        summary.append(dict(case=case, layout=layout, adaptive=pair["adaptive"], levels=pair["levels"],
                            levels_visit_speedup=pair["adaptive"]["visit_us"]/pair["levels"]["visit_us"],
                            levels_total_speedup=pair["adaptive"]["total_us"]/pair["levels"]["total_us"]))
(a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
