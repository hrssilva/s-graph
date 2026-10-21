#!/usr/bin/env python3
"""Run identical scheduled joins over postorder and BFS FSTs, sequentially."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("binary", type=Path)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--cases", default="uniform-star,uniform-triangle,uniform-anchored,skew,mixed-triangle,parity-triangle-128,functional-triangle,functional-shared,large-chain,large-cycle")
p.add_argument("--trials", type=int, default=5)
p.add_argument("--target-ms", type=float, default=60)
p.add_argument("--capacity", type=int, default=65536)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
records = []
for pass_number in range(2):
    for case in a.cases.split(","):
        env = dict(os.environ, LAYOUT_TRIALS=str(a.trials), LAYOUT_TARGET_MS=str(a.target_ms),
                   LAYOUT_BLOCK_CAPACITY=str(a.capacity),
                   LAYOUT_POLICIES="adaptive,nested,levels" if pass_number == 0 else "adaptive")
        if pass_number:
            env["LAYOUT_BFS_FIRST"] = "1"
        run = subprocess.run([str(a.binary), case], env=env, capture_output=True, text=True, timeout=180)
        stem = f"{case}-{pass_number}"
        (a.output / f"{stem}.jsonl").write_text(run.stdout)
        (a.output / f"{stem}.log").write_text(run.stderr)
        run.check_returncode()
        rows = [json.loads(line) for line in run.stdout.splitlines()]
        for row in rows:
            row.update(pass_number=pass_number, block_capacity=a.capacity, trials=a.trials)
            records.append(row)
        print(case, pass_number, [(r["layout"], r["policy"], round(r["visit_us"], 3)) for r in rows], flush=True)
        (a.output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
summary = []
for case in a.cases.split(","):
    for policy in ["adaptive", "nested", "levels"]:
        pair = {}
        for layout in ["pow2", "bfs"]:
            selected = [r for r in records if (r["case"], r["policy"], r["layout"]) == (case, policy, layout)]
            value = dict(selected[0])
            for key in ["prepare_us", "visit_us", "total_us", "build_us"]:
                value[key] = statistics.median(r[key] for r in selected)
            value["pass_visit_us"] = [r["visit_us"] for r in selected]
            pair[layout] = value
        summary.append(dict(case=case, policy=policy, pow2=pair["pow2"], bfs=pair["bfs"],
                            bfs_visit_speedup=pair["pow2"]["visit_us"]/pair["bfs"]["visit_us"]))
(a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
