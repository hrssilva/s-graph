#!/usr/bin/env python3
"""Compare cyclic BGPs with prefix-friendly, carry, shuffled and dense data."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("binary", type=Path)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--budget", type=int, default=5_000_000)
p.add_argument("--trials", type=int, default=5)
p.add_argument("--target-ms", type=float, default=50)
p.add_argument("--policies", default="adaptive,levels")
p.add_argument("--cases", help="Comma-separated kind:vertices:atoms; dense uses vertices per group")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
cases = [(kind, 128 if kind == "odd-planted" else 4096, atoms)
         for kind in ["xor-ring", "carry-ring", "shuffled-ring", "odd-planted"]
         for atoms in ([9, 33, 129] if kind == "odd-planted" else [8, 32, 128])]
cases.append(("xor-padding", 4096, 128))
if a.cases:
    cases = [(kind, int(n), int(atoms)) for kind, n, atoms in (x.split(":") for x in a.cases.split(","))]
policies = a.policies.split(",")
records = []
for pass_number in range(2):
    for kind, vertices, atoms in cases:
        for layout in (["pow2", "bfs"] if pass_number == 0 else ["bfs", "pow2"]):
            for policy in (policies if pass_number == 0 else list(reversed(policies))):
                env = dict(os.environ, CYCLE_TRIALS=str(a.trials), CYCLE_TARGET_MS=str(a.target_ms))
                stem = f"{kind}-{vertices}-{atoms}-{layout}-{policy}-{pass_number}"
                command = [str(a.binary), kind, str(vertices), str(atoms), layout, policy, str(a.budget)]
                run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
                (a.output / f"{stem}.jsonl").write_text(run.stdout)
                (a.output / f"{stem}.log").write_text(run.stderr)
                run.check_returncode()
                row = json.loads(run.stdout)
                row.update(kind=kind, vertices=vertices, atoms=atoms, layout=layout, policy=policy,
                           pass_number=pass_number, budget=a.budget)
                records.append(row)
                (a.output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
                print(kind, vertices, atoms, layout, policy, pass_number,
                      round(row["visit_us"], 3) if row["finished"] else f"budget exhausted; {row['rows']} rows", flush=True)
summary = []
for kind, vertices, atoms in cases:
    for layout in ["bfs", "pow2"]:
        for policy in policies:
            selected = [r for r in records if (r["kind"], r["vertices"], r["atoms"], r["layout"], r["policy"]) ==
                        (kind, vertices, atoms, layout, policy)]
            row = dict(selected[0])
            for key in ["finished", "rows", "expected_rows", "nodes", "candidates", "planning_nodes", "cut", "order"]:
                assert all(r[key] == row[key] for r in selected), (kind, atoms, key)
            if row["finished"]:
                for key in ["prepare_us", "visit_us", "total_us"]:
                    row[key] = statistics.median(r[key] for r in selected)
            row["pass_visit_us"] = [r["visit_us"] for r in selected]
            summary.append(row)
for row in summary:
    other = next(r for r in summary if (r["kind"], r["vertices"], r["atoms"], r["policy"], r["layout"]) ==
                 (row["kind"], row["vertices"], row["atoms"], row["policy"], "pow2" if row["layout"] == "bfs" else "bfs"))
    for key in ["finished", "rows", "nodes", "candidates", "planning_nodes", "cut", "order"]:
        assert row[key] == other[key]
(a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
