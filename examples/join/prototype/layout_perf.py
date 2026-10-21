#!/usr/bin/env python3
"""Gated perf counters for layout comparisons; excludes build and validation."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("binary", type=Path)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--cases", default="uniform-star:adaptive,skew:adaptive,functional-triangle:adaptive,functional-triangle:levels")
p.add_argument("--seconds", type=float, default=0.8)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
events = ["cycles:u", "instructions:u", "cache-references:u", "cache-misses:u"]
records = []
for case in a.cases.split(","):
    repeats = {}
    for layout in ["pow2", "bfs"]:
        pilot = subprocess.run([str(a.binary), "--profile", layout, case, "10"], capture_output=True, text=True, check=True)
        repeats[layout] = max(1, min(50_000_000, int(a.seconds * 1e6 / max(json.loads(pilot.stdout)["visit_us"], 0.005))))
    for trial in range(3):
        for layout in (["pow2", "bfs"] if trial % 2 == 0 else ["bfs", "pow2"]):
            with tempfile.TemporaryDirectory(prefix="sgraph-layout-perf-") as folder:
                folder = Path(folder)
                control, ack = folder / "control", folder / "ack"
                os.mkfifo(control); os.mkfifo(ack)
                env = dict(os.environ, LC_ALL="C", PROTO_PERF_CONTROL=str(control), PROTO_PERF_ACK=str(ack))
                stem = f"{case}-{layout}-{trial}"
                stat = a.output / f"{stem}.csv"
                command = ["perf", "stat", "-x", ";", "--delay=-1", "--control", f"fifo:{control},{ack}",
                           "-e", ",".join(events), "-o", str(stat), "--", str(a.binary), "--profile", layout, case, str(repeats[layout])]
                run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
                (a.output / f"{stem}.json").write_text(run.stdout)
                (a.output / f"{stem}.log").write_text(run.stderr)
                run.check_returncode()
                row = json.loads(run.stdout)
                row.update(case=case, layout=layout, trial=trial)
                for parts in csv.reader(stat.open(), delimiter=";"):
                    if len(parts) >= 5 and parts[2] in events:
                        row[parts[2]] = float(parts[0]) / repeats[layout]
                        row[parts[2] + "_running_pct"] = float(parts[4])
                assert all(e in row for e in events)
                records.append(row)
                print(case, layout, trial, round(row["visit_us"], 3), flush=True)
                (a.output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
summary = []
for case in a.cases.split(","):
    for layout in ["pow2", "bfs"]:
        selected = [r for r in records if (r["case"], r["layout"]) == (case, layout)]
        row = dict(case=case, layout=layout)
        for key in ["visit_us"] + events + [e + "_running_pct" for e in events]:
            row[key] = statistics.median(r[key] for r in selected)
        summary.append(row)
(a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
