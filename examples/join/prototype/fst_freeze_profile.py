#!/usr/bin/env python3
"""Profile consuming freeze on packed inputs, with FIFO gates and verified limits."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import fst_index_scale as limits

ROOT = limits.ROOT
TMP = ROOT / "target/automata-tmp"
CORE = ["cycles:u", "instructions:u", "branches:u", "branch-misses:u"]
CACHE = ["cycles:u", "instructions:u", "cache-references:u", "cache-misses:u"]


def capped(name, command, out):
    avail = limits.available()
    cap = int(min(2 * 1024**3, avail * 0.5)) // (16 * 1024**2) * (16 * 1024**2)
    if cap < 512 * 1024**2:
        raise RuntimeError("Insufficient available memory for profiling")
    unit = f"sgraph-freeze-{name}-{os.getpid()}"
    launch = ["systemd-run", "--user", "--wait", "--pipe", f"--unit={unit}",
              "--property=MemoryAccounting=yes", f"--property=MemoryMax={cap}",
              f"--property=MemoryHigh={cap*4//5}", "--property=MemorySwapMax=0",
              f"--property=LimitAS={cap}", "--property=LimitCORE=0",
              "--property=OOMPolicy=kill", "--property=RuntimeMaxSec=600",
              "--property=TasksMax=32", f"--working-directory={ROOT}",
              f"--setenv=TMPDIR={TMP}", "/usr/bin/python3", str(Path(limits.__file__).resolve()),
              "--inside-unit", "--cap", str(cap), "--", *command]
    start = time.monotonic()
    print(f"{name}: start; available={avail/1024**3:.2f} GiB, AS cap={cap/1024**3:.2f} GiB", flush=True)
    run = subprocess.run(launch, capture_output=True, text=True, timeout=660)
    (out/f"{name}.jsonl").write_text(run.stdout)
    (out/f"{name}.log").write_text(run.stderr)
    rows = []
    for line in run.stdout.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    record = dict(name=name, command=launch, exit_code=run.returncode,
                  elapsed_s=time.monotonic()-start, available_bytes=avail,
                  address_space_cap_bytes=cap, systemd=limits.show(unit))
    (out/f"{name}-unit.json").write_text(json.dumps(record, indent=2)+"\n")
    if not any(r.get("phase") == "limits_verified" for r in rows):
        raise RuntimeError(f"No verified containment: {name}: {run.stderr}")
    if run.returncode:
        raise RuntimeError(f"Failed {name}: {run.stderr[-2500:]}")
    freezes = [r for r in rows if r.get("phase") == "freeze"]
    if not freezes:
        raise RuntimeError(f"No freeze result: {name}")
    print(f"{name}: complete in {record['elapsed_s']:.2f}s; freeze={sum(r['elapsed_s'] for r in freezes):.3f}s", flush=True)
    return freezes


def counters(path, events):
    result = {}
    for parts in csv.reader(path.open(), delimiter=";"):
        if len(parts) >= 5 and parts[2] in events:
            result[parts[2]] = float(parts[0])
            result[parts[2]+"_running_pct"] = float(parts[4])
    if not all(e in result for e in events):
        raise RuntimeError(f"Missing counters: {path}")
    return result


def profile(binary, case, out, trials, stat_only=False):
    triples, backend, layout, mode = case
    stem = f"{triples}-{backend}-{layout}-{mode}"
    command = [str(binary), str(triples), backend, layout, mode]
    samples = []
    for trial in range(trials if stat_only else trials + 1):
        events = CORE if trial < trials else CACHE
        name = f"{stem}-stat-{trial}" if trial < trials else f"{stem}-cache"
        with tempfile.TemporaryDirectory(prefix="freeze-perf-", dir=TMP) as folder:
            control, ack = Path(folder)/"control", Path(folder)/"ack"
            os.mkfifo(control); os.mkfifo(ack)
            env = ["/usr/bin/env", "LC_ALL=C", "DEBUGINFOD_URLS=",
                   f"PROTO_PERF_CONTROL={control}", f"PROTO_PERF_ACK={ack}"]
            stat = out/f"{name}.csv"
            rows = capped(name, env + ["/usr/bin/perf", "stat", "-x", ";", "--delay=-1",
                          "--control", f"fifo:{control},{ack}", "-e", ",".join(events),
                          "-o", str(stat), "--", *command, "1"], out)
            row = rows[0] | counters(stat, events)
            if trial < trials:
                samples.append(row)
            else:
                cache = row
    summary = {key: statistics.median(r[key] for r in samples)
               for key in samples[0] if isinstance(samples[0][key], (int, float))}
    summary.update(triples=triples, backend=backend, layout=layout, mode=mode,
                   samples_ms=[r["elapsed_s"]*1000 for r in samples],
                   ipc=summary["instructions:u"]/summary["cycles:u"],
                   branch_miss_pct=100*summary["branch-misses:u"]/summary["branches:u"])
    if stat_only:
        return summary
    summary.update(cache_miss_pct=100*cache["cache-misses:u"]/cache["cache-references:u"],
                   cache_counters=cache)
    repeats = max(1, min(4, math.ceil(1.0/summary["elapsed_s"])))
    name = f"{stem}-record"
    with tempfile.TemporaryDirectory(prefix="freeze-record-", dir=TMP) as folder:
        control, ack = Path(folder)/"control", Path(folder)/"ack"
        os.mkfifo(control); os.mkfifo(ack)
        env = ["/usr/bin/env", "LC_ALL=C", "DEBUGINFOD_URLS=",
               f"PROTO_PERF_CONTROL={control}", f"PROTO_PERF_ACK={ack}"]
        data = out/f"{stem}.data"
        capped(name, env + ["/usr/bin/perf", "record", "--no-buildid-cache", "--delay=-1",
               "--control", f"fifo:{control},{ack}", "-e", "cycles:u", "-F", "1999",
               "--call-graph", "dwarf,4096", "-m", "256", "-o", str(data),
               "--", *command, str(repeats)], out)
        report_env = dict(os.environ, LC_ALL="C", DEBUGINFOD_URLS="")
        for suffix, options in [("self", ["--no-children", "-g", "none"]),
                                ("callgraph", ["--children", "-g", "graph,0.5,caller"])]:
            report = subprocess.run(["perf", "report", "--stdio", *options,
                       "--percent-limit", "0.5", "-i", str(data)], env=report_env,
                       capture_output=True, text=True, check=True, timeout=120)
            (out/f"{stem}-{suffix}.txt").write_text(report.stdout)
            (out/f"{stem}-{suffix}.log").write_text(report.stderr)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=ROOT/"target/automata-prototype/profiling/examples/fst_freeze_profile")
    p.add_argument("--output", type=Path, default=ROOT/"examples/join/prototype/results/fst-freeze-profile")
    p.add_argument("--sizes", type=int, nargs="+", default=[1_000_000, 4_000_000])
    p.add_argument("--backends", nargs="+", default=["heap", "file", "compressed"])
    p.add_argument("--trials", type=int, default=3)
    # Retain compatibility with recorded commands; every case is now packed.
    p.add_argument("--packed-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--default-control", action="store_true")
    p.add_argument("--stat-only", action="store_true", help="Only core-counter timing trials; skip cache and sampling passes")
    p.add_argument("--methods", nargs="+", choices=["blocks", "depths", "streaming", "freeze"], default=["streaming"])
    a = p.parse_args()
    assert all(n >= 1_000_000 and n % 2 == 0 for n in a.sizes) and a.trials > 0
    a.output.mkdir(parents=True, exist_ok=True); TMP.mkdir(exist_ok=True)
    cases = [(n, b, "packed", m) for n in a.sizes for b in a.backends for m in a.methods]
    if a.default_control:
        cases.append((min(a.sizes), "heap", "packed", "freeze"))
    sources = ["examples/fst_freeze_profile.rs", "examples/join/prototype/fst_freeze_profile.py",
               "src/fst_block_compact.rs", "src/fst_depth_compact.rs", "src/fst_stream_compact.rs", "src/mutable_fst.rs", "src/path_fst.rs",
               "src/storage.rs", "src/storage_file.rs", "src/storage_compressed.rs"]
    manifest = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
                    binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
                    source_sha256={s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                    cases=cases, trials=a.trials, stat_only=a.stat_only,
                    perf_version=subprocess.check_output(["perf", "version"], text=True).strip())
    (a.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    summary = []
    for case in cases:
        summary.append(profile(a.binary, case, a.output, a.trials, a.stat_only))
        (a.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
