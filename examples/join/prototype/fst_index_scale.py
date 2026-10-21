#!/usr/bin/env python3
"""Run exact-count FST scaling in sequential, verified systemd memory limits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import resource
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
GIB = 1024 ** 3


def available():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0]) * 1024


def own_cgroup():
    path = next(line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    return Path("/sys/fs/cgroup") / path.lstrip("/")


def inside(args):
    cgroup = own_cgroup()
    # User cgroups may not have the memory controller delegated. systemd's
    # RLIMIT_AS remains an enforced per-process bound in that configuration.
    # exec replaces this verifier with the single-process Rust benchmark.
    as_limits = resource.getrlimit(resource.RLIMIT_AS)
    if as_limits != (args.cap, args.cap):
        raise RuntimeError(f"Address-space limit not enforced: {as_limits}")
    effective = {name: (cgroup / name).read_text().strip() if (cgroup / name).exists() else None
                 for name in ["memory.max", "memory.high", "memory.swap.max"]}
    print(json.dumps(dict(phase="limits_verified", pid=os.getpid(), cgroup=str(cgroup),
                          enforced="RLIMIT_AS", address_space_limits=as_limits,
                          cgroup_controls=effective)), flush=True)
    os.environ["SGRAPH_EXPECTED_AS_BYTES"] = str(args.cap)
    os.execvpe(args.command[0], args.command, os.environ)



def show(unit):
    run = subprocess.run(["systemctl", "--user", "show", unit,
                          "-p", "Result", "-p", "ExecMainCode", "-p", "ExecMainStatus",
                          "-p", "MemoryPeak", "-p", "MemorySwapPeak", "-p", "MemoryMax",
                          "-p", "MemoryHigh", "-p", "MemorySwapMax", "-p", "OOMPolicy",
                          "-p", "ControlGroup", "-p", "LimitAS", "-p", "LimitASSoft"], capture_output=True, text=True)
    return dict(line.split("=", 1) for line in run.stdout.splitlines() if "=" in line)


def capped_job(name, command, out, fraction=0.5, max_gib=2):
    avail = available()
    cap = int(min(max_gib * GIB, avail * fraction, avail - 512 * 1024**2))
    cap = cap // (16 * 1024**2) * (16 * 1024**2)
    if cap < 256 * 1024**2:
        raise RuntimeError("Less than 512 MiB available: refusing another run")
    unit = f"sgraph-index-{name}-{os.getpid()}"
    launch = ["systemd-run", "--user", "--wait", "--pipe", f"--unit={unit}",
              "--property=MemoryAccounting=yes", f"--property=MemoryMax={cap}",
              f"--property=MemoryHigh={cap * 4 // 5}", f"--property=LimitAS={cap}", "--property=MemorySwapMax=0",
              "--property=OOMPolicy=kill", "--property=LimitCORE=0", "--property=RuntimeMaxSec=600",
              "--property=TasksMax=32", "--property=Nice=10", "--property=CPUQuota=100%",
              f"--working-directory={ROOT}",
              f"--setenv=TMPDIR={ROOT / 'target/automata-tmp'}",
              f"--setenv=CARGO_TARGET_DIR={ROOT / 'target/automata-prototype'}",
              "/usr/bin/python3", str(Path(__file__).resolve()), "--inside-unit", "--cap", str(cap), "--", *command]
    print(f"{name}: available={avail / GIB:.2f} GiB, hard cap={cap / GIB:.2f} GiB", flush=True)
    start = time.monotonic()
    run = subprocess.run(launch, capture_output=True, text=True, timeout=660)
    (out / f"{name}.jsonl").write_text(run.stdout)
    (out / f"{name}.log").write_text(run.stderr)
    rows = []
    for line in run.stdout.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    properties = show(unit)
    record = dict(name=name, available_bytes=avail, address_space_cap_bytes=cap,
                  enforced_limit="RLIMIT_AS", requested_memory_max_bytes=cap,
                  elapsed_s=time.monotonic() - start, command=launch, exit_code=run.returncode,
                  systemd=properties, rows=rows)
    (out / f"{name}-unit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{name}: exit={run.returncode}, result={properties.get('Result')}, peak={properties.get('MemoryPeak')}", flush=True)
    # Stop the sequence if the limit check itself failed. OOM/timeout is a
    # benchmark outcome, provided execution reached verified containment.
    if not any(row.get("phase") == "limits_verified" for row in rows):
        print(run.stderr, file=sys.stderr)
        raise RuntimeError("No verified memory containment; no benchmarks launched")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside-unit", action="store_true")
    parser.add_argument("--cap", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/join/prototype/results/fst-index-scale")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1_000_000, 10_000_000, 100_000_000])
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--max-gib", type=float, default=2)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if args.inside_unit:
        return inside(args)
    args.output.mkdir(parents=True, exist_ok=True)
    (ROOT / "target/automata-tmp").mkdir(parents=True, exist_ok=True)
    records = []
    if not 0 < args.fraction <= 0.8:
        raise ValueError("fraction must leave at least 20% of available RAM")
    if not args.skip_build:
        build = capped_job("build", ["/usr/bin/cargo", "rustc", "--offline", "--profile", "profiling",
                                   "--jobs", "1", "--example", "fst_index_scale", "--", "-C", "link-arg=-Wl,--threads=1"],
                           args.output, args.fraction, args.max_gib)
        records.append(build)
        if build["exit_code"]:
            raise RuntimeError("Capped build failed; see build logs")
    binary = ROOT / "target/automata-prototype/profiling/examples/fst_index_scale"
    for triples in args.sizes:
        for shape in ["functional", "mixed"]:
            record = capped_job(f"{triples}-{shape}", [str(binary), str(triples), shape], args.output, args.fraction, args.max_gib)
            records.append(record)
            (args.output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
    manifest = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [ROOT / "examples/fst_index_scale.rs", Path(__file__).resolve(),
                                             ROOT / "src/path_fst.rs", ROOT / "src/mutable_fst.rs"]})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
