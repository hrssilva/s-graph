#!/usr/bin/env python3
"""Profile validated, warmed joins; FIFO control excludes setup and validation."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("binary", type=Path)
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/shape-perf")
parser.add_argument("--engines", help="Comma-separated engines (default: automaton,demand)")
parser.add_argument("--cases", help="Comma-separated cases, including mixed-triangle")
parser.add_argument("--target-seconds", type=float, help="Calibrate repetitions using an unprofiled warm visit")
parser.add_argument("--mode", choices=["perf", "stats", "rare-first"], default="perf")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
events = ["cycles:u", "instructions:u", "branches:u", "branch-misses:u", "cache-references:u", "cache-misses:u"]
cases = [
    ("uniform-star", 100, 70),
    ("uniform-triangle", 130, 80),
    ("uniform-anchored", 900_000, 26_000),
    ("skew", 20, 200),
    ("parity-triangle-128", 8, 5_000_000),
]
if args.cases:
    requested=args.cases.split(",")
    presets={c:(c,a,d) for c,a,d in cases}
    presets["mixed-triangle"]=("mixed-triangle",100,50000)
    cases=[presets[c] for c in requested]
if args.mode == "rare-first":
    cases = [("skew", 20, 200)]
engines = ["automaton", "demand"] if args.mode != "rare-first" else ["automaton", "automaton-rare-first"]
if args.engines:
    engines=args.engines.split(",")
summary = []
work = []
for case, auto_repeats, demand_repeats in cases:
    for engine in engines:
        stem = f"{case}-{engine}" + ("-control" if args.mode == "rare-first" else "")
        env = dict(os.environ, LC_ALL="C")
        if engine == "automaton-rare-first":
            env["JOIN_INPUT_ORDER"] = "1"
        repeats = 400 if engine == "automaton-rare-first" else demand_repeats if engine == "demand" else auto_repeats
        if args.mode == "stats":
            run = subprocess.run([str(args.binary),engine,case,"stats","1"],env=env,capture_output=True,text=True,check=True)
            value = json.loads(run.stdout)
            value.update(case=case,engine=engine)
            work.append(value)
            print(case,engine,value,flush=True)
            continue
        if args.target_seconds is not None:
            pilot=subprocess.run([str(args.binary),engine,case,"visit","10"],env=env,capture_output=True,text=True,check=True)
            per_visit=json.loads(pilot.stdout)["visit_us"]
            repeats=max(1,min(50_000_000,int(args.target_seconds*1e6/max(per_visit,0.001))))
        records = []
        for trial in range(3):
            with tempfile.TemporaryDirectory(prefix="sgraph-paired-stat-") as folder:
                folder = Path(folder)
                control, ack = folder/"control", folder/"ack"
                os.mkfifo(control)
                os.mkfifo(ack)
                env.update(PROTO_PERF_CONTROL=str(control),PROTO_PERF_ACK=str(ack))
                stat_path = args.output / f"{stem}-stat-{trial}.csv"
                cmd = ["perf","stat","-x",";","--delay=-1","--control",f"fifo:{control},{ack}","-e",",".join(events),"-o",str(stat_path),"--",str(args.binary),engine,case,"visit",str(repeats)]
                run = subprocess.run(cmd,env=env,capture_output=True,text=True,check=True,timeout=60)
                (args.output/f"{stem}-stat-{trial}.json").write_text(run.stdout)
                (args.output/f"{stem}-stat-{trial}.log").write_text(run.stderr)
                record = json.loads(run.stdout)
                for parts in csv.reader(stat_path.open(),delimiter=";"):
                    if len(parts) < 3 or parts[2] not in events:
                        continue
                    key=parts[2].removesuffix(":u").replace("-","_")
                    try:
                        record[key] = float(parts[0])/repeats
                        record[key+"_running_pct"] = float(parts[4])
                    except ValueError:
                        record[key] = None
                records.append(record)
        result = {"case":case,"engine":engine,"repeats":repeats,"rows":records[0]["rows"]}
        for key in records[0]:
            if key not in result and all(isinstance(r.get(key),(int,float)) for r in records):
                result[key]=statistics.median(r[key] for r in records)
        summary.append(result)
        with tempfile.TemporaryDirectory(prefix="sgraph-paired-record-") as folder:
            folder=Path(folder)
            control,ack=folder/"control",folder/"ack"
            os.mkfifo(control)
            os.mkfifo(ack)
            env.update(PROTO_PERF_CONTROL=str(control),PROTO_PERF_ACK=str(ack))
            cmd=["perf","record","--no-buildid-cache","--delay=-1","--control",f"fifo:{control},{ack}","-e","cycles:u","-F","999","--call-graph","dwarf,4096","-o",str(folder/"perf.data"),"--",str(args.binary),engine,case,"visit",str(repeats)]
            run=subprocess.run(cmd,env=env,capture_output=True,text=True,check=True,timeout=60)
            (args.output/f"{stem}-record.json").write_text(run.stdout)
            (args.output/f"{stem}-record.log").write_text(run.stderr)
            report=subprocess.run(["perf","report","--stdio","--no-children","-g","none","--percent-limit","0.5","-i",str(folder/"perf.data")],capture_output=True,text=True,check=True)
            (args.output/f"{stem}-self.txt").write_text(report.stdout)
        print(case,engine,result,flush=True)
        # Save progress so an interrupted run retains completed pairs.
        (args.output/f"{args.mode}-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
if work:
    (args.output/"work.json").write_text(json.dumps(work,indent=2)+"\n")
