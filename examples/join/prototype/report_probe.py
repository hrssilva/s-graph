#!/usr/bin/env python3
"""Reproduce report diagnostics in an isolated crate; never patch production.

Builds existing carry/star/cycle controls and a ten-atom chain with endpoint
constraints. A query-only connected-order candidate reuses the production
compiler and Nested executor. Every measurement runs in its own child process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONNECTED = r'''
// Diagnostic constructor only: same compiler, source, guards and executor.
impl<S: TripleSource> ScheduledJoin<S> {
    pub fn prototype_connected(query: &JoinQuery<Term>, source: S) -> Result<Self> {
        let mut join = Self::with_source(query, source, Policy::Nested)?;
        let atoms: Vec<_> = query.patterns.iter().map(|p| [p.subject, p.predicate, p.object]).collect();
        assert_eq!(atoms.len(), join.plan.order.len(), "diagnostic requires unique atoms");
        let mut order: Vec<_> = (0..atoms.len()).collect();
        let mut bound = vec![false; query.num_vars];
        for i in 0..order.len() {
            let j = (i..order.len()).max_by_key(|&j| {
                let shared = atoms[order[j]].iter().filter(|t| matches!(t, Term::Var(v) if bound[*v])).count();
                (shared > 0, atoms[order[j]].iter().filter(|t| matches!(t, Term::Const(_))).count(), shared, std::cmp::Reverse(j))
            }).unwrap();
            order.swap(i, j);
            for t in atoms[order[i]] { if let Term::Var(v) = t { bound[v] = true; } }
        }
        let mut zero = vec![0; query.num_vars];
        let domains = join.source.domains();
        for a in &atoms {
            for (d, t) in a.iter().enumerate() {
                if let Term::Var(v) = *t {
                    let width = 64 - (domains[d] - 1).leading_zeros();
                    zero[v] |= !((1u64 << width) - 1) as u32;
                }
            }
        }
        join.plan = compile(&join.source, &atoms, &order, &zero, 0);
        join.planning.selected_order = order;
        Ok(join)
    }
}
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-binaries", type=Path, help="optional previously built example directory for paired comparison")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        parser.error("choose a fresh output directory to preserve earlier measurements")
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src").glob("*.rs"))}
    source_hashes.update({str(p.relative_to(ROOT)): sha(p) for p in [ROOT / "Cargo.toml", ROOT / "Cargo.lock", HERE / "prepare.py", Path(__file__).resolve(), HERE / "report_probe.rs"]})
    snapshot = Path(subprocess.check_output(["python3", str(HERE / "prepare.py")], cwd=ROOT, text=True).strip())
    with (snapshot / "src/scheduled_join.rs").open("a") as f:
        constructor = CONNECTED
        if "fn compile_checked" in (snapshot / "src/scheduled_join.rs").read_text():
            constructor = constructor.replace("compile(&join.source, &atoms, &order, &zero, 0)", "compile_checked(&join.source, &atoms, &order, &zero, 0, &Uncontrolled).unwrap()")
        f.write(constructor)
    shutil.copy2(HERE / "report_probe.rs", snapshot / "report_probe.rs")
    with (snapshot / "Cargo.toml").open("a") as f:
        f.write('\n[[example]]\nname = "report_probe"\npath = "report_probe.rs"\n')
    cargo = ["cargo", "--offline", "--locked", "--profile", "profiling", "--manifest-path", str(snapshot / "Cargo.toml"), "--target-dir", str(out / "build")]
    # Cargo subcommands precede common options.
    def build_command(subcommand, *extra):
        return [cargo[0], subcommand, *cargo[1:], *extra]
    build = build_command("build", "--example", "report_probe", "--example", "schedule_stress", "--example", "layout_bench", "--example", "cycle_bench")
    tests = build_command("test", "--example", "report_probe", "--example", "cycle_bench")
    with (out / "build.log").open("w") as log:
        subprocess.run(build, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out / "tests.log").open("w") as log:
        subprocess.run(tests, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    bindir = out / "build/profiling/examples"
    names = ["report_probe", "schedule_stress", "layout_bench", "cycle_bench"]
    baseline = args.baseline_binaries.resolve() if args.baseline_binaries else None
    manifest = dict(
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dirty=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        rustc=subprocess.check_output(["rustc", "--version"], text=True).strip(),
        platform=platform.platform(),
        cpu=next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")), "unknown"),
        profile="profiling", features="default (heap)", snapshot=str(snapshot),
        limits=dict(address_space_bytes=2 << 30, cpu_seconds=30, process_timeout_seconds=40),
        build=build, tests=tests, sources=source_hashes,
        snapshot_scheduled_join_sha256=sha(snapshot / "src/scheduled_join.rs"),
        binaries={name: sha(bindir / name) for name in names},
        note="Synthetic diagnostics, not TEP3. Connected constructor and node-budget wrapper exist only in snapshot. Timed visits have no counters or limits; diagnostic visits are bounded."
    )
    if baseline:
        manifest["baseline_binaries"] = {name: sha(baseline / name) for name in names}
        old_manifest = baseline.parent.parent.parent / "manifest.json"
        if old_manifest.exists():
            manifest["baseline_manifest"] = json.loads(old_manifest.read_text())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []

    def run_one(name, arguments, family, trial, version):
        command = [str((baseline if version == "baseline" else bindir) / name), *map(str, arguments)]
        env = dict(os.environ, LAYOUT_TRIALS="5", LAYOUT_TARGET_MS="10", LAYOUT_POLICIES="adaptive,nested,levels", CYCLE_TRIALS="5", CYCLE_TARGET_MS="10")
        if trial % 2:
            env["LAYOUT_BFS_FIRST"] = "1"
        stem = f"{len(records):03d}-{family}"
        record = dict(family=family, trial=trial, command=command, version=version)
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=40, preexec_fn=limits)
            (out / f"{stem}.jsonl").write_text(result.stdout)
            (out / f"{stem}.log").write_text(result.stderr)
            result.check_returncode()
            record["results"] = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
            assert record["results"]
            if family == "cycle":
                for row in record["results"]:
                    row.update(case=arguments[0], n=arguments[1], atoms=arguments[2], layout=arguments[3], policy=arguments[4])
        except subprocess.TimeoutExpired:
            record["timed_out"] = True
        records.append(record)
        (out / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"{version}/{family}: {' '.join(map(str, arguments))}", flush=True)

    def run(name, arguments, family, trial):
        versions = ["current"]
        if baseline and "controlled" not in arguments:
            versions.append("baseline")
        if trial % 2:
            versions.reverse()
        for version in versions:
            run_one(name, arguments, family, trial, version)

    for trial in range(2):
        policies = ["adaptive", "controlled", "nested", "levels", "frontier", "connected"]
        if trial % 2:
            policies.reverse()
        for n in [256, 1024]:
            for ids in ["ordered", "shuffled"]:
                for policy in policies:
                    run("report_probe", [n, ids, policy, 1_000_000], "endpoints", trial)
        for atoms in [4, 8, 12, 16, 20, 32]:
            for policy in ["adaptive", "levels"]:
                run("schedule_stress", ["budget", "bfs", policy, atoms, 1_000_000], "carry", trial)
        run("layout_bench", ["uniform-star"], "star", trial)
        run("layout_bench", ["functional-triangle"], "functional", trial)
        for case in ["large-chain-4096", "large-star-1024", "large-cycle-4097"]:
            run("layout_bench", [case], "large", trial)
        for kind in ["odd-planted", "xor-ring", "xor-padding"]:
            for policy in ["adaptive", "levels", "frontier"]:
                run("cycle_bench", [kind, 128 if kind == "odd-planted" else 4096, 33 if kind == "odd-planted" else 32, "bfs", policy, 1_000_000], "cycle", trial)
    groups = {}
    for record in records:
        for r in record.get("results", []):
            key = (record["version"], record["family"], r.get("case"), r.get("n"), r.get("ids"), r.get("atoms"), r.get("layout"), r.get("policy"))
            groups.setdefault(key, []).append(r)
    summary = []
    for key, rows in groups.items():
        r = dict(rows[0], version=key[0], family=key[1], passes=len(rows))
        for field in ["nodes", "candidates", "rows", "finished", "cut", "order", "frontier", "planning_nodes"]:
            if field in r:
                assert all(x[field] == r[field] for x in rows), (key, field)
        for field in r:
            if field.endswith("_us") and all(isinstance(x.get(field), (int, float)) for x in rows):
                r[field] = statistics.median(x[field] for x in rows)
        summary.append(r)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert all(sha(ROOT / path) == value for path, value in source_hashes.items()), "source changed during measurement"


if __name__ == "__main__":
    main()
