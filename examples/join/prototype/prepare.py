#!/usr/bin/env python3
"""Benchmark production automata alongside retained experimental alternatives.

Promoted modules are copied from src; historical copies in this directory are
not injected. No repository source is patched by this script.
"""
from pathlib import Path
import re
import os
import subprocess
import shutil
import tempfile

here = Path(__file__).resolve().parent
root = here.parents[2]
snapshot = Path(tempfile.mkdtemp(prefix="sgraph-composition-"))
shutil.copytree(root / "src", snapshot / "src")
# The production module is test-only; this lab also exposes its fixtures.
p = snapshot / "src/lib.rs"
p.write_text(p.read_text().replace('#[cfg(all(test, feature = "heap"))]\nmod large_bgp_cases;', '#[cfg(feature = "heap")]\npub mod large_bgp_cases;'))
shutil.copytree(root / "tests", snapshot / "tests")
shutil.copy2(root / "Cargo.lock", snapshot / "Cargo.lock")
shutil.copy2(root / "README.md", snapshot / "README.md")
manifest = re.sub(r"(?ms)^\[\[(?:bench|example)\]\]\n.*?(?=^\[|\Z)", "", (root / "Cargo.toml").read_text())
manifest += '\n[[example]]\nname = "composition_bench"\npath = "composition_bench.rs"\n'
manifest += '\n[[example]]\nname = "composition_profile"\npath = "composition_profile.rs"\n'
manifest += '\n[[example]]\nname = "schedule_stress"\npath = "schedule_stress.rs"\n'
manifest += '\n[[example]]\nname = "cycle_bench"\npath = "cycle_bench.rs"\n'
manifest += '\n[[example]]\nname = "layout_bench"\npath = "layout_bench.rs"\n'
manifest += '\n[[example]]\nname = "join_profile"\npath = "join_profile.rs"\n'
manifest += '\n[[example]]\nname = "analytics_bench"\npath = "analytics_bench.rs"\n'
manifest += '\n[[example]]\nname = "large_bgp"\npath = "large_bgp.rs"\n'
manifest += '\n[[example]]\nname = "query_ops_bench"\npath = "query_ops_bench.rs"\n'
(snapshot / "Cargo.toml").write_text(manifest)
for name in ["compiled_filter", "lifted_join", "demand_join", "layout_source"]:
    shutil.copy2(here / f"{name}.rs", snapshot / "src" / f"{name}.rs")
diagnostics = os.environ.get("PROTO_DIAGNOSTICS") == "1"
if diagnostics:
    # Prevent repository discovery when TMPDIR puts the disposable crate in
    # target/: git apply must operate on this snapshot, never on its parent.
    subprocess.run(["git", "apply", str(here.parent / "lab/automaton.patch")],
                   cwd=snapshot, check=True,
                   env={**os.environ, "GIT_CEILING_DIRECTORIES": str(snapshot.parent)})
    path = snapshot / "src/automaton_join.rs"
    src = path.read_text()
    src = src.replace("    pub candidates: u64,", "    pub candidates: u64,\n    pub entries_by_atom: [u64; 16],\n    pub nodes_by_atom: [u64; 16],\n    pub leaves_by_atom: [u64; 16],")
    src = src.replace("        stats.atom_entries += 1;", "        if TRACK { stats.entries_by_atom[atom_idx] += 1; }\n        stats.atom_entries += 1;")
    src = src.replace("            for (d, term) in plan.terms.iter().enumerate() {", "            if TRACK { stats.leaves_by_atom[atom_idx] += 1; }\n            for (d, term) in plan.terms.iter().enumerate() {")
    src = src.replace("        stats.node_visits += nodes;", "        stats.node_visits += nodes;\n        if TRACK { stats.nodes_by_atom[atom_idx] += nodes; }")
    path.write_text(src)
with (snapshot / "src/lib.rs").open("a") as f:
    f.write('\n#[cfg(feature="heap")]\npub mod layout_source;\n')
    f.write(f"\npub const PROTOTYPE_DIAGNOSTICS: bool = {str(diagnostics).lower()};\n")
    f.write('\n#[cfg(feature="heap")]\npub mod compiled_filter;\n#[cfg(feature="heap")]\npub mod lifted_join;\n#[cfg(feature="heap")]\npub mod demand_join;\n')
# Only add equality/hash derivations for compile-time query-state interning.
p = snapshot / "src/level_automaton.rs"
s = p.read_text().replace('#[derive(Clone, Copy, PartialEq, Eq)]', '#[derive(Clone, Copy, PartialEq, Eq, Hash)]')
s = s.replace('#[derive(Clone, Copy)]\npub(crate) struct RangeState', '#[derive(Clone, Copy, PartialEq, Eq, Hash)]\npub(crate) struct RangeState')
s = s.replace('#[derive(Clone)]\npub(crate) enum DimState', '#[derive(Clone, PartialEq, Eq, Hash)]\npub(crate) enum DimState')
s += '''
impl RangeTable {
    pub(crate) fn prototype_bytes(&self) -> usize {
        (self.lower_boundary_mask.capacity() + self.upper_boundary_mask.capacity()) * 8
        + (self.a_sub.capacity() + self.b_sub.capacity() + self.dim_bit_masks.capacity()) * 4
        + self.shifts.capacity()
    }
}
impl DimRestrictionAutomaton {
    pub(crate) fn prototype_bytes(&self) -> usize {
        match &self.kind { DimKind::Free => 0, DimKind::Range(t) => t.prototype_bytes(),
            DimKind::Multi(ts) => ts.capacity() * std::mem::size_of::<RangeTable>() + ts.iter().map(RangeTable::prototype_bytes).sum::<usize>() }
    }
}
'''
p.write_text(s)
p=snapshot / "src/automaton_join.rs"
with p.open("a") as f:
    f.write("\nimpl AutomatonJoin<'_> { pub fn prototype_allocated_bytes(&self) -> usize { self.atoms.capacity() * std::mem::size_of::<AtomPlan>() } }\n")
with p.open("a") as f:
    # Diagnostics execute separately from timing and sampling.
    fields = ('stats.node_visits, stats.candidates, stats.entries_by_atom, stats.nodes_by_atom, stats.leaves_by_atom'
              if diagnostics else '0u64, 0u64, [0u64; 16], [0u64; 16], [0u64; 16]')
    f.write('''
impl AutomatonJoin<'_> {
    pub fn prototype_profile_stats(&self, bindings: &mut Vec<u32>) -> String {
        let mut stats = AutomatonJoinStats::default();
        let _: ControlFlow<()> = self.visit_with_stats(bindings, &mut stats, |_| ControlFlow::Continue(()));
        let (nodes, candidates, entries, per_atom_nodes, leaves) = (''' + fields + ''');
        format!(r#"{{"nodes":{},"candidates":{},"rows":{},"atom_entries":{},"entries_by_atom":{:?},"nodes_by_atom":{:?},"leaves_by_atom":{:?}}}"#, nodes, candidates, stats.results, stats.atom_entries, entries, per_atom_nodes, leaves)
    }
}
''')
shutil.copy2(root / "examples/common/rdf_gen.rs", snapshot / "rdf_gen.rs")
shutil.copy2(here / "bench.rs", snapshot / "composition_bench.rs")
shutil.copy2(here / "profile.rs", snapshot / "composition_profile.rs")
shutil.copy2(here / "join_profile.rs", snapshot / "join_profile.rs")
shutil.copy2(here / "cycle_bench.rs", snapshot / "cycle_bench.rs")
shutil.copy2(here / "analytics_bench.rs", snapshot / "analytics_bench.rs")
shutil.copy2(here / "large_bgp.rs", snapshot / "large_bgp.rs")
shutil.copy2(here / "query_ops_bench.rs", snapshot / "query_ops_bench.rs")
# The postorder adapter has exactly the BFS adapter's access operations, with
# subtraction for backward deltas. Only this disposable source copy is extended.
with (snapshot / "src/path_fst.rs").open("a") as f:
    f.write('''
impl PathFstDeltaPow2 {
    pub(crate) fn layout_root_addr(&self) -> u64 { self.root_virtual_addr }
    pub(crate) fn layout_domains(&self) -> [u64; 3] { self.dimension_sizes }
    pub(crate) fn layout_schedule(&self, level: usize) -> ([u8; 3], [u8; 3], [u8; 3]) {
        (self.level_dim_bits[level], self.level_shifts_cache[level], self.level_child_cell_bits_cache[level])
    }
    pub(crate) fn layout_read_word(&self, addr: u64) -> u64 {
        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        self.storage.read_u64(block, offset)
    }
    pub(crate) fn layout_read_delta(&self, addr: u64, real_index: usize) -> u64 {
        let block = (addr >> self.capacity_bits) as usize;
        let offset = (addr & self.capacity_mask) as usize;
        self.storage.read_var(block, offset + WORD_BYTES + real_index * DELTA_BYTES as usize, DELTA_BYTES)
    }
}
''')
# A diagnostic node budget for otherwise exponential schedule stress cases.
# It invokes the existing budgeted executor without changing schedule logic.
with (snapshot / "src/scheduled_join.rs").open("a") as f:
    f.write('''
impl<S: TripleSource> ScheduledJoin<S> {
    pub fn prototype_visit_budget(&self, ws: &mut ScheduleWorkspace<S::Cursor>, stats: &mut ScheduleStats,
        limit: u64, mut emit: impl FnMut(&[u32])) -> bool {
        *stats = ScheduleStats::default();
        matches!(self.run::<true, true, false, ()>(ws, stats, limit, &mut |row| {
            emit(row); ControlFlow::Continue(())
        }), ControlFlow::Continue(()))
    }
}
''')
shutil.copy2(here / "schedule_stress.rs", snapshot / "schedule_stress.rs")
shutil.copy2(here / "layout_bench.rs", snapshot / "layout_bench.rs")
print(snapshot)
