#!/usr/bin/env python3
"""Build a disposable crate containing the prototypes and unchanged baseline executors."""
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
shutil.copytree(root / "tests", snapshot / "tests")
shutil.copy2(root / "Cargo.lock", snapshot / "Cargo.lock")
manifest = re.sub(r"(?ms)^\[\[(?:bench|example)\]\]\n.*?(?=^\[|\Z)", "", (root / "Cargo.toml").read_text())
manifest += '\n[[example]]\nname = "composition_bench"\npath = "composition_bench.rs"\n'
manifest += '\n[[example]]\nname = "composition_profile"\npath = "composition_profile.rs"\n'
manifest += '\n[[example]]\nname = "join_profile"\npath = "join_profile.rs"\n'
manifest += '\n[[example]]\nname = "analytics_bench"\npath = "analytics_bench.rs"\n'
manifest += '\n[[example]]\nname = "large_bgp"\npath = "large_bgp.rs"\n'
manifest += '\n[[example]]\nname = "query_ops_bench"\npath = "query_ops_bench.rs"\n'
(snapshot / "Cargo.toml").write_text(manifest)
for name in ["compiled_filter", "lifted_join", "demand_join", "automata_core", "scheduled_join", "analytics", "large_bgp_cases", "query_ops", "query_ops_tests", "path_ops", "path_ops_tests"]:
    shutil.copy2(here / f"{name}.rs", snapshot / "src" / f"{name}.rs")
diagnostics = os.environ.get("PROTO_DIAGNOSTICS") == "1"
if diagnostics:
    subprocess.run(["git", "apply", str(here.parent / "lab/automaton.patch")], cwd=snapshot, check=True)
    path = snapshot / "src/automaton_join.rs"
    src = path.read_text()
    src = src.replace("    pub candidates: u64,", "    pub candidates: u64,\n    pub entries_by_atom: [u64; 16],\n    pub nodes_by_atom: [u64; 16],\n    pub leaves_by_atom: [u64; 16],")
    src = src.replace("        stats.atom_entries += 1;", "        if TRACK { stats.entries_by_atom[atom_idx] += 1; }\n        stats.atom_entries += 1;")
    src = src.replace("            for (d, term) in plan.terms.iter().enumerate() {", "            if TRACK { stats.leaves_by_atom[atom_idx] += 1; }\n            for (d, term) in plan.terms.iter().enumerate() {")
    src = src.replace("        stats.node_visits += nodes;", "        stats.node_visits += nodes;\n        if TRACK { stats.nodes_by_atom[atom_idx] += nodes; }")
    path.write_text(src)
with (snapshot / "src/lib.rs").open("a") as f:
    f.write('\n#[cfg(feature="heap")]\npub mod query_ops;\n#[cfg(feature="heap")]\npub mod path_ops;\n')
    f.write('\n#[cfg(feature="heap")]\npub mod analytics;\n')
    f.write('\n#[cfg(feature="heap")]\npub mod large_bgp_cases;\n')
    f.write(f"\npub const PROTOTYPE_DIAGNOSTICS: bool = {str(diagnostics).lower()};\n")
    f.write('\n#[cfg(feature="heap")]\npub mod compiled_filter;\n#[cfg(feature="heap")]\npub mod lifted_join;\n#[cfg(feature="heap")]\npub mod demand_join;\n#[cfg(feature="heap")]\npub mod automata_core;\n#[cfg(feature="heap")]\npub mod scheduled_join;\n')
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
shutil.copy2(here / "analytics_bench.rs", snapshot / "analytics_bench.rs")
shutil.copy2(here / "large_bgp.rs", snapshot / "large_bgp.rs")
shutil.copy2(here / "query_ops_bench.rs", snapshot / "query_ops_bench.rs")
# Expose the existing NFA metadata inside the disposable crate only; the
# shared-language adapter and the specialized runner use one compiled machine.
with (snapshot / "src/automaton.rs").open("a") as f:
    f.write('''
impl PathAutomaton {
    pub(crate) fn prototype_start(&self) -> u32 { self.start }
    pub(crate) fn prototype_states(&self) -> usize { self.accepting.len() }
    pub(crate) fn prototype_accepts(&self, state: u32) -> bool { self.accepting[state as usize] }
    pub(crate) fn prototype_transitions(&self, state: u32) -> &[Transition] {
        &self.transitions[self.offsets[state as usize]..self.offsets[state as usize + 1]]
    }
}
''')
print(snapshot)
