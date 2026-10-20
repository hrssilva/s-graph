#!/usr/bin/env python3
"""Build a disposable crate containing the prototypes and unchanged baseline executors."""
from pathlib import Path
import re
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
(snapshot / "Cargo.toml").write_text(manifest)
for name in ["compiled_filter", "lifted_join", "demand_join"]:
    shutil.copy2(here / f"{name}.rs", snapshot / "src" / f"{name}.rs")
with (snapshot / "src/lib.rs").open("a") as f:
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
shutil.copy2(root / "examples/common/rdf_gen.rs", snapshot / "rdf_gen.rs")
shutil.copy2(here / "bench.rs", snapshot / "composition_bench.rs")
shutil.copy2(here / "profile.rs", snapshot / "composition_profile.rs")
print(snapshot)
