#!/usr/bin/env python3
"""Copy current sources to /tmp and apply diagnostics; never edit the live library."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

here = Path(__file__).resolve().parent
root = here.parents[2]
snapshot = Path(tempfile.mkdtemp(prefix="sgraph-join-lab-"))
shutil.copytree(root / "src", snapshot / "src")
shutil.copy2(root / "Cargo.lock", snapshot / "Cargo.lock")
manifest = (root / "Cargo.toml").read_text()
manifest = re.sub(r"(?ms)^\[\[(?:bench|example)\]\]\n.*?(?=^\[|\Z)", "", manifest)
manifest += '\n[[example]]\nname = "investigate"\npath = "investigate.rs"\n'
(snapshot / "Cargo.toml").write_text(manifest)
shutil.copy2(here / "investigate.rs", snapshot / "investigate.rs")
subprocess.run(
    ["git", "apply", str(here / "automaton.patch")],
    cwd=snapshot,
    check=True,
    stdout=subprocess.DEVNULL,
)
print(snapshot)
