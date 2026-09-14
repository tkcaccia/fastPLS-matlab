#!/usr/bin/env python3
import filecmp
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
upstream = Path(sys.argv[1]).resolve() / "inst" / "include" / "fastpls"
vendored = root / "vendor" / "fastpls" / "include" / "fastpls"
comparison = filecmp.dircmp(upstream, vendored)
differences = comparison.left_only + comparison.right_only + comparison.diff_files
for child in comparison.common_dirs:
    nested = filecmp.dircmp(upstream / child, vendored / child)
    differences.extend(
        str(Path(child) / item)
        for item in nested.left_only + nested.right_only + nested.diff_files
    )
if differences:
    print("Core differences:", *differences, sep="\n")
    raise SystemExit(1)
metadata = json.loads((root / "UPSTREAM_CORE.json").read_text())
print(f"Core matches fastPLS {metadata['package_version']} at {metadata['commit']}")
