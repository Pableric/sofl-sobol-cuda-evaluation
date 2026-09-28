#!/usr/bin/env python3
"""Hash every finished result file without including the manifest itself."""

import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
files = sorted(
    path for path in root.rglob("*")
    if path.is_file() and path.name != "SHA256SUMS"
)
lines = [
    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}"
    for path in files
]
(root / "SHA256SUMS").write_text("\n".join(lines) + "\n")
print(f"HASHED_RESULT_FILES={len(files)}")
