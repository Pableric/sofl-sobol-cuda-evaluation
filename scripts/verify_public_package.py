#!/usr/bin/env python3
"""Check the exact release library and refuse unlisted public files."""

import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
library = root / "lib/libsofl_sobol_sm75.so"
audit = json.loads((root / "evidence/binary_audit.json").read_text())
digest = hashlib.sha256(library.read_bytes()).hexdigest()
if not audit["qualified"] or digest != audit["library_sha256"]:
    raise SystemExit("release binary/audit mismatch")
if audit["embedded_ptx_entries"] or audit["debug_sections"]:
    raise SystemExit("release binary contains PTX or debug sections")

allowed = {
    ".gitignore", "README.md", "EVALUATION_TERMS.md", "SHA256SUMS",
    "docs/COLAB.md",
    "include/sofl_sobol.h", "lib/libsofl_sobol_sm75.so",
    "bench/compare.cu", "vendor/mathdx.json",
    "evidence/binary_audit.json", "evidence/validation.log",
    "evidence/BUILD_INFO.json",
    "scripts/analyze.py", "scripts/run_t4.sh",
    "scripts/verify_public_package.py", "scripts/write_result_hashes.py",
}
found = {
    str(path.relative_to(root))
    for path in root.rglob("*")
    if path.is_file()
    and ".git" not in path.relative_to(root).parts
    and "__pycache__" not in path.relative_to(root).parts
    and path.suffix != ".pyc"
    and path.relative_to(root).parts[0] not in {"results", "build"}
}
if found != allowed:
    raise SystemExit(
        f"unexpected package files: {sorted(found - allowed)}; "
        f"missing: {sorted(allowed - found)}"
    )
subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=root, check=True)
print(f"PUBLIC_PACKAGE=PASS library_sha256={digest}")
