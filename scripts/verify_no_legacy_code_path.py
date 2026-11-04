#!/usr/bin/env python3
"""Reject legacy server roots only where they can affect project execution."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


RUNTIME_SURFACES = ("pipeline/", "scripts/", "configs/", "runtime/", "src/", "matlab/", "third_party/")
RUNTIME_SUFFIXES = {".py", ".sh", ".m", ".json", ".yaml", ".yml"}
FORBIDDEN = "/data/" + "dengyz/code/Code_10w"
SELF = "scripts/verify_no_legacy_code_path.py"


def legacy_hits(root: Path) -> list[str]:
    """Return tracked runtime/config paths containing the retired code root."""
    tracked = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    hits = []
    for relative in tracked:
        path = root / relative
        if relative == SELF or not relative.startswith(RUNTIME_SURFACES):
            continue
        if path.suffix not in RUNTIME_SUFFIXES or not path.is_file():
            continue
        if FORBIDDEN in path.read_text(encoding="utf-8", errors="ignore"):
            hits.append(relative)
    return sorted(hits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    hits = legacy_hits(args.root.resolve())
    if hits:
        raise SystemExit("legacy runtime/config path found:\n" + "\n".join(hits))
    print("NO LEGACY RUNTIME/CONFIG CODE_10W PATH: PASS")


if __name__ == "__main__":
    main()
