#!/usr/bin/env python3
"""Reject historical dataset roots only on active project runtime surfaces."""
from __future__ import annotations

import argparse
from pathlib import Path


FORBIDDEN_ROOTS = (
    "/data/dengyz/dataset/Code_10w_v1",
    "/data/dengyz/dataset/mapping_subject_dicom",
)
ROOT_MANUALS = ("README_en.md", "README_zh-CN.md")
ACTIVE_SCRIPTS = (
    "scripts/run_pipeline.sh",
    "scripts/verify_key_data_layout.sh",
    "scripts/verify_project_integrity.py",
    "scripts/verify_runtime_contract.py",
    "scripts/verify_scientific_freeze.py",
    "scripts/verify_no_legacy_code_path.py",
)
SELF = "scripts/verify_self_contained_project.py"


def active_paths(root: Path) -> list[Path]:
    paths = [root / name for name in ROOT_MANUALS]
    paths.extend(path for path in (root / "pipeline").glob("**/*") if path.is_file() and (path.suffix == ".sh" or path.name.startswith("README")))
    paths.extend(root / name for name in ACTIVE_SCRIPTS)
    return sorted({path for path in paths if path.is_file()})


def active_hits(root: Path) -> list[str]:
    hits: list[str] = []
    for path in active_paths(root):
        relative = path.relative_to(root).as_posix()
        if relative == SELF:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(forbidden in text for forbidden in FORBIDDEN_ROOTS):
            hits.append(relative)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    hits = active_hits(root)
    if hits:
        raise SystemExit("active runtime dependency on a historical dataset root:\n" + "\n".join(hits))
    print("SELF-CONTAINED ACTIVE PATHS: PASS (migration/provenance metadata not scanned)")


if __name__ == "__main__":
    main()
