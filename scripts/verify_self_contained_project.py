#!/usr/bin/env python3
"""Reject historical dataset roots only on active project runtime surfaces."""
from __future__ import annotations

import argparse
import os
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
PROJECT_OWNED_SURFACES = ("data", "runs", "subjects", "runtime/model_assets")


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


def _surface_ancestors(surface: Path, root: Path) -> list[Path]:
    """Return existing lexical ancestors, including a symlinked surface root."""
    ancestors: list[Path] = []
    current = surface
    while current != root:
        ancestors.append(current)
        current = current.parent
    return ancestors


def _symlink_paths(surface: Path, root: Path) -> list[Path]:
    """Find symlinks on a project-owned surface without following directory links."""
    found = [path for path in _surface_ancestors(surface, root) if path.is_symlink()]
    if surface.is_symlink() or not surface.is_dir():
        return found
    for current, directories, files in os.walk(surface, followlinks=False):
        found.extend(Path(current) / name for name in (*directories, *files) if (Path(current) / name).is_symlink())
    return found


def symlink_issues(root: Path) -> list[dict[str, str]]:
    """Report external and broken links in project-owned data/runtime surfaces."""
    issues: list[dict[str, str]] = []
    seen: set[Path] = set()
    for relative in PROJECT_OWNED_SURFACES:
        surface = root / relative
        for link in _symlink_paths(surface, root):
            if link in seen:
                continue
            seen.add(link)
            raw_target = os.readlink(link)
            try:
                resolved = link.resolve(strict=True)
            except (FileNotFoundError, OSError, RuntimeError):
                issues.append({
                    "path": str(link),
                    "raw_target": raw_target,
                    "resolved": str(link.resolve(strict=False)),
                    "reason": "broken",
                })
                continue
            if resolved != root and root not in resolved.parents:
                issues.append({
                    "path": str(link),
                    "raw_target": raw_target,
                    "resolved": str(resolved),
                    "reason": "external",
                })
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    hits = active_hits(root)
    links = symlink_issues(root)
    if hits or links:
        messages: list[str] = []
        if hits:
            messages.append("active runtime dependency on a historical dataset root:\n" + "\n".join(hits))
        if links:
            messages.append(f"project-owned symlink audit failed ({len(links)} violation(s))")
            for issue in links:
                messages.append(
                    "symlink: {path}\nraw target: {raw_target}\nresolved target: {resolved}\nreason: {reason}".format(**issue)
                )
        raise SystemExit("\n".join(messages))
    print("EXTERNAL SYMLINKS: 0")
    print("SELF-CONTAINED ACTIVE PATHS: PASS (migration/provenance metadata not scanned)")


if __name__ == "__main__":
    main()
