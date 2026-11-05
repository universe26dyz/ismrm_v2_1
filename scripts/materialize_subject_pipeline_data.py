#!/usr/bin/env python3
"""Copy one completed subject into the project-local v2.1 data layout.

The operation is copy-only and resumable. Existing destination files are
accepted only when their SHA256 equals the source file. Large patient data and
generated outputs remain outside Git; deployment and archive manifests keep
their distinct meanings under ``subjects/<SUBJECT>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


RUN_COMPONENTS = ("config", "mapping", "native_cache", "reconstruction", "evaluation")
ROOT_METADATA_SUFFIXES = {".json", ".jsonl", ".log", ".txt"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path, records: list[dict[str, Any]], *, label: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"{label}: source file is missing: {source}")
    source_hash = sha256(source)
    if destination.exists():
        if not destination.is_file() or sha256(destination) != source_hash:
            raise ValueError(f"{label}: destination exists with a different SHA256: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    records.append({
        "label": label,
        "source": str(source.resolve()),
        "archive": str(destination),
        "size": source.stat().st_size,
        "sha256": source_hash,
    })


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"source directory is missing: {root}")
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def _copy_tree(source: Path, destination: Path, records: list[dict[str, Any]], *, label: str) -> None:
    for path in _iter_files(source):
        relative = path.relative_to(source)
        _copy_file(path, destination / relative, records, label=f"{label}/{relative}")


def _copy_run_outputs(source: Path, destination: Path, records: list[dict[str, Any]]) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"run root is missing: {source}")
    for component in RUN_COMPONENTS:
        path = source / component
        if path.exists():
            _copy_tree(path, destination / component, records, label=f"run/{component}")
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix.lower() in ROOT_METADATA_SUFFIXES:
            _copy_file(path, destination / path.name, records, label=f"run/{path.name}")


def materialize_subject(*, project_root: str | Path, subject: str, run_root: str | Path,
                        raw_dicom_subject_root: str | Path, preprocessed_subject_root: str | Path,
                        prepared_subject_root: str | Path, roi_bundle: str | Path,
                        deployment_manifest: str | Path | None = None,
                        supplemental_proposed_root: str | Path | None = None) -> Path:
    project = Path(project_root).resolve()
    subject_root = project / "subjects" / subject
    raw_root = project / "data" / "raw_dicom" / subject
    roi_root = project / "data" / "roi" / subject / Path(roi_bundle).resolve().name
    frozen_root = project / "runs" / "frozen_baseline" / subject
    records: list[dict[str, Any]] = []

    _copy_tree(Path(raw_dicom_subject_root), raw_root, records, label="data/raw_dicom")
    _copy_tree(Path(preprocessed_subject_root), frozen_root / "preprocessed" / subject, records, label="frozen/preprocessed")
    _copy_tree(Path(prepared_subject_root), frozen_root / "prepared" / subject, records, label="frozen/prepared")
    _copy_tree(Path(roi_bundle), roi_root, records, label="data/roi")

    sparse_source = Path(run_root).resolve() / "sparse_inputs" / subject
    _copy_tree(sparse_source, frozen_root / "sparse_inputs" / subject, records, label="frozen/sparse_inputs")

    _copy_run_outputs(Path(run_root).resolve(), frozen_root, records)

    supplemental_source = None
    if supplemental_proposed_root is not None:
        supplemental_source = Path(supplemental_proposed_root).resolve()
        formal_proposed = Path(run_root).resolve() / "reconstruction" / "proposed" / subject
        if formal_proposed.exists():
            raise ValueError(
                "supplemental Proposed source is ambiguous because the formal "
                f"run already contains {formal_proposed}"
            )
        _copy_tree(
            supplemental_source,
            frozen_root / "reconstruction" / "proposed" / subject,
            records,
            label="supplemental/proposed",
        )

    if deployment_manifest is not None:
        _copy_file(Path(deployment_manifest), subject_root / "deployment_manifest.json", records,
                   label="subject/deployment_manifest.json")

    subject_root.mkdir(parents=True, exist_ok=True)
    sha_manifest = subject_root / "sha256.jsonl"
    sha_content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in sorted(records, key=lambda item: item["archive"])
    )
    if sha_manifest.exists() and sha_manifest.read_text(encoding="utf-8") != sha_content:
        raise ValueError(f"SHA256 manifest exists with different content: {sha_manifest}")
    sha_manifest.write_text(sha_content, encoding="utf-8")

    manifest = {
        "schema": "ismrm_v2_1_subject_pipeline_data/v2",
        "subject": subject,
        "archive_policy": "copy-only; existing files must have identical SHA256",
        "source_run_root": str(Path(run_root).resolve()),
        "source_inputs": {
            "raw_dicom": str(Path(raw_dicom_subject_root).resolve()),
            "preprocessed": str(Path(preprocessed_subject_root).resolve()),
            "prepared": str(Path(prepared_subject_root).resolve()),
            "roi": str(Path(roi_bundle).resolve()),
            "sparse_inputs": str(sparse_source),
        },
        "supplemental_sources": {
            "proposed": str(supplemental_source) if supplemental_source is not None else None,
        },
        "project_data": {
            "raw_dicom": str(raw_root.relative_to(project)),
            "roi": str(roi_root.relative_to(project)),
            "frozen_baseline": str(frozen_root.relative_to(project)),
        },
        "file_count": len(records),
        "sha256_manifest": str(sha_manifest.relative_to(project)),
        "gitignored_data": True,
    }
    archive_manifest = subject_root / "archive_manifest.json"
    if archive_manifest.exists() and archive_manifest.read_bytes() != (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"):
        raise ValueError(f"archive manifest exists with different content: {archive_manifest}")
    archive_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return subject_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--raw-dicom-subject-root", required=True)
    parser.add_argument("--preprocessed-subject-root", required=True)
    parser.add_argument("--prepared-subject-root", required=True)
    parser.add_argument("--roi-bundle", required=True)
    parser.add_argument("--deployment-manifest")
    parser.add_argument("--supplemental-proposed-root")
    print(materialize_subject(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
