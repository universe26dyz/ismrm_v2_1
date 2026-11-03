#!/usr/bin/env python3
"""Copy one completed subject's pipeline inputs and outputs into ismrm_v2_1.

The operation is copy-only and resumable. Existing destination files are
accepted only when their SHA256 equals the source file. Large patient data and
generated outputs remain outside Git; the subject manifest and SHA256 JSONL
are the tracked provenance records.
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
                        deployment_manifest: str | Path | None = None) -> Path:
    project = Path(project_root).resolve()
    subject_root = project / "subjects" / subject
    input_root = subject_root / "inputs"
    output_root = project / "runs" / subject
    records: list[dict[str, Any]] = []

    _copy_tree(Path(raw_dicom_subject_root), input_root / "raw_dicom", records, label="input/raw_dicom")
    _copy_tree(Path(preprocessed_subject_root), input_root / "preprocessed", records, label="input/preprocessed")
    _copy_tree(Path(prepared_subject_root), input_root / "prepared", records, label="input/prepared")
    _copy_tree(Path(roi_bundle), input_root / "roi", records, label="input/roi")

    sparse_source = Path(run_root).resolve() / "sparse_inputs" / subject
    _copy_tree(sparse_source, input_root / "sparse_inputs", records, label="input/sparse_inputs")

    _copy_run_outputs(Path(run_root).resolve(), output_root, records)

    if deployment_manifest is not None:
        _copy_file(Path(deployment_manifest), subject_root / "deployment_manifest.json", records,
                   label="subject/deployment_manifest.json")

    subject_root.mkdir(parents=True, exist_ok=True)
    sha_manifest = subject_root / "sha256.jsonl"
    with sha_manifest.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item["archive"]):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema": "ismrm_v2_1_subject_pipeline_data/v1",
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
        "archive_inputs": str(input_root.relative_to(project)),
        "archive_outputs": str(output_root.relative_to(project)),
        "file_count": len(records),
        "sha256_manifest": str(sha_manifest.relative_to(project)),
        "gitignored_data": True,
    }
    (subject_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
    print(materialize_subject(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
