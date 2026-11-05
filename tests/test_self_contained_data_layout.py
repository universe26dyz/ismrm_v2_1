"""Regression tests for the project-local data-layout contract."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER = ROOT / "scripts" / "materialize_subject_pipeline_data.py"
SELF_CONTAINED = ROOT / "scripts" / "verify_self_contained_project.py"
LAYOUT_AUDIT = ROOT / "scripts" / "verify_key_data_layout.sh"


def _materializer_module():
    spec = importlib.util.spec_from_file_location("materializer", MATERIALIZER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_file(path: Path, content: str = "payload") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_materializer_keeps_deployment_and_archive_manifests_separate(tmp_path: Path) -> None:
    project, source, subject = tmp_path / "project", tmp_path / "source", "CYJ"
    deployment = source / "deployment_manifest.json"
    _write_file(deployment, json.dumps({"schema_version": "deployment-v1.1", "kind": "deployment"}))
    _write_file(source / "raw" / "sax" / "001.dcm")
    _write_file(source / "preprocessed" / "sax" / "preprocessed.mat")
    _write_file(source / "prepared" / "sax" / "observations.npz")
    _write_file(source / "roi" / "20261026_scmr_native_roi_bundle" / "roi.json")
    _write_file(source / "run" / "sparse_inputs" / subject / "L100_fold0" / "sparse_manifest.json")
    _write_file(source / "run" / "config" / "resolved.json")
    _write_file(source / "run" / "reconstruction" / "proposed" / subject / "L100_fold0" / "T1_3D.nii.gz")

    _materializer_module().materialize_subject(
        project_root=project,
        subject=subject,
        run_root=source / "run",
        raw_dicom_subject_root=source / "raw",
        preprocessed_subject_root=source / "preprocessed",
        prepared_subject_root=source / "prepared",
        roi_bundle=source / "roi" / "20261026_scmr_native_roi_bundle",
        deployment_manifest=deployment,
    )

    subject_root = project / "subjects" / subject
    assert json.loads((subject_root / "deployment_manifest.json").read_text(encoding="utf-8"))["kind"] == "deployment"
    archive = json.loads((subject_root / "archive_manifest.json").read_text(encoding="utf-8"))
    assert archive["schema"] == "ismrm_v2_1_subject_pipeline_data/v2"
    assert (subject_root / "sha256.jsonl").is_file()
    assert (project / "data" / "raw_dicom" / subject / "sax" / "001.dcm").is_file()
    assert (project / "data" / "roi" / subject / "20261026_scmr_native_roi_bundle" / "roi.json").is_file()
    frozen = project / "runs" / "frozen_baseline" / subject
    assert (frozen / "preprocessed" / subject / "sax" / "preprocessed.mat").is_file()
    assert (frozen / "prepared" / subject / "sax" / "observations.npz").is_file()
    assert (frozen / "sparse_inputs" / subject / "L100_fold0" / "sparse_manifest.json").is_file()
    assert not (subject_root / "manifest.json").exists()

    (subject_root / "sha256.jsonl").write_text("different metadata\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 manifest exists"):
        _materializer_module().materialize_subject(
            project_root=project,
            subject=subject,
            run_root=source / "run",
            raw_dicom_subject_root=source / "raw",
            preprocessed_subject_root=source / "preprocessed",
            prepared_subject_root=source / "prepared",
            roi_bundle=source / "roi" / "20261026_scmr_native_roi_bundle",
            deployment_manifest=deployment,
        )


def test_self_contained_validator_rejects_active_path_but_not_migration_provenance(tmp_path: Path) -> None:
    bad_active = tmp_path / "pipeline" / "01_RAW_DICOM" / "bad.sh"
    _write_file(bad_active, "ROOT=/data/dengyz/dataset/Code_10w_v1\n")
    provenance = tmp_path / "docs" / "migration_provenance.md"
    _write_file(provenance, "copied from /data/dengyz/dataset/Code_10w_v1\n")

    rejected = subprocess.run([sys.executable, str(SELF_CONTAINED), "--root", str(tmp_path)], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "active runtime dependency" in rejected.stderr

    bad_active.unlink()
    accepted = subprocess.run([sys.executable, str(SELF_CONTAINED), "--root", str(tmp_path)], capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stderr + accepted.stdout


def test_self_contained_validator_accepts_current_active_tree() -> None:
    result = subprocess.run([sys.executable, str(SELF_CONTAINED), "--root", str(ROOT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "SELF-CONTAINED ACTIVE PATHS: PASS" in result.stdout


def _run_self_contained(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SELF_CONTAINED), "--root", str(root)], capture_output=True, text=True)


def test_self_contained_validator_rejects_external_absolute_symlink(tmp_path: Path) -> None:
    root, external = tmp_path / "project", tmp_path / "external_dataset"
    external.mkdir()
    link = root / "runs" / "frozen_baseline" / "CYJ" / "preprocessed"
    link.parent.mkdir(parents=True)
    link.symlink_to(external, target_is_directory=True)

    result = _run_self_contained(root)

    assert result.returncode != 0
    assert "external" in result.stderr
    assert str(link) in result.stderr
    assert str(external) in result.stderr


def test_self_contained_validator_rejects_external_relative_symlink(tmp_path: Path) -> None:
    root, external = tmp_path / "project", tmp_path / "external_dataset"
    external.mkdir()
    link = root / "runs" / "frozen_baseline" / "CYJ" / "preprocessed"
    link.parent.mkdir(parents=True)
    link.symlink_to(Path("../../../../external_dataset"), target_is_directory=True)

    result = _run_self_contained(root)

    assert result.returncode != 0
    assert "reason: external" in result.stderr
    assert "../../../../external_dataset" in result.stderr


def test_self_contained_validator_rejects_broken_symlink(tmp_path: Path) -> None:
    root = tmp_path / "project"
    link = root / "subjects" / "CYJ" / "missing"
    link.parent.mkdir(parents=True)
    link.symlink_to("missing_target")

    result = _run_self_contained(root)

    assert result.returncode != 0
    assert "reason: broken" in result.stderr
    assert str(link) in result.stderr


def test_self_contained_validator_allows_internal_symlink(tmp_path: Path) -> None:
    root = tmp_path / "project"
    real = root / "data" / "real"
    real.mkdir(parents=True)
    (real / "marker").write_text("ok", encoding="utf-8")
    (root / "data" / "alias").symlink_to(real, target_is_directory=True)

    result = _run_self_contained(root)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "EXTERNAL SYMLINKS: 0" in result.stdout


def test_frozen_layout_audit_uses_only_project_local_contracts() -> None:
    source = LAYOUT_AUDIT.read_text(encoding="utf-8")

    assert 'RAW_ROOT="$PROJECT_ROOT/data/raw_dicom"' in source
    assert 'ROI_ROOT="$PROJECT_ROOT/data/roi"' in source
    assert 'BASE="$PROJECT_ROOT/runs/frozen_baseline"' in source
    assert "338f2c0b8db90851f733ed31caa2c99914605fed3f7215ec2f69f19eba59583b" in source
    assert "15981fbe93d2c70e14755ea756a02856c9383481e7d92d239619268a628a29b5" in source
    assert 'if [ "$SUBJECT" = HHZ ]' not in source
    assert "/data/dengyz/dataset" not in source
