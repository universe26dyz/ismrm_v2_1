"""Phase 2A contracts for generic subject-admission I/O."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _modules():
    sys.path.insert(0, str(SRC))
    validator = importlib.import_module("ismrm_v2.io.validate_subject_id")
    builder = importlib.import_module("ismrm_v2.io.build_subject_deployment_manifest")
    return validator, builder


@pytest.mark.parametrize("subject", ["CYJ", "DYL_2", "subject-01", "subject.01", "ABC_123"])
def test_validate_subject_id_accepts_safe_generic_ids(subject: str) -> None:
    validator, _ = _modules()
    assert validator.validate_subject_id(subject) == subject


@pytest.mark.parametrize("subject", ["", ".", "..", "../CYJ", "CYJ/2ch", "a..b", "/absolute"])
def test_validate_subject_id_rejects_unsafe_ids(subject: str) -> None:
    validator, _ = _modules()
    with pytest.raises(ValueError):
        validator.validate_subject_id(subject)


def _make_stacks(root: Path, subject: str) -> None:
    for stack in ("sax", "2ch", "4ch"):
        (root / subject / stack).mkdir(parents=True, exist_ok=True)


def test_build_manifest_uses_local_root_by_default(tmp_path: Path) -> None:
    _, builder = _modules()
    raw_root = tmp_path / "raw"
    _make_stacks(raw_root, "CYJ")
    output = tmp_path / "manifest.json"

    assert builder.build_subject_deployment_manifest(raw_root, "CYJ", output) == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "deployment-v1.1"
    stacks = payload["subjects"][0]["stacks"]
    assert [stack["stack"] for stack in stacks] == ["sax", "2ch", "4ch"]
    assert [stack["local_dicom_dir"] for stack in stacks] == [
        str((raw_root / "CYJ" / name).resolve()) for name in ("sax", "2ch", "4ch")
    ]
    assert [stack["server_dicom_dir"] for stack in stacks] == [
        str((raw_root / "CYJ" / name).resolve()) for name in ("sax", "2ch", "4ch")
    ]


def test_build_manifest_uses_explicit_server_root_independently(tmp_path: Path) -> None:
    _, builder = _modules()
    raw_root, server_root = tmp_path / "local", tmp_path / "server"
    _make_stacks(raw_root, "FUTURE")
    _make_stacks(server_root, "FUTURE")
    output = tmp_path / "manifest.json"

    builder.build_subject_deployment_manifest(raw_root, "FUTURE", output, server_root)
    stacks = json.loads(output.read_text(encoding="utf-8"))["subjects"][0]["stacks"]
    assert [stack["local_dicom_dir"] for stack in stacks] == [
        str((raw_root / "FUTURE" / name).resolve()) for name in ("sax", "2ch", "4ch")
    ]
    assert [stack["server_dicom_dir"] for stack in stacks] == [
        str((server_root / "FUTURE" / name).resolve()) for name in ("sax", "2ch", "4ch")
    ]


def test_build_manifest_fails_for_missing_stack_or_existing_output(tmp_path: Path) -> None:
    _, builder = _modules()
    raw_root = tmp_path / "raw"
    _make_stacks(raw_root, "CYJ")
    (raw_root / "CYJ" / "4ch").rmdir()
    output = tmp_path / "manifest.json"
    with pytest.raises(FileNotFoundError):
        builder.build_subject_deployment_manifest(raw_root, "CYJ", output)

    _make_stacks(raw_root, "CYJ")
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        builder.build_subject_deployment_manifest(raw_root, "CYJ", output)
    with pytest.raises(ValueError):
        builder.build_subject_deployment_manifest(raw_root, "../CYJ", tmp_path / "new.json")


def test_new_paths_are_authoritative_and_pipeline_uses_modules() -> None:
    assert not (ROOT / "scripts/validate_subject_id.py").exists()
    assert not (ROOT / "scripts/build_subject_deployment_manifest.py").exists()
    assert (SRC / "ismrm_v2/io/validate_subject_id.py").is_file()
    assert (SRC / "ismrm_v2/io/build_subject_deployment_manifest.py").is_file()
    validator_source = (SRC / "ismrm_v2/io/validate_subject_id.py").read_text(encoding="utf-8")
    assert "CYJ" not in validator_source
    assert "DYL_2" not in validator_source
    for stage in sorted((ROOT / "pipeline").glob("0[1-8]_*")):
        script = stage / "run_stage.sh"
        if not script.is_file():
            continue
        source = script.read_text(encoding="utf-8")
        assert "scripts/validate_subject_id.py" not in source
        assert "-m ismrm_v2.io.validate_subject_id \"$SUBJECT\"" in source
        subprocess.run(["bash", "-n", str(script)], check=True)
    stage01 = (ROOT / "pipeline/01_RAW_DICOM/run_stage.sh").read_text(encoding="utf-8")
    assert "scripts/build_subject_deployment_manifest.py" not in stage01
    assert "-m ismrm_v2.io.build_subject_deployment_manifest" in stage01


def test_module_cli_smoke_and_builder_smoke(tmp_path: Path) -> None:
    env = {"PYTHONPATH": str(SRC)}
    for subject in ("CYJ", "FUTURE"):
        result = subprocess.run(
            [sys.executable, "-m", "ismrm_v2.io.validate_subject_id", subject],
            env=env, text=True, capture_output=True,
        )
        assert result.returncode == 0, result.stderr
    invalid = subprocess.run(
        [sys.executable, "-m", "ismrm_v2.io.validate_subject_id", "../CYJ"],
        env=env, text=True, capture_output=True,
    )
    assert invalid.returncode != 0
    raw_root = tmp_path / "raw"
    _make_stacks(raw_root, "CYJ")
    output = tmp_path / "manifest.json"
    built = subprocess.run(
        [sys.executable, "-m", "ismrm_v2.io.build_subject_deployment_manifest", "--raw-root", str(raw_root), "--subject", "CYJ", "--output", str(output)],
        env=env, text=True, capture_output=True,
    )
    assert built.returncode == 0, built.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "deployment-v1.1"
