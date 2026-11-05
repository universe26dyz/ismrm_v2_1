"""Data-free contract tests for generic subject admission and Stage 01."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = "ismrm_v2.io.validate_subject_id"
STAGE01 = ROOT / "pipeline" / "01_RAW_DICOM" / "run_stage.sh"


def test_generic_subject_validator_accepts_new_subject_and_rejects_traversal() -> None:
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    accepted = subprocess.run([sys.executable, "-m", VALIDATOR, "NEW_SUBJECT_001"], env=environment, capture_output=True, text=True)
    rejected = subprocess.run([sys.executable, "-m", VALIDATOR, "../escape"], env=environment, capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stderr
    assert rejected.returncode != 0


def test_stage01_builds_a_subject_manifest_from_actual_raw_stack_directories(tmp_path: Path) -> None:
    subject, raw_root, manifest_root = "NEW_SUBJECT_001", tmp_path / "raw", tmp_path / "subjects"
    for stack in ("sax", "2ch", "4ch"):
        stack_dir = raw_root / subject / stack
        stack_dir.mkdir(parents=True)
        for index in range(10):
            (stack_dir / f"{index:02d}.dcm").touch()
    environment = dict(os.environ, MANIFEST_ROOT=str(manifest_root))
    result = subprocess.run(["bash", str(STAGE01), subject, str(raw_root)], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads((manifest_root / subject / "deployment_manifest.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "deployment-v1.1"
    assert payload["subjects"][0]["subject_id"] == subject
    stacks = payload["subjects"][0]["stacks"]
    assert [entry["stack"] for entry in stacks] == ["sax", "2ch", "4ch"]
    assert all(Path(entry["local_dicom_dir"]).is_absolute() for entry in stacks)
    assert all(Path(entry["server_dicom_dir"]).is_absolute() for entry in stacks)


def test_stage01_defaults_to_the_project_local_raw_dicom_root() -> None:
    source = STAGE01.read_text(encoding="utf-8")

    assert 'RAW_ROOT=${2:-"$PROJECT_ROOT/data/raw_dicom"}' in source
    assert "/data/dengyz/dataset" not in source
