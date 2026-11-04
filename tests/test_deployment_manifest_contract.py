"""Regression contract for the clean deployment-manifest loader."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
NEW_PATH = SRC / "ismrm_v2/io/deployment_manifest.py"
OLD_PATH = ROOT / "common/scripts/deployment_manifest.py"
CALLERS = (
    ROOT / "src/ismrm_v2/preprocessing/validation.py",
    ROOT / "src/ismrm_v2/observations/prepare_all.py",
)


def _loader():
    sys.path.insert(0, str(SRC))
    return importlib.import_module("ismrm_v2.io.deployment_manifest").load_deployment_manifest


def _stack(stack: str = "sax", **overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "stack": stack,
        "local_dicom_dir": "/local/subject/sax",
        "server_dicom_dir": "/server/subject/sax",
    }
    entry.update(overrides)
    return entry


def _subject(subject_id: str = "subject-01", stacks: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {"subject_id": subject_id, "stacks": stacks if stacks is not None else [_stack()]}


def _write_manifest(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loader_accepts_v11_and_preserves_explicit_entry_order_and_fields(tmp_path: Path) -> None:
    load = _loader()
    payload = {
        "schema_version": "deployment-v1.1",
        "subjects": [
            _subject("subject-01", [_stack("2ch"), _stack("sax")]),
            _subject("subject-02", [_stack("4ch")]),
        ],
    }
    entries = load(_write_manifest(tmp_path, payload))
    assert entries == [
        {"subject_id": "subject-01", "stack": "2ch", "local_dicom_dir": "/local/subject/sax", "server_dicom_dir": "/server/subject/sax"},
        {"subject_id": "subject-01", "stack": "sax", "local_dicom_dir": "/local/subject/sax", "server_dicom_dir": "/server/subject/sax"},
        {"subject_id": "subject-02", "stack": "4ch", "local_dicom_dir": "/local/subject/sax", "server_dicom_dir": "/server/subject/sax"},
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": "deployment-v1", "subjects": [_subject()]},
        {},
        {"schema_version": "deployment-v1.1", "subjects": []},
        {"schema_version": "deployment-v1.1", "subjects": [_subject(), _subject()]},
        {"schema_version": "deployment-v1.1", "subjects": [_subject(stacks=[_stack("long_axis")])]},
        {"schema_version": "deployment-v1.1", "subjects": [_subject(stacks=[_stack(), _stack()])]},
        {"schema_version": "deployment-v1.1", "subjects": [_subject(stacks=[_stack(local_dicom_dir="")])]},
        {"schema_version": "deployment-v1.1", "subjects": [_subject(stacks=[_stack(server_dicom_dir="")])]},
    ],
)
def test_loader_rejects_invalid_deployment_contracts(tmp_path: Path, payload: object) -> None:
    with pytest.raises(ValueError):
        _loader()(_write_manifest(tmp_path, payload))


def test_clean_path_is_authoritative_and_deployment_callers_import_clean_loader() -> None:
    assert not OLD_PATH.exists()
    assert NEW_PATH.is_file()
    for caller in CALLERS:
        source = caller.read_text(encoding="utf-8")
        assert "from ismrm_v2.io.deployment_manifest import load_deployment_manifest" in source
