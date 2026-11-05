"""Regression tests for the final self-contained path hardening fixes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_inventory_generator():
    path = ROOT / "scripts" / "build_active_dependency_inventory.py"
    spec = importlib.util.spec_from_file_location("inventory_generator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage02_manuals_match_wrapper_argument_contract() -> None:
    expected = '"$PROJECT_ROOT/subjects/$SUBJECT/deployment_manifest.json" ' + '\\\n  matlab "$PYTHON"'
    for name in ("README_en.md", "README_zh-CN.md"):
        source = (ROOT / "pipeline" / "02_PREPROCESS_MATLAB" / name).read_text(encoding="utf-8")
        assert expected in source
        assert "protocol_scmr_v5_10_100.yaml" not in source


def test_stage07_and_stage08_manuals_use_public_wrappers_and_canonical_roi() -> None:
    expected_roi = 'ROI_BUNDLE="$PROJECT_ROOT/data/roi/$SUBJECT/20261026_scmr_native_roi_bundle"'
    for stage, wrapper in (
        ("07_FINAL_POSE", "pipeline/07_FINAL_POSE/run_stage.sh"),
        ("08_EVALUATION_AND_VISUALIZATION", "pipeline/08_EVALUATION_AND_VISUALIZATION/run_stage.sh"),
    ):
        for name in ("README_en.md", "README_zh-CN.md"):
            source = (ROOT / "pipeline" / stage / name).read_text(encoding="utf-8")
            assert "data/roi/$SUBJECT/20261026_scmr_native_roi_bundle" in source
            assert "subjects/$SUBJECT/roi" not in source
            assert wrapper in source
        assert expected_roi in "\n".join(
            (ROOT / "pipeline" / stage / name).read_text(encoding="utf-8")
            for name in ("README_en.md", "README_zh-CN.md")
        )


def test_model_assets_are_project_local_not_external_inventory_boundaries(tmp_path: Path) -> None:
    generator = _load_inventory_generator()
    output = tmp_path / "inventory.json"
    inventory = generator.scan(ROOT)
    output.write_text(json.dumps(inventory), encoding="utf-8")

    external = {item["id"] for item in inventory["external_runtime_boundaries"]}
    assert {"matlab", "nesvor", "cuda"} <= external
    assert not external.intersection({"decoder_scmr_v5_10_100.pth", "b6_model.pt"})

    assets = {item["id"]: item for item in inventory["project_local_runtime_assets"]}
    assert assets == {
        "decoder_scmr_v5_10_100.pth": {
            "id": "decoder_scmr_v5_10_100.pth",
            "expected_sha256": "338f2c0b8db90851f733ed31caa2c99914605fed3f7215ec2f69f19eba59583b",
            "git_tracked": False,
            "path": "runtime/model_assets/decoder_scmr_v5_10_100.pth",
            "project_local": True,
            "required_by_stages": ["06"],
        },
        "b6_model.pt": {
            "id": "b6_model.pt",
            "expected_sha256": "15981fbe93d2c70e14755ea756a02856c9383481e7d92d239619268a628a29b5",
            "git_tracked": False,
            "path": "runtime/model_assets/b6_model.pt",
            "project_local": True,
            "required_by_stages": ["06"],
        },
    }
