from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "build_active_dependency_inventory.py"
ACTIVE_INTERNAL = {"ACTIVE_ENTRYPOINT", "ACTIVE_EXECUTED", "ACTIVE_TRANSITIVE", "CONDITIONAL_ACTIVE", "CONFIG_RUNTIME_ASSET", "THIRD_PARTY_ACTIVE"}
ALLOWED = ACTIVE_INTERNAL | {"COMPATIBILITY_IMPORT_ONLY", "TEST_ONLY", "DOCUMENTATION_ONLY", "LEGACY_UNUSED_BY_PIPELINE", "EXTERNAL_RUNTIME_BOUNDARY", "UNRESOLVED_DYNAMIC"}


def _inventory(tmp_path: Path) -> dict:
    output, trace = tmp_path / "inventory.json", tmp_path / "trace.md"
    result = subprocess.run([sys.executable, str(GENERATOR), "--root", str(ROOT), "--json", str(output), "--markdown", str(trace)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads(output.read_text(encoding="utf-8"))


def test_inventory_is_deterministic_and_has_no_missing_internal_active_nodes(tmp_path: Path) -> None:
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    for output in (first, second):
        result = subprocess.run([sys.executable, str(GENERATOR), "--root", str(ROOT), "--json", str(output), "--markdown", str(tmp_path / "trace.md")], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr + result.stdout
    assert first.read_bytes() == second.read_bytes()
    inventory = json.loads(first.read_text(encoding="utf-8"))
    assert len(inventory["entrypoints"]) == 9
    assert len({node["path"] for node in inventory["nodes"]}) == len(inventory["nodes"])
    assert {node["classification"] for node in inventory["nodes"]} <= ALLOWED
    for node in inventory["nodes"]:
        assert {"path", "language", "classification", "functional_area", "reason", "evidence", "inputs", "outputs", "git_sha", "discovery_method", "historical_name", "generic_runtime_usage"} <= node.keys()
        if node["classification"] in ACTIVE_INTERNAL and node["path"].startswith(("common/", "pipeline/", "scripts/", "configs/", "runtime/")):
            assert (ROOT / node["path"]).exists(), node


def test_stage08_paths_and_external_boundaries_are_semantically_separate(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    paths = {node["path"] for node in inventory["nodes"]}
    for path in (
        "common/paper_experiments/joint_vs_2dfit/aha/common_native.py",
        "common/paper_experiments/joint_vs_2dfit/aha/common_native_metrics_qc.py",
        "common/paper_experiments/joint_vs_2dfit/aha/common_native_all_stack_roi_metrics.py",
    ):
        assert path in paths
    assert not any("phase2_clean/common_native" in path for path in paths)
    boundaries = {item["id"] for item in inventory["external_runtime_boundaries"]}
    assert {"matlab", "nesvor", "cuda", "decoder_scmr_v5_10_100.pth", "b6_model.pt"} <= boundaries
    assert not any(node["classification"] == "EXTERNAL_RUNTIME_BOUNDARY" for node in inventory["nodes"])


def test_edges_are_resolved_and_classifications_are_meaningful(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    paths = {node["path"] for node in inventory["nodes"]}
    for edge in inventory["edges"]:
        assert edge["from"] in paths
        assert edge["to"] in paths or edge["edge_type"] == "EXTERNAL_RUNTIME"
    assert len({tuple(sorted(edge.items())) for edge in inventory["edges"]}) == len(inventory["edges"])
    for category in ("ACTIVE_TRANSITIVE", "CONDITIONAL_ACTIVE", "COMPATIBILITY_IMPORT_ONLY"):
        assert inventory["classification_summary"].get(category, 0) > 0
    historical = next(node for node in inventory["nodes"] if "cyj11" in node["path"])
    assert historical["historical_name"] is True
    assert historical["classification"] != "LEGACY_UNUSED_BY_PIPELINE"


def test_declared_missing_repository_path_fails_closed() -> None:
    spec = importlib.util.spec_from_file_location("inventory_generator", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.DECLARED_EDGES.append(("01", module.STAGE_ENTRYPOINTS["01"], "common/does_not_exist.py", "PYTHON_CALL", "always", "test missing path", "ACTIVE_EXECUTED"))
    with pytest.raises(FileNotFoundError, match="declared active repository dependency does not exist"):
        module.scan(ROOT)
