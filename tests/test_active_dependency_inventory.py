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
STAGE02_MATLAB = {
    "common/trad/scripts/local_preprocess_all.m",
    "common/trad/scripts/local_preprocess_one.m",
    "common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/preprocess_options_v1.m",
    "common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/preprocess_stack_v1.m",
    "common/trad/modules/module_01_preprocess_matlab/hhz_original/Utility/HHMMSS2Sec.m",
    "common/trad/modules/module_01_preprocess_matlab/hhz_original/MIND_registration/deformableReg2Dmind_asym_nodisplay.m",
    "common/trad/modules/module_01_preprocess_matlab/hhz_original/MIND_registration/MIND_descriptor2D.m",
    "common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/mppca_denoise_multimap_stack.m",
}
STAGE04_PROFILES = {
    f"common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/{name}.json"
    for name in ("SAX100_2CH0_4CH0", "SAX100_2CH0_4CH100", "SAX100_2CH100_4CH0", "S2CH100_4CH50", "S2CH50_4CH100")
}
STAGE07_CLIS = {
    "common/paper_experiments/joint_vs_2dfit/aha/paired_aha_sd.py",
    "common/paper_experiments/joint_vs_2dfit/aha/paired_final_pose_through_plane.py",
}
ROLE_PRIORITY = {"ACTIVE_ENTRYPOINT": 9, "ACTIVE_EXECUTED": 8, "CONDITIONAL_ACTIVE": 7, "ACTIVE_TRANSITIVE": 6, "COMPATIBILITY_IMPORT_ONLY": 5, "CONFIG_RUNTIME_ASSET": 4, "THIRD_PARTY_ACTIVE": 3, "TEST_ONLY": 2, "LEGACY_UNUSED_BY_PIPELINE": 1}


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
    assert inventory["schema_version"] == "active-dependency-inventory/v3"
    assert len({node["path"] for node in inventory["nodes"]}) == len(inventory["nodes"])
    assert {node["classification"] for node in inventory["nodes"]} <= ALLOWED
    for node in inventory["nodes"]:
        assert {"path", "language", "classification", "stage_roles", "functional_area", "reason", "evidence", "inputs", "outputs", "git_sha", "discovery_methods", "historical_name", "generic_runtime_usage"} <= node.keys()
        assert node["stage_roles"]
        assert node["classification"] == max(node["stage_roles"].values(), key=ROLE_PRIORITY.__getitem__)
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


def test_stage02_matlab_closure_and_stage04_profiles_are_active(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    nodes = {node["path"]: node for node in inventory["nodes"]}
    for path in STAGE02_MATLAB:
        assert nodes[path]["stage_roles"]["02"] in {"ACTIVE_EXECUTED", "ACTIVE_TRANSITIVE"}
    for path in STAGE04_PROFILES:
        assert nodes[path]["stage_roles"] == {"04": "CONFIG_RUNTIME_ASSET"}


def test_stage05_matcher_and_stage07_formal_shell_chain(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    nodes = {node["path"]: node for node in inventory["nodes"]}
    matcher_root = ROOT / "common/trad/evaluation/scmr/matlab/scmr_v5_matcher"
    for source in matcher_root.rglob("*.m"):
        path = source.relative_to(ROOT).as_posix()
        assert nodes[path]["stage_roles"]["05"] == "ACTIVE_TRANSITIVE"
    wrapper = "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh"
    for path in STAGE07_CLIS:
        assert nodes[path]["stage_roles"]["07"] == "CONDITIONAL_ACTIVE"
        assert any(edge["from"] == wrapper and edge["to"] == path and edge["stage"] == "07" for edge in inventory["edges"])


def test_markdown_uses_the_stage_specific_role_not_global_role(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    node = next(node for node in inventory["nodes"] if node["path"] == "common/reconstruction_core/orchestration.py")
    assert node["stage_roles"] == {"06": "ACTIVE_EXECUTED", "07": "ACTIVE_TRANSITIVE"}
    trace = (tmp_path / "trace.md").read_text(encoding="utf-8")
    stage07 = trace.split("## Stage 07", 1)[1].split("## Stage 08", 1)[0]
    direct, transitive = stage07.split("Direct execution chain:", 1)[1].split("Transitive implementation dependencies:", 1)
    assert "common/reconstruction_core/orchestration.py" not in direct
    assert "common/reconstruction_core/orchestration.py" in transitive


def test_external_runtime_edges_cover_every_declared_stage(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    external_edges = {(edge["to"].removeprefix("external:"), edge["stage"]) for edge in inventory["edges"] if edge["edge_type"] == "EXTERNAL_RUNTIME"}
    assert {("matlab", "02"), ("matlab", "05"), ("nesvor", "05"), ("cuda", "06"), ("cuda", "07"), ("decoder_scmr_v5_10_100.pth", "06"), ("b6_model.pt", "06")} <= external_edges


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
