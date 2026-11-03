from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "build_active_dependency_inventory.py"
ALLOWED = {
    "ACTIVE_ENTRYPOINT", "ACTIVE_EXECUTED", "ACTIVE_TRANSITIVE", "CONDITIONAL_ACTIVE",
    "COMPATIBILITY_IMPORT_ONLY", "CONFIG_RUNTIME_ASSET", "THIRD_PARTY_ACTIVE", "TEST_ONLY",
    "DOCUMENTATION_ONLY", "LEGACY_UNUSED_BY_PIPELINE", "UNRESOLVED_DYNAMIC",
}


def test_inventory_generator_is_data_free_and_deterministic(tmp_path: Path) -> None:
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    trace = tmp_path / "trace.md"
    for output in (first, second):
        result = subprocess.run(
            [sys.executable, str(GENERATOR), "--root", str(ROOT), "--json", str(output), "--markdown", str(trace)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr + result.stdout
    assert first.read_bytes() == second.read_bytes()
    inventory = json.loads(first.read_text(encoding="utf-8"))
    assert len(inventory["entrypoints"]) == 9
    assert len({node["path"] for node in inventory["nodes"]}) == len(inventory["nodes"])
    assert {node["classification"] for node in inventory["nodes"]} <= ALLOWED
    paths = {node["path"] for node in inventory["nodes"]}
    for entrypoint in inventory["entrypoints"]:
        assert entrypoint in paths
    for edge in inventory["edges"]:
        assert edge["from"] in paths
        assert edge["to"] in paths or edge["edge_type"] == "UNRESOLVED_DYNAMIC"
    assert len({tuple(sorted(edge.items())) for edge in inventory["edges"]}) == len(inventory["edges"])
    for node in inventory["nodes"]:
        assert {"path", "language", "classification", "functional_area", "reason", "evidence", "inputs", "outputs", "git_sha"} <= node.keys()
        if node["classification"].startswith("ACTIVE_"):
            assert (ROOT / node["path"]).exists()
    assert inventory["duplicate_or_proxy_modules"]
    assert "06" in inventory["stages"]
    assert trace.is_file()
