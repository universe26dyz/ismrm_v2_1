from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = {"ACTIVE_ENTRYPOINT", "ACTIVE_EXECUTED", "ACTIVE_TRANSITIVE", "CONDITIONAL_ACTIVE"}


def test_active_scientific_python_authorities_are_not_left_under_common(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.json"
    trace_path = tmp_path / "trace.md"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_active_dependency_inventory.py",
            "--root",
            ".",
            "--json",
            str(inventory_path),
            "--markdown",
            str(trace_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    remaining = [
        node["path"]
        for node in inventory["nodes"]
        if node["path"].startswith("common/")
        and node["path"].endswith(".py")
        and any(role in ACTIVE for role in node["stage_roles"].values())
    ]
    assert remaining == []
