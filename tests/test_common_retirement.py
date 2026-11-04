from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SURFACES = ("pipeline", "scripts", "src", "configs", "matlab", "third_party")


def test_common_is_absent_from_runtime_tree_and_inventory() -> None:
    assert not (ROOT / "common").exists()
    inventory = json.loads((ROOT / "docs/active_dependency_inventory.json").read_text(encoding="utf-8"))
    assert not [node["path"] for node in inventory["nodes"] if node["path"].startswith("common/")]
    references = []
    for surface in RUNTIME_SURFACES:
        for path in (ROOT / surface).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh", ".m", ".json", ".yaml", ".yml"}:
                if "common/" in path.read_text(encoding="utf-8", errors="ignore"):
                    references.append(path.relative_to(ROOT).as_posix())
    assert not references
