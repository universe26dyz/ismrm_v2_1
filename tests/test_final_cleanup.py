from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_cleanup_leaves_no_legacy_or_compatibility_python_authority_in_common() -> None:
    inventory = json.loads((ROOT / "docs/active_dependency_inventory.json").read_text(encoding="utf-8"))
    leftovers = [
        node["path"]
        for node in inventory["nodes"]
        if node["path"].startswith("common/")
        and node["path"].endswith(".py")
        and node["classification"] in {"COMPATIBILITY_IMPORT_ONLY", "LEGACY_UNUSED_BY_PIPELINE"}
    ]
    assert leftovers == []
