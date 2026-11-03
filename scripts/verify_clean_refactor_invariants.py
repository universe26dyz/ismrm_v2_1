#!/usr/bin/env python3
"""Verify that Phase 1 adds planning structure without changing runtime authority."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = {"ACTIVE_ENTRYPOINT", "ACTIVE_EXECUTED", "ACTIVE_TRANSITIVE", "CONDITIONAL_ACTIVE", "CONFIG_RUNTIME_ASSET", "THIRD_PARTY_ACTIVE"}


def main(root: Path = ROOT) -> None:
    inventory = json.loads((root / "docs/active_dependency_inventory.json").read_text(encoding="utf-8"))
    freeze = json.loads((root / "configs/scientific_freeze_contract.json").read_text(encoding="utf-8"))
    proposed = freeze["proposed"]
    required = {
        "experiment_id": "Q002S640_B6_roi_same_anchor_mse",
        "training_psf_samples": 8,
        "anchor_batch_size": 640,
        "weights_per_anchor": 10,
        "signal_residual_count": 6400,
        "stage_a_iterations": 2000,
        "stage_b_iterations": 4000,
    }
    if any(proposed.get(key) != value for key, value in required.items()):
        raise SystemExit("formal Q002 authority changed")
    if not (root / "scripts/verify_scientific_freeze.py").is_file():
        raise SystemExit("scientific freeze checker missing")
    missing = [node["path"] for node in inventory["nodes"] if not (root / node["path"]).is_file()]
    if missing:
        raise SystemExit(f"inventory source missing: {missing}")
    for node in inventory["nodes"]:
        if any(role in ACTIVE for role in node["stage_roles"].values()) and "standard" in node["path"].lower():
            raise SystemExit(f"active standardization route detected: {node['path']}")
    for entrypoint in inventory["entrypoints"]:
        if not (root / entrypoint).is_file():
            raise SystemExit(f"pipeline entrypoint missing: {entrypoint}")
    skeleton = root / "src/ismrm_v2"
    unexpected = [path.relative_to(skeleton).as_posix() for path in skeleton.rglob("*.py") if path.name != "__init__.py"]
    if unexpected:
        raise SystemExit(f"clean skeleton contains implementation: {unexpected}")
    result = subprocess.run(["git", "grep", "-nE", r"(^|[[:space:]])(from|import)[[:space:]]+ismrm_v2|python[^\n]*-m[[:space:]]+ismrm_v2", "--", "pipeline", "common", "scripts"], cwd=root, text=True, capture_output=True)
    if result.returncode == 0:
        raise SystemExit(f"formal runtime imports clean skeleton:\n{result.stdout}")
    if result.returncode != 1:
        raise SystemExit(result.stderr)
    subprocess.check_call([sys.executable, str(root / "scripts/verify_server_local_state_policy.py"), str(root)])
    print("CLEAN REFACTOR INVARIANTS: PASS")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT)
