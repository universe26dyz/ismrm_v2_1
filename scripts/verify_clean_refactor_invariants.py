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
    package = root / "src/ismrm_v2"
    migrated = {
        "io/validate_subject_id.py",
        "io/build_subject_deployment_manifest.py",
        "io/deployment_manifest.py",
    }
    unexpected = [path.relative_to(package).as_posix() for path in package.rglob("*.py") if path.name != "__init__.py" and path.relative_to(package).as_posix() not in migrated]
    if unexpected:
        raise SystemExit(f"unexpected clean-package implementation: {unexpected}")
    manifest = json.loads((root / "docs/clean_code_migration_manifest.json").read_text(encoding="utf-8"))
    entries = {entry["current_path"]: entry for entry in manifest["entries"]}
    for relative in migrated:
        entry = entries.get(f"src/ismrm_v2/{relative}")
        expected_status = "MIGRATED_PHASE_2B" if relative == "io/deployment_manifest.py" else "MIGRATED_PHASE_2A"
        if entry is None or entry.get("migration_status") != expected_status:
            raise SystemExit(f"completed package module is not recorded as migrated: {relative}")
    for deferred in ("proposed", "fit_first", "evaluation"):
        implementation = [path.relative_to(package).as_posix() for path in (package / deferred).rglob("*.py") if path.name != "__init__.py"]
        if implementation:
            raise SystemExit(f"Phase 2A unexpectedly migrated {deferred}: {implementation}")
    subprocess.check_call([sys.executable, str(root / "scripts/verify_server_local_state_policy.py"), str(root)])
    print("CLEAN REFACTOR INVARIANTS: PASS")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT)
