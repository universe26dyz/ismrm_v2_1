from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_completed_migrations_define_the_clean_package_implementation_set() -> None:
    result = subprocess.run([sys.executable, "scripts/verify_clean_refactor_invariants.py"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    skeleton = ROOT / "src/ismrm_v2"
    implementation = {path.relative_to(skeleton).as_posix() for path in skeleton.rglob("*.py") if path.name != "__init__.py"}
    manifest = json.loads((ROOT / "docs/clean_code_migration_manifest.json").read_text(encoding="utf-8"))
    expected = {
        entry["current_path"].removeprefix("src/ismrm_v2/")
        for entry in manifest["entries"]
        if entry["current_path"].startswith("src/ismrm_v2/")
        and entry.get("migration_status", "").startswith("MIGRATED_PHASE_")
    }
    assert implementation == expected
