from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase1_keeps_runtime_authority_outside_clean_skeleton() -> None:
    result = subprocess.run([sys.executable, "scripts/verify_clean_refactor_invariants.py"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    skeleton = ROOT / "src/ismrm_v2"
    assert all(path.name == "__init__.py" for path in skeleton.rglob("*.py"))
