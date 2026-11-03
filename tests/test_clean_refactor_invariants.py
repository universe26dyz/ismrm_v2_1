from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase2a_allows_only_subject_admission_io_in_clean_package() -> None:
    result = subprocess.run([sys.executable, "scripts/verify_clean_refactor_invariants.py"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    skeleton = ROOT / "src/ismrm_v2"
    implementation = {path.relative_to(skeleton).as_posix() for path in skeleton.rglob("*.py") if path.name != "__init__.py"}
    assert implementation == {"io/validate_subject_id.py", "io/build_subject_deployment_manifest.py"}
