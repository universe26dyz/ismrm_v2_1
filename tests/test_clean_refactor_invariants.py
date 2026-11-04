from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_completed_migrations_define_the_clean_package_implementation_set() -> None:
    result = subprocess.run([sys.executable, "scripts/verify_clean_refactor_invariants.py"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    skeleton = ROOT / "src/ismrm_v2"
    implementation = {path.relative_to(skeleton).as_posix() for path in skeleton.rglob("*.py") if path.name != "__init__.py"}
    assert implementation == {
        "io/validate_subject_id.py",
        "io/build_subject_deployment_manifest.py",
        "io/deployment_manifest.py",
        "preprocessing/validation.py",
        "observations/prepare_all.py",
        "observations/build_manifest.py",
        "observations/mat_v73.py",
        "observations/timing.py",
        "observations/prepare_observations.py",
        "sampling/sparse.py",
        "sampling/sparse_profile_manifest.py",
    }
