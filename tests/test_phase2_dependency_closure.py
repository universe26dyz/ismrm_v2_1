"""Regression test for the standalone Phase-2 import closure."""

import os
import subprocess
import sys
from pathlib import Path


def test_dictionary_imports_from_the_clean_package_tree():
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(project_root / "src"), str(project_root / "common")))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from ismrm_v2.fit_first.dictionary import frozen_dictionary",
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
