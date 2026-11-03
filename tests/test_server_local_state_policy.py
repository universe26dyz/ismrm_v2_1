from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_server_local_state_policy_and_ignore_rules() -> None:
    result = subprocess.run([sys.executable, "scripts/verify_server_local_state_policy.py"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    policy = json.loads((ROOT / "configs/server_local_state_policy.json").read_text(encoding="utf-8"))
    assert policy["schema_version"] == "server-local-state-policy/v1"
    for path in ("subjects/CYJ/example.tmp", "runs/CYJ/example.tmp", "runtime/model_assets/example.pt", "runtime/model_assets/example.pth"):
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0
