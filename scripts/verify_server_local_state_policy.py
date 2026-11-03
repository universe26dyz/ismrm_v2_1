#!/usr/bin/env python3
"""Static safety checks for server-local state; never accesses a server."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBES = ("subjects/CYJ/example.tmp", "runs/CYJ/example.tmp", "runtime/model_assets/example.pt", "runtime/model_assets/example.pth")


def _tracked(root: Path) -> list[str]:
    return subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()


def main(root: Path = ROOT) -> None:
    policy = json.loads((root / "configs/server_local_state_policy.json").read_text(encoding="utf-8"))
    if policy.get("schema_version") != "server-local-state-policy/v1":
        raise SystemExit("invalid server-local-state policy schema")
    protected = tuple(item["path"] for item in policy["protected_paths"])
    if protected != ("subjects/", "runs/", "runtime/model_assets/"):
        raise SystemExit("protected path policy is incomplete")
    bad = [path for path in _tracked(root) if path.startswith(("subjects/", "runs/")) or (path.startswith("runtime/model_assets/") and path.endswith((".pt", ".pth")))]
    if bad:
        raise SystemExit(f"protected server-local state is tracked: {bad}")
    for probe in PROBES:
        result = subprocess.run(["git", "check-ignore", "-q", probe], cwd=root)
        if result.returncode != 0:
            raise SystemExit(f"protected path is not ignored: {probe}")
    dangerous = ("git" + " clean -fd", "git" + " clean -fdx", "rm" + " -rf /data/dengyz/code/ismrm_v2_1")
    unsafe: list[str] = []
    for path in _tracked(root):
        if Path(path).suffix not in {".py", ".sh"}:
            continue
        if path == "scripts/verify_server_local_state_policy.py":
            continue
        text = (root / path).read_text(encoding="utf-8", errors="ignore")
        if any(token in text for token in dangerous):
            unsafe.append(path)
    if unsafe:
        raise SystemExit(f"dangerous tracked deployment helper: {unsafe}")
    print("SERVER LOCAL STATE POLICY: PASS")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT)
