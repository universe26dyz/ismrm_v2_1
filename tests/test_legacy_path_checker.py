from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/verify_no_legacy_code_path.py"
LEGACY = "/data/dengyz/code/Code_10w"


@pytest.fixture()
def checker():
    spec = importlib.util.spec_from_file_location("legacy_checker", CHECKER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tracked_tree(tmp_path: Path, relative: str, content: str) -> Path:
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", relative], cwd=tmp_path, check=True)
    return tmp_path


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        ("scripts/example.py", f'ROOT = "{LEGACY}"\n'),
        ("pipeline/example/run_stage.sh", f'ROOT="{LEGACY}"\n'),
        ("configs/example.json", '{"root": "' + LEGACY + '"}\n'),
    ],
)
def test_runtime_and_config_surfaces_fail_closed(checker, tmp_path: Path, relative: str, content: str) -> None:
    root = _tracked_tree(tmp_path, relative, content)
    assert checker.legacy_hits(root) == [relative]


def test_documented_historical_path_is_not_runtime_violation(checker, tmp_path: Path) -> None:
    root = _tracked_tree(tmp_path, "docs/history.md", f"Historical root: {LEGACY}\n")
    assert checker.legacy_hits(root) == []
