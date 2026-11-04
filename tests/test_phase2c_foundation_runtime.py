"""Phase 2C contracts for moved foundation runtime modules."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
COMMON = ROOT / "common"


def _module(name: str):
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(COMMON))
    return importlib.import_module(name)


def _write_observations(path: Path, group_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = np.repeat(np.arange(group_count, dtype=np.int64), 10)
    weights = np.tile(np.arange(10, dtype=np.int64), group_count)
    rows = len(groups)
    affine = np.repeat(np.eye(4, dtype=np.float64)[None], rows, axis=0)
    affine[:, :3, 0] = [1.0, 0.0, 0.0]
    affine[:, :3, 1] = [0.0, 1.0, 0.0]
    affine[:, :3, 2] = [0.0, 0.0, 5.0]
    affine[:, 2, 3] = -groups.astype(float)
    np.savez_compressed(
        path,
        images=np.ones((rows, 2, 2), dtype=np.float32),
        masks=np.ones((rows, 2, 2), dtype=bool),
        group_idx=groups,
        weight_idx=weights,
        stack_idx=np.zeros(rows, dtype=np.int64),
        acquisition_time_ms=np.zeros(rows, dtype=np.float64),
        timing9_ms=np.repeat(np.arange(9, dtype=np.float64)[None], rows, axis=0),
        affine_lps_rc=affine,
        pixel_spacing_rc_mm=np.repeat([[1.0, 1.0]], rows, axis=0),
        slice_thickness_mm=np.full(rows, 5.0),
        tr_ms=np.full(rows, 2.61),
        vps=np.full(rows, 87, dtype=np.int64),
    )


def _audit(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        groups = np.unique(data["group_idx"])
        positions = [float(data["affine_lps_rc"][np.flatnonzero(data["group_idx"] == group)[0], 2, 3]) for group in groups]
    gap = abs(positions[1] - positions[0]) if len(positions) > 1 else 0.0
    return {"regular_gap": True, "gap_mm": gap, "group_count": len(groups)}


def test_phase2c_clean_paths_are_the_only_foundation_authority() -> None:
    moved = {
        "common/trad/scripts/validate_transferred_preprocessed.py": "src/ismrm_v2/preprocessing/validation.py",
        "common/trad/scripts/prepare_all_observations.py": "src/ismrm_v2/observations/prepare_all.py",
        "common/trad/modules/module_02_data_bridge/build_manifest.py": "src/ismrm_v2/observations/build_manifest.py",
        "common/trad/modules/module_02_data_bridge/mat_v73.py": "src/ismrm_v2/observations/mat_v73.py",
        "common/trad/modules/module_02_data_bridge/timing.py": "src/ismrm_v2/observations/timing.py",
        "common/trad/modules/module_02_data_bridge/prepare_observations.py": "src/ismrm_v2/observations/prepare_observations.py",
        "common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse.py": "src/ismrm_v2/sampling/sparse.py",
        "common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse_profile_manifest.py": "src/ismrm_v2/sampling/sparse_profile_manifest.py",
    }
    for old, new in moved.items():
        assert not (ROOT / old).exists()
        assert (ROOT / new).is_file()
    assert (ROOT / "src/ismrm_v2/observations/geometry.py").is_file()


def test_timing_contract_remains_hhz_ten_heartbeat_physics() -> None:
    timing = _module("ismrm_v2.observations.timing")
    result = timing.compute_duration_before_acq(
        np.arange(10, dtype=np.float64) * 100.0,
        tr_ms=4.0,
        n_ex=5,
        ti_ms=np.array([50.0, 150.0]),
        t2prep_ms=np.array([35.0, 45.0, 55.0]),
    )
    np.testing.assert_allclose(result, [0.0, 80.0, 80.0, 80.0, -70.0, 80.0, 80.0, 45.0, 35.0, 25.0])


def test_sparse_l100_and_l50_keep_sax_and_partition_long_axis_groups(tmp_path: Path) -> None:
    sparse = _module("ismrm_v2.sampling.sparse")
    validator = _module("ismrm_v2.sampling.sparse_profile_manifest")
    contracts = importlib.import_module("ismrm_v2.fit_first.contracts")
    source = tmp_path / "prepared"
    for stack in ("sax", "2ch", "4ch"):
        _write_observations(source / "FUTURE" / stack / "observations.npz", group_count=4)
    output = tmp_path / "sparse"
    l100 = sparse.build_sparse_condition(source, "FUTURE", "L100", 0, output, git_sha="a" * 40)
    l100_manifest = json.loads(Path(str(l100["manifest_path"])).read_text(encoding="utf-8"))
    assert l100_manifest["stacks"]["sax"]["retained_original_group_ids"] == [0, 1, 2, 3]
    assert l100_manifest["stacks"]["2ch"]["retained_original_group_ids"] == [0, 1, 2, 3]
    l50 = sparse.build_sparse_condition(source, "FUTURE", "L50", 0, tmp_path / "sparse_l50", git_sha="a" * 40, audit_stack=_audit)
    l50_manifest = json.loads(Path(str(l50["manifest_path"])).read_text(encoding="utf-8"))
    for stack in ("sax", "2ch", "4ch"):
        retained = l50_manifest["stacks"][stack]["retained_original_group_ids"]
        heldout = l50_manifest["stacks"][stack]["heldout_original_group_ids"]
        assert retained == ([0, 1, 2, 3] if stack == "sax" else [0, 2])
        assert set(retained).isdisjoint(heldout)
        assert set(retained).union(heldout) == {0, 1, 2, 3}
    manifest = Path(str(l50["manifest_path"]))
    validated = validator.validate_sparse_profile_manifest(
        manifest,
        subject="FUTURE",
        strict=True,
        expected_generator_git_sha="a" * 40,
        expected_manifest_sha256=contracts.sha256(manifest),
    )
    assert validated["case_id"] == "L50_fold0"


def test_stage_callers_and_q002_use_clean_modules_only() -> None:
    expected_imports = {
        "src/ismrm_v2/proposed/q002/contracts.py": "from ismrm_v2.sampling.sparse_profile_manifest import LEGACY_SCHEMA_VERSION, SCHEMA_VERSION",
        "src/ismrm_v2/proposed/q002/run_q002_reconstruction.py": "from ismrm_v2.sampling.sparse_profile_manifest import validate_sparse_profile_manifest",
        "src/ismrm_v2/preprocessing/validation.py": "from ismrm_v2.observations.mat_v73 import",
        "src/ismrm_v2/observations/prepare_all.py": "from ismrm_v2.observations.prepare_observations import",
    }
    for path, expected in expected_imports.items():
        assert expected in (ROOT / path).read_text(encoding="utf-8")
