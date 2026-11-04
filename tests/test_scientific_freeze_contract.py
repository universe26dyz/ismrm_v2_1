"""Regression coverage for the active ISMRM-v2.1 scientific freeze."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE_CONTRACT = ROOT / "configs" / "scientific_freeze_contract.json"
FREEZE_CHECKER = ROOT / "scripts" / "verify_scientific_freeze.py"
SUBJECT_VALIDATOR = "ismrm_v2.io.validate_subject_id"


def test_scientific_freeze_checker_passes_for_the_active_tree():
    result = subprocess.run(
        [sys.executable, str(FREEZE_CHECKER), "--root", str(ROOT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "SCIENTIFIC FREEZE CONTRACT: PASS" in result.stdout


def test_active_freeze_contract_is_the_corrected_hhz_q002s640_route():
    contract = json.loads(FREEZE_CONTRACT.read_text(encoding="utf-8"))
    assert contract["models"]["decoder_sha256"] == "338f2c0b8db90851f733ed31caa2c99914605fed3f7215ec2f69f19eba59583b"
    assert contract["models"]["b6_model_sha256"] == "15981fbe93d2c70e14755ea756a02856c9383481e7d92d239619268a628a29b5"
    assert contract["models"]["protocol_sha256"] == "3865b6bde1154750d4ed687456823fafad7b0cad5a61f763decb27dd1a4ed1c7"
    assert contract["proposed"] == {
        "experiment_id": "Q002S640_B6_roi_same_anchor_mse",
        "training_psf_samples": 8,
        "anchor_batch_size": 640,
        "weights_per_anchor": 10,
        "signal_residual_count": 6400,
        "seed": 20260911,
        "stage_a_iterations": 2000,
        "stage_b_iterations": 4000,
        "cosine_loss": False,
        "heteroscedastic_uncertainty": False,
        "amplitude_guidance": False,
    }


def test_submission_cohort_is_reporting_metadata_not_a_runtime_admission_gate():
    contract = json.loads(FREEZE_CONTRACT.read_text(encoding="utf-8"))
    assert contract["active_submission_cohort"] == ["CYJ", "DYL_2", "DYZ", "HHZ", "HJL"]
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    allowed = subprocess.run([sys.executable, "-m", SUBJECT_VALIDATOR, "DYL"], env=environment, capture_output=True, text=True)
    new_subject = subprocess.run([sys.executable, "-m", SUBJECT_VALIDATOR, "NEW_SUBJECT_001"], env=environment, capture_output=True, text=True)
    assert allowed.returncode == 0, allowed.stderr
    assert new_subject.returncode == 0, new_subject.stderr


def test_active_pipeline_cannot_reconnect_slice_standardization():
    active = [
        ROOT / "pipeline" / stage / "run_stage.sh"
        for stage in (
            "01_RAW_DICOM", "02_PREPROCESS_MATLAB", "03_PREPARE_OBSERVATIONS",
            "04_SPARSE_PROFILES", "05_SCMR_V5_FIT_FIRST", "06_PROPOSED",
        )
    ]
    for path in active:
        assert "standardiz" not in path.read_text(encoding="utf-8").lower(), path


def test_final_pose_preserves_l100_full_prepared_root_semantics():
    source = (ROOT / "pipeline/07_FINAL_POSE/internal/hhz_paired_final_pose.sh").read_text(encoding="utf-8")
    assert 'if [ "$CASE" = "L100_fold0" ]; then' in source
    assert 'PREPARED_FOR_CASE=$FIT_BASE/prepared/$SUBJECT' in source
    assert 'PREPARED_FOR_CASE=$CASE_ROOT' in source


def test_stage_09_runs_the_scientific_freeze_checker():
    source = (ROOT / "pipeline/09_INTEGRITY_AND_REPORTING/run_stage.sh").read_text(encoding="utf-8")
    assert "verify_scientific_freeze.py" in source
