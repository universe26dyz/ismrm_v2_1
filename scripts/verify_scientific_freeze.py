#!/usr/bin/env python3
"""Verify the local, data-free scientific freeze contract for ISMRM-v2.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _require_contains(path: Path, text: str, label: str) -> None:
    if text not in path.read_text(encoding="utf-8"):
        raise ValueError(f"{label}: missing {text!r} in {path}")


def _require_absent(path: Path, text: str, label: str) -> None:
    if text in path.read_text(encoding="utf-8").lower():
        raise ValueError(f"{label}: forbidden active dependency {text!r} in {path}")


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    contract = _read_json(root / "configs/scientific_freeze_contract.json")
    protocol = yaml.safe_load((root / "configs/protocol_scmr_v5_10_100.yaml").read_text(encoding="utf-8"))
    runtime = _read_json(root / "runtime/runtime_contract.json")
    manifest = _read_json(root / "configs/models_manifest.json")
    expected_protocol = contract["protocol"]

    _require_equal(protocol["tr_ms"], expected_protocol["tr_ms"], "protocol TR")
    _require_equal(protocol["vps"], expected_protocol["vps"], "protocol VPS")
    _require_equal(protocol["inversion_times_ms"], expected_protocol["ti_ms"], "protocol TI")
    _require_equal(protocol["t2prep_ms"], expected_protocol["t2prep_ms"], "protocol T2prep")
    _require_equal(protocol["flip_angle_degrees"], expected_protocol["fa_deg"], "protocol FA")
    _require_equal(protocol["ramp_up_pulses"], expected_protocol["ramp_up_pulses"], "protocol ramp-up")
    _require_equal(protocol["num_images"], expected_protocol["num_images"], "protocol image count")

    models = contract["models"]
    _require_equal(runtime["decoder_sha256"], models["decoder_sha256"], "runtime decoder SHA")
    _require_equal(runtime["b6_model_sha256"], models["b6_model_sha256"], "runtime B6 SHA")
    _require_equal(runtime["protocol_sha256"], models["protocol_sha256"], "runtime protocol SHA")
    _require_equal(manifest["decoder"]["sha256"], models["decoder_sha256"], "manifest decoder SHA")
    _require_equal(manifest["b6_model"]["sha256"], models["b6_model_sha256"], "manifest B6 SHA")
    _require_equal(manifest["protocol"]["sha256"], models["protocol_sha256"], "manifest protocol SHA")
    _require_equal(runtime["forbidden_decoder_ti_ms"], models["forbidden_decoder_ti_ms"], "forbidden decoder TI")

    proposed = root / "common/paper_experiments/joint_vs_2dfit/phase2_clean/proposed.py"
    q002_contracts = root / "common/quality_experiments/Q002_B6_roi_same_anchor_mse/contracts.py"
    q002_training = root / "common/quality_experiments/Q002_B6_roi_same_anchor_mse/training.py"
    stage06 = root / "pipeline/06_PROPOSED/run_stage.sh"
    _require_contains(proposed, "SCMR_V5_CORRECTED_DECODER_SHA256", "corrected decoder default")
    _require_contains(root / "common/paper_experiments/joint_vs_2dfit/phase2_clean/contracts.py", models["decoder_sha256"], "corrected decoder provenance")
    _require_contains(stage06, models["decoder_sha256"], "corrected decoder binding")
    for flag, value in (("--expected-b6-model-sha256", models["b6_model_sha256"]), ("--expected-protocol-sha256", models["protocol_sha256"])):
        _require_contains(stage06, flag, "Proposed runtime validation")
        _require_contains(stage06, value, "Proposed runtime validation")
    for flag, value in (("--experiment-id", contract["proposed"]["experiment_id"]), ("--anchor-batch-size", "640")):
        _require_contains(proposed, flag, "Proposed route binding")
        _require_contains(proposed, value, "Proposed route binding")
    _require_contains(q002_contracts, '"shared_training_psf_samples": 8', "Q002S640 K8 sampling")
    for text in ('"stage_a_iterations": 2000', '"stage_b_iterations": 4000', '"signal_residual_count": 6400', '"weights_per_anchor": 10'):
        _require_contains(q002_training, text, "Q002 formal training controls")
    _require_absent(proposed, "fingerprint_cosine", "active Proposed loss")

    resolved = yaml.safe_load((root / "common/mlp/configs/recon_server_train_scmr_v5.yaml").read_text(encoding="utf-8"))
    for key, expected in contract["inr"].items():
        if key in resolved["inr"]:
            _require_equal(resolved["inr"][key], expected, f"resolved INR {key}")
    _require_equal(resolved["variance"], {"enabled": False, "pixel": False, "slice": False, "method": "experimental_project_heteroscedastic"}, "formal variance status")
    _require_equal(resolved["spatial_regularization"]["amplitude_guidance"]["enabled"], False, "amplitude guidance")
    _require_equal(resolved["training"]["learning_rates"]["encoding"], 0.01, "encoding LR")
    _require_equal(resolved["training"]["learning_rates"]["network"], 0.001, "network LR")
    _require_equal(resolved["training"]["learning_rates"]["rigid"], 0.0001, "rigid LR")
    _require_equal(resolved["training"]["learning_rates"]["weight_decay"], 0.01, "weight decay")
    _require_equal(resolved["training"]["scheduler_milestones"], [0.5, 0.8], "scheduler milestones")
    _require_equal(resolved["training"]["scheduler_gamma"], 0.5, "scheduler gamma")
    _require_equal(resolved["spatial_regularization"]["n_points"], 256, "regularization points")

    inr_source = root / "common/mlp/modules/module_04_quantitative_inr/quantitative_inr.py"
    decoder_source = root / "common/mlp/modules/module_05_signal_decoder/frozen_decoder.py"
    for text in ("t2_ms = (5.0 + 195.0", "2500.0 - t2_ms", "b1 = (0.1 + 1.1", "F.softplus(raw[..., 3])"):
        _require_contains(inr_source, text, "quantitative INR bounds")
    for text in ("parameter.requires_grad_(False)", "self.mlp.eval()", "normalize is not True"):
        _require_contains(decoder_source, text, "frozen decoder contract")
    decoder_forward = decoder_source.read_text(encoding="utf-8").split("    def forward", 1)[1]
    if "torch.no_grad" in decoder_forward:
        raise ValueError("frozen decoder forward must preserve gradients to tissue inputs")

    baseline = contract["baseline_fit_first"]
    baseline_source = root / "common/paper_experiments/joint_vs_2dfit/phase2_clean/baseline.py"
    fit_first_source = root / "common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first.py"
    native_cache = root / "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh"
    for text in ("--prebuilt-native-maps-root", "--reuse-native-maps-root", "--require-scmr-v5-native-maps"):
        _require_contains(native_cache, text, "fit-first active route")
    _require_absent(native_cache, "standardiz", "fit-first active route")
    _require_contains(baseline_source, "SCMR one-stage", "fit-first baseline route")
    for option, value in (("--thicknesses", "['8']"), ("--registration", "'stack'"), ("--output-intensity-mean", "'0'"), ("--output-resolution", "'1.0'"), ("--n-iter", "'10000'"), ("--n-samples", "'256'"), ("--n-inference-samples", "'128'"), ("--pixel-variance-weight", "'1'")):
        _require_contains(fit_first_source, option, "fit-first command")
        _require_contains(fit_first_source, value, "fit-first command")
    _require_contains(fit_first_source, "'--no-slice-scale'", "fit-first no-slice-scale")
    _require_contains(fit_first_source, "QC score threshold 0.90", "fit-first dictionary QC")
    if baseline["matching"] != protocol["matching"] or baseline["interpolation_refinement"] != protocol["interpolation_refinement"]:
        raise ValueError("fit-first matching contract disagrees with active protocol")

    profile_root = root / "common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles"
    for profile_name in ("SAX100_2CH0_4CH0", "SAX100_2CH0_4CH100", "SAX100_2CH100_4CH0", "S2CH100_4CH50", "S2CH50_4CH100"):
        profile = _read_json(profile_root / f"{profile_name}.json")
        _require_equal(profile["stacks"]["sax"]["mode"], "all", f"{profile_name} SAX retention")
    sparse_manifest = root / "common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse_profile_manifest.py"
    _require_contains(sparse_manifest, 'if stack == "sax"', "L100/L50/L25 SAX retention")
    _require_contains(sparse_manifest, "retained and held-out groups must form an exhaustive disjoint source partition", "sparse physical selection")

    active_stages = ("01_RAW_DICOM", "02_PREPROCESS_MATLAB", "03_PREPARE_OBSERVATIONS", "04_SPARSE_PROFILES", "05_SCMR_V5_FIT_FIRST", "06_PROPOSED")
    for stage in active_stages:
        script = root / "pipeline" / stage / "run_stage.sh"
        _require_absent(script, "standardiz", "active sparse route")
        _require_contains(script, "validate_active_subject.py", "active cohort admission")
    final_pose = root / "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh"
    _require_contains(final_pose, 'PREPARED_FOR_CASE=$FIT_BASE/prepared/$SUBJECT', "L100 final-pose input")
    _require_contains(final_pose, 'PREPARED_FOR_CASE=$CASE_ROOT', "sparse final-pose input")
    stage09 = root / "pipeline/09_INTEGRITY_AND_REPORTING/run_stage.sh"
    _require_contains(stage09, "verify_scientific_freeze.py", "Stage 09 scientific freeze check")

    return {"authority": contract["authority"], "proposed": contract["proposed"], "cohort": contract["active_submission_cohort"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    summary = verify(args.root)
    print("SCIENTIFIC FREEZE CONTRACT: PASS")
    print(f"authority={summary['authority']['experiment_id']} K={summary['proposed']['training_psf_samples']} anchors={summary['proposed']['anchor_batch_size']} weights={summary['proposed']['weights_per_anchor']}")
    print(f"stages=A{summary['proposed']['stage_a_iterations']}/B{summary['proposed']['stage_b_iterations']} cohort={','.join(summary['cohort'])}")


if __name__ == "__main__":
    main()
