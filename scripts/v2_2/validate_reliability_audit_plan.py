"""CPU-only admission check for the formal v2.2 reliability audit plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ismrm_v2.proposed.q002.joint_dataset import JointAnchorDataset
from ismrm_v2.proposed.quantitative_point_dataset import QuantPointDataset
from ismrm_v2.proposed.training.training_space import TrainingSpace
from ismrm_v2.v2_2.preflight.cross_view_reliability_audit import _joint_anchor_order_sha256, _sha256_file, load_geometry_checkpoint
from ismrm_v2.v2_2.preflight.export_timing_only_anchor_parameters import load_frozen_quantitative_state_for_timing_audit
from ismrm_v2.v2_2.preflight.same_world_reliability_audit import _manifest_active_inputs, validate_decoder_timing_domain

APPROVED_DECODER_SHA256 = "338f2c0b8db90851f733ed31caa2c99914605fed3f7215ec2f69f19eba59583b"
APPROVED_PROTOCOL_SHA256 = "3865b6bde1154750d4ed687456823fafad7b0cad5a61f763decb27dd1a4ed1c7"


class AdmissionBlocked(ValueError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def _resolved_cases(plan: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    cases = plan.get("cases") if isinstance(plan, dict) else None
    if not isinstance(cases, list) or len(cases) != 8:
        raise AdmissionBlocked("BLOCKED_PLAN_RESOLUTION", "formal admission requires exactly eight cases")
    if plan.get("schema") == "ismrm_v2_2/reliability_audit_resolved_case_plan/v1":
        required = ("subject", "resolved_case_id", "input_dir", "case_runtime")
        if any(not all(isinstance(item.get(key), str) for key in required) for item in cases):
            raise AdmissionBlocked("BLOCKED_PLAN_RESOLUTION", "resolved case plan lacks authoritative runtime identity")
        return cases
    if plan.get("schema") != "ismrm_v2_2/reliability_audit_case_plan/v1":
        raise AdmissionBlocked("BLOCKED_PLAN_RESOLUTION", "unsupported case-plan schema")
    resolved = []
    for item in cases:
        if not isinstance(item.get("case_id"), str):
            raise AdmissionBlocked("BLOCKED_PLAN_RESOLUTION", "profile selectors require resolved_case_plan.json")
        case_id = item["case_id"]
        try:
            input_dir = str((root / item["input_dir"].format(**(item | {"case_id": case_id}))).resolve())
        except (KeyError, ValueError) as error:
            raise AdmissionBlocked("BLOCKED_PLAN_RESOLUTION", f"invalid exact-case input_dir: {error}") from error
        resolved.append({**item, "resolved_case_id": case_id, "input_dir": input_dir, "case_runtime": str(Path(input_dir) / "case_runtime.json")})
    return resolved


def _active_timing_consistency(inputs: list[Path], active_stacks: list[str]) -> None:
    for path in inputs:
        stack = path.parent.name
        if stack not in active_stacks:
            continue
        with np.load(path, allow_pickle=False) as data:
            required = {"timing9_ms", "group_idx", "weight_idx"}
            missing = required.difference(data.files)
            if missing:
                raise AdmissionBlocked("BLOCKED_TIMING_CONSISTENCY", f"{stack} observation lacks {sorted(missing)}")
            timing = np.asarray(data["timing9_ms"], dtype=np.float64)
            groups = np.asarray(data["group_idx"], dtype=np.int64)
            weights = np.asarray(data["weight_idx"], dtype=np.int64)
            if timing.ndim != 2 or timing.shape[1] != 9 or groups.shape != (timing.shape[0],) or weights.shape != groups.shape:
                raise AdmissionBlocked("BLOCKED_TIMING_CONSISTENCY", f"{stack} timing/group arrays have inconsistent shapes")
            for group in np.unique(groups):
                rows = np.flatnonzero(groups == group)
                if rows.size == 10 and np.array_equal(np.sort(weights[rows]), np.arange(10)) and not np.allclose(timing[rows], timing[rows[0]][None, :], rtol=0.0, atol=1e-5):
                    raise AdmissionBlocked("BLOCKED_TIMING_CONSISTENCY", f"{stack} group {int(group)} timing9_ms differs across ten weights")


def validate_plan(*, root: str | Path, case_plan: str | Path, decoder: str | Path, protocol: str | Path, output: str | Path, b6_model: str | Path | None = None) -> Path:
    root_path = Path(root).resolve()
    plan = json.loads(Path(case_plan).read_text(encoding="utf-8"))
    cases = _resolved_cases(plan, root_path)
    report: dict[str, Any] = {"schema": "ismrm_v2_2/reliability_audit_admission/v2", "status": "PASS_ADMISSION", "resolved_case_plan": str(Path(case_plan).resolve()), "cases": []}
    decoder_path, protocol_path = Path(decoder).resolve(), Path(protocol).resolve()
    if not decoder_path.is_file() or _sha256_file(decoder_path) != APPROVED_DECODER_SHA256:
        raise AdmissionBlocked("BLOCKED_ASSET_SHA", "approved decoder SHA256 mismatch")
    if not protocol_path.is_file() or _sha256_file(protocol_path) != APPROVED_PROTOCOL_SHA256:
        raise AdmissionBlocked("BLOCKED_ASSET_SHA", "approved protocol SHA256 mismatch")
    seen: set[tuple[str, str]] = set()
    for item in cases:
        subject, planned = item["subject"], item["resolved_case_id"]
        key = (subject, planned)
        case_report: dict[str, Any] = {"subject": subject, "resolved_case": planned, "input_dir": str(Path(item["input_dir"]).resolve()), "status": "PASS_ADMISSION"}
        try:
            if key in seen:
                raise AdmissionBlocked("BLOCKED_PLAN_RESOLUTION", f"duplicate resolved case: {key}")
            seen.add(key)
            runtime_path = Path(item["case_runtime"]).resolve()
            if not runtime_path.is_file():
                raise AdmissionBlocked("BLOCKED_INPUT_PROVENANCE", "case_runtime.json is required")
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime_case = str(runtime.get("case_id"))
            if runtime_case != planned:
                raise AdmissionBlocked("BLOCKED_PLAN_RESOLUTION", f"runtime case_id {runtime_case!r} != resolved plan {planned!r}")
            if runtime.get("geometry_checkpoint") != runtime.get("reconstruction_checkpoint") or not str(runtime.get("geometry_checkpoint", "")).endswith("/model.pt"):
                raise AdmissionBlocked("BLOCKED_CHECKPOINT_COMPATIBILITY", "geometry and reconstruction must use one staged model.pt")
            model_path = Path(runtime["model_checkpoint"]).resolve()
            if not model_path.is_file():
                raise AdmissionBlocked("BLOCKED_INPUT_PROVENANCE", f"missing model.pt: {model_path}")
            inputs, manifest_inputs = _manifest_active_inputs(runtime["sparse_manifest"], subject=subject, case_id=runtime_case, prepared_root=runtime["prepared_root"])
            active_stacks = list(manifest_inputs.get("active_stacks", []))
            scalar = QuantPointDataset(inputs, device=torch.device("cpu"))
            joint = JointAnchorDataset(inputs, scalar)
            try:
                geometry, geometry_provenance = load_geometry_checkpoint(model_path, scalar.group_axisangle_init, scalar.group_resolution_xyz_mm)
            except (FileNotFoundError, ValueError) as error:
                raise AdmissionBlocked("BLOCKED_CASE_GEOMETRY", str(error)) from error
            try:
                _, _, checkpoint_provenance = load_frozen_quantitative_state_for_timing_audit(model_path, device=torch.device("cpu"), fallback_b6_model=b6_model)
                current_space = TrainingSpace.from_dataset(scalar, float(checkpoint_provenance["training_space_spatial_scaling"]), group_axisangle_init_physical=geometry, bbox_margin_mm=float(checkpoint_provenance["training_space_bbox_margin_mm"]))
                load_frozen_quantitative_state_for_timing_audit(model_path, device=torch.device("cpu"), fallback_b6_model=b6_model, current_space=current_space)
            except (RuntimeError, ValueError) as error:
                status = "BLOCKED_TRAINING_SPACE" if "training_space" in str(error) or "intensity" in str(error) else "BLOCKED_CHECKPOINT_COMPATIBILITY"
                raise AdmissionBlocked(status, str(error)) from error
            try:
                timing_domain = validate_decoder_timing_domain(decoder_path, scalar.timing.detach().cpu().numpy())
            except (FileNotFoundError, ValueError) as error:
                raise AdmissionBlocked("BLOCKED_TIMING_DOMAIN", str(error)) from error
            _active_timing_consistency(inputs, active_stacks)
            case_report.update({"active_stacks": active_stacks, "target_stacks": [stack for stack in ("2ch", "4ch") if stack in active_stacks], "anchor_count": int(joint.xyz.shape[0]), "anchor_order_sha256": _joint_anchor_order_sha256(joint), "model_sha256": _sha256_file(model_path), "manifest_sha256": manifest_inputs.get("sha256"), "decoder_sha256": _sha256_file(decoder_path), "protocol_sha256": _sha256_file(protocol_path), "geometry_provenance": geometry_provenance, "training_space_provenance": checkpoint_provenance, "timing_domain": timing_domain, "training_space_admitted": True})
        except AdmissionBlocked as error:
            case_report["status"] = error.status
            case_report["error"] = str(error)
            report["status"] = error.status if report["status"] == "PASS_ADMISSION" else report["status"]
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
            case_report["status"] = "BLOCKED_INPUT_PROVENANCE"
            case_report["error"] = f"{type(error).__name__}: {error}"
            report["status"] = "BLOCKED_INPUT_PROVENANCE"
        report["cases"].append(case_report)
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--case-plan", required=True); parser.add_argument("--decoder", required=True); parser.add_argument("--protocol", required=True); parser.add_argument("--b6-model"); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        path = validate_plan(root=args.root, case_plan=args.case_plan, decoder=args.decoder, protocol=args.protocol, b6_model=args.b6_model, output=args.output)
    except AdmissionBlocked as error:
        path = Path(args.output).resolve(); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps({"schema": "ismrm_v2_2/reliability_audit_admission/v2", "status": error.status, "error": str(error), "cases": []}, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(2)
    print(path)
    if json.loads(path.read_text())["status"] != "PASS_ADMISSION":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
