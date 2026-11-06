"""CPU-only admission check for the formal v2.2 reliability audit plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ismrm_v2.v2_2.preflight.cross_view_reliability_audit import _sha256_file
from ismrm_v2.sampling.sparse_profile_manifest import validate_sparse_profile_manifest

APPROVED_DECODER_SHA256 = "338f2c0b8db90851f733ed31caa2c99914605fed3f7215ec2f69f19eba59583b"
APPROVED_PROTOCOL_SHA256 = "3865b6bde1154750d4ed687456823fafad7b0cad5a61f763decb27dd1a4ed1c7"


def validate_plan(*, root: str | Path, case_plan: str | Path, decoder: str | Path, protocol: str | Path, output: str | Path) -> Path:
    root_path = Path(root).resolve()
    plan = json.loads(Path(case_plan).read_text(encoding="utf-8"))
    cases = plan.get("cases") if isinstance(plan, dict) else None
    report: dict[str, Any] = {"schema": "ismrm_v2_2/reliability_audit_admission/v1", "status": "PASS_ADMISSION", "cases": []}
    if plan.get("schema") != "ismrm_v2_2/reliability_audit_case_plan/v1" or not isinstance(cases, list) or len(cases) != 8:
        raise ValueError("formal admission requires exactly eight case-plan entries")
    decoder_path, protocol_path = Path(decoder).resolve(), Path(protocol).resolve()
    if _sha256_file(decoder_path) != APPROVED_DECODER_SHA256:
        raise ValueError("approved decoder SHA256 mismatch")
    if _sha256_file(protocol_path) != APPROVED_PROTOCOL_SHA256:
        raise ValueError("approved protocol SHA256 mismatch")
    decoder_checkpoint = torch.load(decoder_path, map_location="cpu", weights_only=False)
    if not isinstance(decoder_checkpoint, dict) or "train_timing9_min_ms" not in decoder_checkpoint or "train_timing9_max_ms" not in decoder_checkpoint:
        raise ValueError("approved decoder lacks train_timing9_min_ms/train_timing9_max_ms")
    timing_lower = torch.as_tensor(decoder_checkpoint["train_timing9_min_ms"], dtype=torch.float64).numpy()
    timing_upper = torch.as_tensor(decoder_checkpoint["train_timing9_max_ms"], dtype=torch.float64).numpy()
    if timing_lower.shape != (9,) or timing_upper.shape != (9,):
        raise ValueError("approved decoder timing domain must have shape [9]")
    seen = set()
    for item in cases:
        subject, planned = item["subject"], item.get("case_id") or f"{item['profile_id']}_fold{item.get('fold_id', 0)}"
        key = (subject, planned)
        if key in seen:
            raise ValueError(f"duplicate planned case: {key}")
        seen.add(key)
        input_dir = (root_path / item["input_dir"].format(**item)).resolve()
        runtime_path = input_dir / "case_runtime.json"
        case_report = {"subject": subject, "planned_case": planned, "input_dir": str(input_dir), "status": "PASS_ADMISSION"}
        try:
            if not runtime_path.is_file():
                raise ValueError("case_runtime.json is required")
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            manifest_path = Path(runtime["sparse_manifest"]).resolve()
            prepared_root = Path(runtime["prepared_root"]).resolve()
            model_path = Path(runtime["model_checkpoint"]).resolve()
            if runtime.get("geometry_checkpoint") != runtime.get("reconstruction_checkpoint") or model_path.name != "model.pt":
                raise ValueError("geometry and reconstruction must use one staged model.pt")
            manifest = validate_sparse_profile_manifest(manifest_path, subject=subject)
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            required = {"model_state", "training_space", "group_axisangle_post_stack_init_physical", "group_resolution_xyz_mm"}
            missing = required.difference(checkpoint) if isinstance(checkpoint, dict) else required
            if missing:
                raise ValueError(f"model.pt lacks required fields: {sorted(missing)}")
            for stack, record in manifest.get("stacks", {}).items():
                for field in ("source_observation", "derived_observation"):
                    value = record.get(field)
                    if isinstance(value, dict) and value.get("path") and not Path(value["path"]).is_file():
                        raise FileNotFoundError(value["path"])
            active_stacks = [stack for stack, value in manifest.get("stacks", {}).items() if value.get("training_stack_present") or value.get("retained_original_group_ids")]
            timing_domain = {"status": "admitted", "min_ms": timing_lower.tolist(), "max_ms": timing_upper.tolist()}
            geometry_group_count = None
            for stack in active_stacks:
                spec = manifest["stacks"][stack]
                label = "source_observation" if runtime.get("input_observation_mode") == "source" else "derived_observation"
                record = spec.get(label) or spec.get("source_observation")
                if not isinstance(record, dict):
                    raise ValueError(f"active stack {stack} lacks resolved observation record")
                observation_path = Path(record["path"]).resolve()
                with np.load(observation_path, allow_pickle=False) as data:
                    timing = np.asarray(data["timing9_ms"], dtype=float)
                    if timing.ndim != 2 or timing.shape[1] != 9:
                        raise ValueError(f"{stack} timing9_ms must have shape [N,9]")
                    bad = (timing < timing_lower[None, :] - 1e-4) | (timing > timing_upper[None, :] + 1e-4)
                    if bad.any():
                        dimensions = [int(index) for index in np.flatnonzero(bad.any(axis=0))]
                        raise ValueError(f"BLOCKED_TIMING_DOMAIN stack={stack} dimensions={dimensions}")
                    if stack == "sax":
                        groups = np.asarray(data["group_idx"], dtype=int)
                        weights = np.asarray(data["weight_idx"], dtype=int)
                        group_count = int(groups.max()) + 1 if groups.size else 0
                        geometry_group_count = group_count
                        for group in np.unique(groups):
                            rows = np.flatnonzero(groups == group)
                            if rows.size != 10 or not np.array_equal(np.sort(weights[rows]), np.arange(10)) or not np.allclose(timing[rows], timing[rows[0]][None, :], rtol=0.0, atol=1e-5):
                                raise ValueError(f"SAX group {int(group)} timing9_ms is not consistent across ten weights")
            checkpoint_group_count = int(torch.as_tensor(checkpoint["group_axisangle_post_stack_init_physical"]).shape[0])
            if geometry_group_count is not None and checkpoint_group_count != geometry_group_count:
                raise ValueError(f"checkpoint/SAX group count mismatch: {checkpoint_group_count} != {geometry_group_count}")
            case_report.update({"resolved_case": manifest.get("case_id", manifest.get("condition", planned)), "prepared_root": str(prepared_root), "active_stacks": active_stacks, "target_stacks": [stack for stack in ("2ch", "4ch") if stack in active_stacks], "model_sha256": _sha256_file(model_path), "manifest_sha256": manifest.get("_manifest_sha256"), "decoder_sha256": _sha256_file(decoder_path), "protocol_sha256": _sha256_file(protocol_path), "training_space_present": True, "timing_domain": timing_domain, "checkpoint_group_count": checkpoint_group_count})
        except Exception as error:
            case_report["status"] = "BLOCKED_INPUT_PROVENANCE"
            case_report["error"] = str(error)
            report["status"] = "BLOCKED_INPUT_PROVENANCE"
        report["cases"].append(case_report)
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--case-plan", required=True); parser.add_argument("--decoder", required=True); parser.add_argument("--protocol", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = validate_plan(root=args.root, case_plan=args.case_plan, decoder=args.decoder, protocol=args.protocol, output=args.output)
    print(path)
    if json.loads(path.read_text())["status"] != "PASS_ADMISSION":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
