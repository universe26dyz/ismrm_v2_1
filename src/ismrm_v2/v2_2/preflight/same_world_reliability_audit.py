"""Same-world-coordinate 10-D reliability audit, deliberately detached from training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cross_view_reliability_audit import (
    OBSERVATION_STACKS,
    _dataset_arrays,
    _git_provenance,
    _load_timing_only_parameters,
    _joint_anchor_order_sha256,
    resolve_preflight_sparse_manifest,
    _sha256_file,
    load_geometry_checkpoint,
    match_lax_to_sax,
    nearest_neighbor_backend,
    timing_statistics,
)
from .discrepancy import (
    JointRobustNormalization,
    fit_joint_robust_normalization,
    fit_case_joint_robust_normalization,
    gate_dry_run,
    normalized_l1_discrepancy,
    raw_cosine_discrepancy,
)
from .reference_sampling import RegularGridReference, build_regular_grid_reference, build_slice_stack_reference, replace_slice_values, sample_regular_grid_trilinear, sample_slice_stack


def _summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {"count": int(values.size), "valid_count": int(finite.size), "invalid_count": int(values.size - finite.size), "mean": float(finite.mean()) if finite.size else None, "std": float(finite.std()) if finite.size else None, "median": float(np.quantile(finite, .5)) if finite.size else None, "p90": float(np.quantile(finite, .9)) if finite.size else None, "p95": float(np.quantile(finite, .95)) if finite.size else None, "p99": float(np.quantile(finite, .99)) if finite.size else None}


def _resolve_case_manifest(path: str | Path | None, *, subject: str, case_id: str, observation_paths: dict[str, Path]) -> dict[str, Any]:
    """Validate L100 with the existing strict admission, other profiles read-only."""
    if path is None:
        return {"sparse_manifest_status": "not_supplied", "requested_case": case_id, "manifest_resolved_case": None}
    from ismrm_v2.sampling.sparse_profile_manifest import validate_sparse_profile_manifest
    manifest = validate_sparse_profile_manifest(path, subject=subject, strict=False)
    if str(case_id).upper().startswith("L100"):
        validated = resolve_preflight_sparse_manifest(path, subject=subject, requested_case=case_id, observation_paths=observation_paths)
    else:
        validated = {"sparse_manifest_status": "validated"}
    resolved = str(manifest.get("case_id") or manifest.get("condition") or manifest.get("profile_id"))
    if resolved != case_id and not resolved.startswith(f"{case_id}_") and not case_id.startswith(resolved):
        raise ValueError(f"sparse manifest case {resolved!r} disagrees with requested case {case_id!r}")
    stacks = manifest.get("stacks", {})
    active = []
    records = {}
    for stack in OBSERVATION_STACKS:
        spec = stacks.get(stack, {})
        present = bool(spec.get("training_stack_present")) if manifest.get("schema_version") == "phase4r_sparse_profile_v2" else bool(spec.get("retained_original_group_ids"))
        if present:
            label = "source_observation" if validated.get("input_observation_mode") == "source" else "derived_observation"
            record = spec.get(label) or spec.get("source_observation")
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise ValueError(f"manifest active stack {stack} lacks {label}")
            active.append(stack); records[stack] = {"mode": label.removesuffix("_observation"), "path": str(Path(record["path"]).resolve()), "sha256": record.get("sha256"), "retained_original_group_ids": spec.get("retained_original_group_ids", []), "derived_to_original_group_idx": spec.get("derived_to_original_group_idx")}
    return {**validated, "sparse_manifest_status": "validated", "requested_case": case_id, "manifest_resolved_case": resolved, "resolved_case": resolved, "profile_id": manifest.get("profile_id"), "fold_id": manifest.get("fold_id"), "fold_count": manifest.get("fold_count"), "schema_version": manifest.get("schema_version"), "active_stacks": active, "observation_records": records, "path": str(manifest.get("_manifest_path", Path(path).resolve())), "sha256": str(manifest.get("_manifest_sha256", _sha256_file(Path(path))))}


def _manifest_active_inputs(manifest_path: str | Path, *, subject: str, case_id: str, prepared_root: str | Path) -> tuple[list[Path], dict[str, Any]]:
    """Resolve exact manifest-declared observations; never fall back to full data."""
    from ismrm_v2.sampling.sparse_profile_manifest import validate_sparse_profile_manifest
    manifest = validate_sparse_profile_manifest(manifest_path, subject=subject, strict=False)
    schema = manifest.get("schema_version")
    if schema not in {"phase4r_sparse_v1", "phase4r_sparse_profile_v2"}:
        raise ValueError("unsupported sparse manifest schema")
    resolved_case = str(manifest.get("case_id") or manifest.get("condition") or manifest.get("profile_id"))
    if resolved_case != case_id and not resolved_case.startswith(f"{case_id}_") and not case_id.startswith(resolved_case):
        raise ValueError(f"sparse manifest case {resolved_case!r} disagrees with requested case {case_id!r}")
    condition = str(manifest.get("condition") or manifest.get("case_id") or case_id).upper()
    mode = "source" if schema == "phase4r_sparse_v1" and condition.startswith("L100") else "derived"
    if str(case_id).upper().startswith("L100"):
        source_paths = {stack: Path(manifest["stacks"][stack]["source_observation"]["path"]).resolve() for stack in OBSERVATION_STACKS}
        requested = {stack: Path(value).resolve() for stack, value in zip(OBSERVATION_STACKS, (Path(prepared_root) / stack / "observations.npz" for stack in OBSERVATION_STACKS), strict=True) if value.is_file()}
        mode = "source" if requested and all(requested.get(stack) == source_paths[stack] for stack in requested) else "derived"
    paths = []
    records = {}
    for stack in OBSERVATION_STACKS:
        spec = manifest.get("stacks", {}).get(stack, {})
        present = bool(spec.get("training_stack_present")) if schema == "phase4r_sparse_profile_v2" else bool(spec.get("retained_original_group_ids"))
        if not present:
            continue
        label = "source_observation" if mode == "source" else "derived_observation"
        record = spec.get(label)
        if not isinstance(record, dict):
            raise ValueError(f"active manifest stack {stack} lacks {label}")
        path = Path(record["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"manifest active observation is missing: {path}")
        if record.get("sha256") and _sha256_file(path) != record["sha256"]:
            raise ValueError(f"manifest observation SHA256 mismatch: {path}")
        paths.append(path); records[stack] = {"mode": mode, "path": str(path), "sha256": record.get("sha256", _sha256_file(path)), "retained_original_group_ids": spec.get("retained_original_group_ids", []), "derived_to_original_group_idx": spec.get("derived_to_original_group_idx")}
    if not paths:
        raise ValueError("sparse manifest has no active stacks")
    declared_root = Path(manifest.get("prepared_root", "")).resolve()
    if mode == "derived" and Path(prepared_root).resolve() != declared_root:
        raise ValueError("--prepared-root disagrees with manifest prepared_root")
    return paths, {"sparse_manifest_status": "validated", "requested_case": case_id, "input_observation_mode": mode, "active_stacks": [stack for stack in OBSERVATION_STACKS if stack in records], "observation_records": records, "profile_id": manifest.get("profile_id"), "resolved_case": resolved_case, "schema_version": schema, "fold_id": manifest.get("fold_id"), "fold_count": manifest.get("fold_count"), "path": str(manifest.get("_manifest_path", Path(manifest_path).resolve())), "sha256": manifest.get("_manifest_sha256")}


def _anchor_identity_hash(arrays: dict[str, np.ndarray], indices: np.ndarray) -> str:
    identity = np.column_stack((arrays["stack_idx"][indices], arrays["group_idx"][indices], arrays["row_col"][indices])).astype("<i8", copy=False)
    return hashlib.sha256(identity.tobytes()).hexdigest()


def _sample_timing_grid(points: np.ndarray, timing: np.ndarray, *, tolerance: float = 1e-5) -> RegularGridReference:
    return build_regular_grid_reference(points, np.column_stack((timing, np.zeros((timing.shape[0], 1)))).reshape(timing.shape[0], 10), tolerance=tolerance)


def _timing_only_pair(
    decoder: torch.nn.Module | None,
    parameters: dict[str, Any] | None,
    arrays: dict[str, np.ndarray],
    target_indices: np.ndarray,
    timing_reference: np.ndarray,
    reference_valid: np.ndarray,
    normalization: JointRobustNormalization | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    if decoder is None:
        return None, None, {"status": "not_run", "reason": "no decoder was supplied"}
    if parameters is None:
        return None, None, {"status": "not_run", "reason": "no anchor-aligned T1/T2/B1/amplitude parameters were supplied"}
    missing = {"t1_ms", "t2_ms", "b1", "amplitude"}.difference(parameters)
    if missing:
        return None, None, {"status": "not_run", "reason": f"parameter source lacks {sorted(missing)}"}
    count = target_indices.size
    if timing_reference.shape != (count, 9):
        raise ValueError("same-world timing reference must be [N,9].")
    if not reference_valid.any():
        return np.full(count, np.nan), np.full(count, np.nan), {"status": "not_run", "reason": "no valid same-world reference timing samples"}
    parameter_values = []
    device_parameter = next(decoder.parameters(), None)
    device = device_parameter.device if device_parameter is not None else torch.device("cpu")
    for name in ("t1_ms", "t2_ms", "b1", "amplitude"):
        value = np.asarray(parameters[name])
        if value.shape != (arrays["physical_xyz"].shape[0],):
            raise ValueError(f"timing-only {name} must align with all joint anchors.")
        parameter_values.append(torch.as_tensor(value[target_indices], dtype=torch.float32, device=device))
    target_timing = torch.as_tensor(arrays["timing"][target_indices], dtype=torch.float32, device=device)
    reference_timing = torch.as_tensor(timing_reference, dtype=torch.float32, device=device)
    t1, t2, b1, amplitude_tensor = parameter_values
    with torch.no_grad():
        observed_target = decoder(t1, t2, b1, target_timing).detach().cpu().numpy()
        reference_target = decoder(t1, t2, b1, reference_timing).detach().cpu().numpy()
    amplitude = amplitude_tensor.detach().cpu().numpy()
    observed_target = observed_target * amplitude[:, None]
    reference_target = reference_target * amplitude[:, None]
    if normalization is None:
        l1, l1_valid = _timing_l1(observed_target, reference_target, reference_valid)
    else:
        l1, l1_valid = normalized_l1_discrepancy(observed_target, reference_target, normalization)
        l1[~reference_valid] = np.nan
        l1_valid &= reference_valid
    from .discrepancy import raw_cosine_discrepancy
    cosine, cosine_valid = raw_cosine_discrepancy(observed_target, reference_target)
    return l1, cosine, {"status": "run", "source": str(parameters.get("source", "external anchor-aligned parameters")), "l1_valid_count": int(l1_valid.sum()), "cosine_valid_count": int(cosine_valid.sum())}


def _timing_l1(target: np.ndarray, reference: np.ndarray, reference_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = reference_valid & np.isfinite(target).all(axis=1) & np.isfinite(reference).all(axis=1)
    result = np.full(target.shape[0], np.nan, dtype=np.float64)
    result[valid] = np.abs(target[valid] - reference[valid]).mean(axis=1)
    return result, valid


def _empty_stack_payload(indices: np.ndarray) -> dict[str, Any]:
    return {"target_indices": indices, "target_fingerprints": np.empty((0, 10)), "reference": np.empty((0, 10)), "reference_valid": np.empty(0, dtype=bool), "timing_reference": np.empty((0, 9)), "nearest_distance_mm": np.empty(0), "observed_l1": np.empty(0), "observed_cosine_raw": np.empty(0), "observed_cosine_normalized": np.empty(0), "timing_only_l1": None, "timing_only_cosine": None, "timing_delta": np.empty((0, 9))}


def run_same_world_reliability_audit(
    joint_dataset: Any,
    *,
    group_axisangle_physical: torch.Tensor,
    subject: str,
    case_id: str,
    geometry_source: str,
    timing_only_decoder: torch.nn.Module | None = None,
    timing_only_parameters: dict[str, Any] | None = None,
    normalization_low_quantile: float = .005,
    normalization_high_quantile: float = .995,
    gate_quantile: float = .90,
    gate_power: float = 3.0,
    gate_epsilon: float = 1e-6,
    srpan_tau_min: float = .05,
    cosine_epsilon: float = 1e-8,
    provenance: dict[str, Any] | None = None,
    sax_observation_path: str | Path | None = None,
) -> dict[str, Any]:
    arrays = _dataset_arrays(joint_dataset, group_axisangle_physical)
    sax_indices = np.flatnonzero(arrays["stack_idx"] == 0)
    target_indices_by_stack = {name: np.flatnonzero(arrays["stack_idx"] == code) for code, name in ((1, "2ch"), (2, "4ch"))}
    result: dict[str, Any] = {"manifest": {}, "pairs": [], "fingerprints": {}, "status": "PASS", "stack_payloads": {}}
    if sax_indices.size == 0:
        raise ValueError("same-world audit requires at least one SAX anchor.")
    try:
        if sax_observation_path is not None:
            reference_stack = build_slice_stack_reference(str(sax_observation_path), np.asarray(group_axisangle_physical.detach().cpu() if isinstance(group_axisangle_physical, torch.Tensor) else group_axisangle_physical))
            with np.load(sax_observation_path, allow_pickle=False) as raw:
                timing9 = np.asarray(raw["timing9_ms"], dtype=np.float64)
                groups = np.asarray(raw["group_idx"], dtype=np.int64)
                weights = np.asarray(raw["weight_idx"], dtype=np.int64)
                images = np.asarray(raw["images"])
            timing_values = {}
            for plane in reference_stack.planes:
                rows = np.flatnonzero((groups == plane.group_idx) & (weights >= 0))
                ordered = rows[np.argsort(weights[rows], kind="stable")]
                timing_values[plane.group_idx] = np.broadcast_to(timing9[ordered[0]][None, None, :], (*images.shape[1:], 9)).copy()
            timing_stack = replace_slice_values(reference_stack, timing_values)
            sampling_admission = {"status": "admitted", "implementation": "slice-stack bilinear plus bracketed through-plane interpolation", "coordinate_convention": reference_stack.coordinate_convention, "plane_count": len(reference_stack.planes), "orientation_spread_deg": reference_stack.orientation_spread_deg, "median_inter_plane_angle_deg": reference_stack.median_inter_plane_angle_deg}
        else:
            reference_stack = None
            reference_grid = build_regular_grid_reference(arrays["physical_xyz"][sax_indices], arrays["observed"][sax_indices])
            timing_grid = build_regular_grid_reference(arrays["physical_xyz"][sax_indices], np.column_stack((arrays["timing"][sax_indices], np.zeros((sax_indices.size, 1)))), axes=reference_grid.axes)
            sampling_admission = {"status": "admitted", "implementation": "test-only RAS Cartesian trilinear interpolation", "coordinate_convention": reference_grid.coordinate_convention, "grid_shape": list(reference_grid.shape)}
    except ValueError as error:
        result["status"] = "BLOCKED_REFERENCE_GEOMETRY"
        result["reference_sampling_qc"] = {"status": "blocked", "reason": str(error), "implementation": "fail-closed; nearest-neighbor is not substituted"}
        for stack, indices in target_indices_by_stack.items():
            result["stack_payloads"][stack] = _empty_stack_payload(indices)
        result["manifest"] = _manifest(subject, case_id, geometry_source, result["status"], provenance, sampling_admission=result["reference_sampling_qc"])
        result["summary"] = _summary_payload(result)
        return result

    sampling_qc = dict(sampling_admission)
    all_valid_target, all_valid_reference = [], []
    for stack, indices in target_indices_by_stack.items():
        if indices.size == 0:
            result["stack_payloads"][stack] = _empty_stack_payload(indices)
            continue
        if reference_stack is not None:
            sampled, valid, support = sample_slice_stack(reference_stack, arrays["physical_xyz"][indices])
            sampled_timing, timing_valid, _ = sample_slice_stack(timing_stack, arrays["physical_xyz"][indices])
        else:
            sampled, valid, support = sample_regular_grid_trilinear(reference_grid, arrays["physical_xyz"][indices])
            sampled_timing, timing_valid, _ = sample_regular_grid_trilinear(timing_grid, arrays["physical_xyz"][indices])
        reference_local, nearest_distance = match_lax_to_sax(arrays["physical_xyz"][sax_indices], arrays["physical_xyz"][indices])
        delta = arrays["timing"][indices] - sampled_timing[:, :9]
        result["stack_payloads"][stack] = {"target_indices": indices, "target_fingerprints": arrays["observed"][indices], "reference": sampled, "reference_valid": valid & timing_valid, "timing_reference": sampled_timing[:, :9], "nearest_distance_mm": nearest_distance, "timing_delta": delta, "support": support}
        all_valid_target.append(arrays["observed"][indices][valid])
        all_valid_reference.append(sampled[valid])
    has_target_stack = any(payload["target_indices"].size for payload in result["stack_payloads"].values())
    active_stack_values = {stack: arrays["observed"][np.flatnonzero(arrays["stack_idx"] == code)] for code, stack in ((0, "sax"), (1, "2ch"), (2, "4ch")) if np.any(arrays["stack_idx"] == code)}
    try:
        stats = fit_case_joint_robust_normalization(active_stack_values, low_quantile=normalization_low_quantile, high_quantile=normalization_high_quantile)
    except ValueError as error:
        result["normalization_error"] = str(error)
        stats = None
        if has_target_stack:
            result["status"] = "BLOCKED_NORMALIZATION"
    if not has_target_stack:
        result["status"] = "PASS_NO_TARGET_STACK"
    for stack, payload in result["stack_payloads"].items():
        indices, reference, valid = payload["target_indices"], payload["reference"], payload["reference_valid"]
        target = arrays["observed"][indices]
        if stats is None or indices.size == 0:
            l1 = np.full(indices.size, np.nan); norm_cosine = np.full(indices.size, np.nan); l1_valid = np.zeros(indices.size, dtype=bool); norm_cosine_valid = np.zeros(indices.size, dtype=bool)
        else:
            l1, l1_valid = normalized_l1_discrepancy(target, reference, stats)
            norm_cosine, norm_cosine_valid = raw_cosine_discrepancy((target - stats.low) / (stats.high - stats.low), (reference - stats.low) / (stats.high - stats.low), epsilon=cosine_epsilon)
            l1[~valid] = np.nan; norm_cosine[~valid] = np.nan; l1_valid &= valid; norm_cosine_valid &= valid
        raw_cosine, raw_valid = raw_cosine_discrepancy(target, reference, epsilon=cosine_epsilon)
        raw_cosine[~valid] = np.nan; raw_valid &= valid
        timing_l1, timing_cosine, timing_status = _timing_only_pair(timing_only_decoder, timing_only_parameters, arrays, indices, payload["timing_reference"], valid, stats)
        payload.update({"observed_l1": l1, "observed_l1_valid": l1_valid, "observed_cosine_raw": raw_cosine, "observed_cosine_raw_valid": raw_valid, "observed_cosine_normalized": norm_cosine, "observed_cosine_normalized_valid": norm_cosine_valid, "timing_only_l1": timing_l1, "timing_only_cosine": timing_cosine, "timing_only_status": timing_status})
        for offset, anchor in enumerate(indices):
            pair = {"subject": subject, "case_id": case_id, "target_stack": stack, "target_anchor_index": int(anchor), "target_group_idx": int(arrays["group_idx"][anchor]), "target_row": int(arrays["row_col"][anchor, 0]), "target_col": int(arrays["row_col"][anchor, 1]), "world_x_mm": float(arrays["physical_xyz"][anchor, 0]), "world_y_mm": float(arrays["physical_xyz"][anchor, 1]), "world_z_mm": float(arrays["physical_xyz"][anchor, 2]), "reference_valid": bool(valid[offset]), "reference_support_info": str(payload.get("support", {}).get("invalid_reason", np.array(["inside_reference_fov"] * indices.size, dtype=object))[offset]), "observed_l1_normalized": float(l1[offset]) if np.isfinite(l1[offset]) else None, "observed_cosine_raw": float(raw_cosine[offset]) if np.isfinite(raw_cosine[offset]) else None, "observed_cosine_normalized": float(norm_cosine[offset]) if np.isfinite(norm_cosine[offset]) else None, "timing_only_l1_normalized": float(timing_l1[offset]) if timing_l1 is not None and np.isfinite(timing_l1[offset]) else None, "timing_only_cosine": float(timing_cosine[offset]) if timing_cosine is not None and np.isfinite(timing_cosine[offset]) else None, "timing_abs_mean_ms": float(np.abs(payload["timing_delta"][offset]).mean()) if valid[offset] else None, "timing_l2_ms": float(np.linalg.norm(payload["timing_delta"][offset])) if valid[offset] else None, "nearest_neighbor_distance_mm": float(payload["nearest_distance_mm"][offset]), "nearest_neighbor_cosine": None}
            result["pairs"].append(pair)
    result["pairs"].sort(key=lambda row: (row["target_stack"], row["target_anchor_index"]))
    result["normalization_statistics"] = stats.as_dict() if stats is not None else {"status": "not_run" if result["status"] == "PASS_NO_TARGET_STACK" else "blocked", "reason": "no target stack anchors" if result["status"] == "PASS_NO_TARGET_STACK" else result.get("normalization_error", "normalization unavailable")}
    result["observed_l1_statistics"] = {stack: _summary(payload["observed_l1"]) for stack, payload in result["stack_payloads"].items()}
    result["observed_cosine_statistics"] = {stack: {"raw": _summary(payload["observed_cosine_raw"]), "normalized": _summary(payload["observed_cosine_normalized"])} for stack, payload in result["stack_payloads"].items()}
    result["timing_statistics"] = {stack: timing_statistics(payload["timing_delta"][np.isfinite(payload["timing_delta"]).all(axis=1)]) for stack, payload in result["stack_payloads"].items()}
    result["timing_only_l1_statistics"] = _timing_summary(result, "timing_only_l1")
    result["timing_only_cosine_statistics"] = _timing_summary(result, "timing_only_cosine")
    result["l1_gate_dry_run"] = {stack: gate_dry_run(payload["observed_l1"], payload.get("observed_l1_valid", np.zeros(payload["observed_l1"].size, bool)), gate_quantile=gate_quantile, gate_power=gate_power, gate_epsilon=gate_epsilon, srpan_tau_min=srpan_tau_min) for stack, payload in result["stack_payloads"].items()}
    result["cosine_gate_dry_run"] = {stack: gate_dry_run(payload["observed_cosine_raw"], payload.get("observed_cosine_raw_valid", np.zeros(payload["observed_cosine_raw"].size, bool)), gate_quantile=gate_quantile, gate_power=gate_power, gate_epsilon=gate_epsilon, srpan_tau_min=srpan_tau_min) for stack, payload in result["stack_payloads"].items()}
    result["normalized_cosine_gate_dry_run"] = {stack: gate_dry_run(payload["observed_cosine_normalized"], payload.get("observed_cosine_normalized_valid", np.zeros(payload["observed_cosine_normalized"].size, bool)), gate_quantile=gate_quantile, gate_power=gate_power, gate_epsilon=gate_epsilon, srpan_tau_min=srpan_tau_min) for stack, payload in result["stack_payloads"].items()}
    result["reference_sampling_qc"] = sampling_qc
    result["manifest"] = _manifest(subject, case_id, geometry_source, result["status"], provenance, sampling_admission=sampling_qc, normalization=result["normalization_statistics"], gate={"quantile": gate_quantile, "power": gate_power, "epsilon": gate_epsilon, "srpan_tau_min": srpan_tau_min}, anchor_order_sha256=_anchor_identity_hash(arrays, np.arange(arrays["physical_xyz"].shape[0])))
    result["summary"] = _summary_payload(result)
    return result


def _timing_summary(result: dict[str, Any], key: str) -> dict[str, Any]:
    return {stack: _summary(np.asarray(payload[key], dtype=float)) if payload[key] is not None else {"status": "not_run", **_summary(np.empty(0))} for stack, payload in result["stack_payloads"].items()}


def _manifest(subject: str, case_id: str, geometry_source: str, status: str, provenance: dict[str, Any] | None, **extra: Any) -> dict[str, Any]:
    provenance_map = dict(provenance or {})
    sparse = provenance_map.get("sparse_manifest") if isinstance(provenance_map.get("sparse_manifest"), dict) else provenance_map
    resolved = sparse.get("resolved_case") or sparse.get("manifest_resolved_case") or case_id
    requested = sparse.get("requested_case") or case_id
    return {"schema": "ismrm_v2_2/same_world_reliability_audit/v1", "subject": subject, "requested_case": requested, "resolved_case": resolved, "status": status, "geometry_source": geometry_source, "selected_geometry": "group_axisangle_post_stack_init_physical", "reference_stack": "sax", "target_stacks": ["2ch", "4ch"], "active_stacks": sparse.get("active_stacks"), "observation_records": sparse.get("observation_records"), "reference_sampling": "same-world-coordinate slice-stack bilinear plus bracketed through-plane interpolation; no extrapolation", "fingerprint_npz_schema": "same_world_fingerprints.npz: <stack>_target_fingerprint and <stack>_reference_fingerprint [N,10]", "discrepancy_definitions": {"l1": "mean absolute difference over ten jointly normalized weights", "cosine_raw": "1-cosine over raw ten weights", "cosine_normalized": "1-cosine over jointly normalized ten weights"}, "timestamp_utc": datetime.now(timezone.utc).isoformat(), **extra, **provenance_map}


def _summary_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {"status": result["status"], "pair_count": {stack: int(payload["target_indices"].size) for stack, payload in result["stack_payloads"].items()}, "reference_valid_fraction": {stack: (float(payload["reference_valid"].mean()) if payload["reference_valid"].size else None) for stack, payload in result["stack_payloads"].items()}, "observed_l1_statistics": result.get("observed_l1_statistics", {}), "observed_cosine_statistics": result.get("observed_cosine_statistics", {}), "timing_statistics": result.get("timing_statistics", {}), "timing_only_l1_statistics": result.get("timing_only_l1_statistics", {}), "timing_only_cosine_statistics": result.get("timing_only_cosine_statistics", {}), "l1_gate_dry_run": result.get("l1_gate_dry_run", {}), "cosine_gate_dry_run": result.get("cosine_gate_dry_run", {}), "normalized_cosine_gate_dry_run": result.get("normalized_cosine_gate_dry_run", {})}


def write_same_world_audit_outputs(result: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"audit output_dir must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / name for name in ("manifest.json", "summary.json", "same_world_pairs.csv", "reference_sampling_qc.json", "normalization_statistics.json", "observed_l1_statistics.json", "observed_cosine_statistics.json", "timing_statistics.json", "timing_only_l1_statistics.json", "timing_only_cosine_statistics.json", "l1_gate_dry_run.json", "cosine_gate_dry_run.json", "normalized_cosine_gate_dry_run.json", "nearest_neighbor_qc.json")}
    for name, payload in (("manifest.json", result["manifest"]), ("summary.json", result["summary"]), ("reference_sampling_qc.json", result.get("reference_sampling_qc", {})), ("normalization_statistics.json", result.get("normalization_statistics", {})), ("observed_l1_statistics.json", result.get("observed_l1_statistics", {})), ("observed_cosine_statistics.json", result.get("observed_cosine_statistics", {})), ("timing_statistics.json", result.get("timing_statistics", {})), ("timing_only_l1_statistics.json", result.get("timing_only_l1_statistics", {})), ("timing_only_cosine_statistics.json", result.get("timing_only_cosine_statistics", {})), ("l1_gate_dry_run.json", result.get("l1_gate_dry_run", {})), ("cosine_gate_dry_run.json", result.get("cosine_gate_dry_run", {})), ("normalized_cosine_gate_dry_run.json", result.get("normalized_cosine_gate_dry_run", {}))):
        paths[name].write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    nearest = {stack: {"backend": nearest_neighbor_backend(), "distance_mm": payload["nearest_distance_mm"].tolist()} for stack, payload in result["stack_payloads"].items()}
    paths["nearest_neighbor_qc.json"].write_text(json.dumps(nearest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["subject", "case_id", "target_stack", "target_anchor_index", "target_group_idx", "target_row", "target_col", "world_x_mm", "world_y_mm", "world_z_mm", "reference_valid", "reference_support_info", "observed_l1_normalized", "observed_cosine_raw", "observed_cosine_normalized", "timing_only_l1_normalized", "timing_only_cosine", "timing_abs_mean_ms", "timing_l2_ms", "nearest_neighbor_distance_mm", "nearest_neighbor_cosine"]
    with paths["same_world_pairs.csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(result["pairs"])
    vectors = {f"{stack}_target_fingerprint": payload["target_fingerprints"] for stack, payload in result["stack_payloads"].items()}
    vectors.update({f"{stack}_reference_fingerprint": payload["reference"] for stack, payload in result["stack_payloads"].items()})
    np.savez_compressed(output / "same_world_fingerprints.npz", **vectors)
    paths["same_world_fingerprints.npz"] = output / "same_world_fingerprints.npz"
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True); parser.add_argument("--case", required=True); parser.add_argument("--prepared-root", required=True); parser.add_argument("--geometry-checkpoint", required=True); parser.add_argument("--sparse-manifest"); parser.add_argument("--signal-decoder-checkpoint"); parser.add_argument("--protocol-yaml"); parser.add_argument("--timing-only-parameters-npz"); parser.add_argument("--output-dir", required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--normalization-low-quantile", type=float, default=.005); parser.add_argument("--normalization-high-quantile", type=float, default=.995); parser.add_argument("--gate-quantile", type=float, default=.90); parser.add_argument("--gate-power", type=float, default=3.0); parser.add_argument("--gate-epsilon", type=float, default=1e-6); parser.add_argument("--srpan-tau-min", type=float, default=.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from ismrm_v2.proposed.q002.contracts import cropped_observation_paths
    from ismrm_v2.proposed.q002.joint_dataset import JointAnchorDataset
    from ismrm_v2.proposed.quantitative_point_dataset import QuantPointDataset
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
    if args.sparse_manifest:
        inputs, manifest_inputs = _manifest_active_inputs(args.sparse_manifest, subject=args.subject, case_id=args.case, prepared_root=args.prepared_root)
    else:
        from ismrm_v2.proposed.q002.contracts import cropped_observation_paths
        inputs = cropped_observation_paths(args.prepared_root, args.subject)
        manifest_inputs = {"input_observation_mode": "unspecified", "active_stacks": [path.parent.name for path in inputs], "observation_records": {path.parent.name: {"path": str(path), "sha256": _sha256_file(path)} for path in inputs}}
    scalar = QuantPointDataset(inputs, device=device); joint = JointAnchorDataset(inputs, scalar)
    try:
        geometry, geometry_provenance = load_geometry_checkpoint(args.geometry_checkpoint, scalar.group_axisangle_init, scalar.group_resolution_xyz_mm)
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError(f"BLOCKED_CASE_GEOMETRY: case-specific checkpoint does not match active dataset: {error}") from error
    decoder = None
    if args.signal_decoder_checkpoint:
        if not args.protocol_yaml:
            raise ValueError("--signal-decoder-checkpoint requires --protocol-yaml")
        from ismrm_v2.proposed.signal_decoder.checkpoint_loader import load_frozen_mlp_decoder_for_benchmark
        decoder = load_frozen_mlp_decoder_for_benchmark(args.signal_decoder_checkpoint, scalar, args.protocol_yaml, device=device)
    elif args.protocol_yaml:
        raise ValueError("--protocol-yaml requires --signal-decoder-checkpoint")
    anchor_order = _joint_anchor_order_sha256(joint)
    parameters = _load_timing_only_parameters(args.timing_only_parameters_npz, int(joint.xyz.shape[0]), anchor_order) if args.timing_only_parameters_npz else None
    if parameters is not None and decoder is None:
        raise ValueError("--timing-only-parameters-npz requires --signal-decoder-checkpoint")
    sparse_manifest = manifest_inputs
    sax_path = next((path for path in inputs if path.parent.name == "sax"), None)
    result = run_same_world_reliability_audit(joint, group_axisangle_physical=geometry, subject=args.subject, case_id=args.case, geometry_source=geometry_provenance["source"], timing_only_decoder=decoder, timing_only_parameters=parameters, normalization_low_quantile=args.normalization_low_quantile, normalization_high_quantile=args.normalization_high_quantile, gate_quantile=args.gate_quantile, gate_power=args.gate_power, gate_epsilon=args.gate_epsilon, srpan_tau_min=args.srpan_tau_min, sax_observation_path=sax_path, provenance={**_git_provenance(), "device": str(device), "geometry_checkpoint": geometry_provenance, "prepared_root": str(Path(args.prepared_root).resolve()), "sparse_manifest": sparse_manifest, "joint_anchor_order_sha256": anchor_order, "timing_only_parameters": {"path": parameters["source"], "sha256": parameters["sha256"], "anchor_order_sha256": anchor_order} if parameters else None, "timing_only_decoder_checkpoint": str(args.signal_decoder_checkpoint) if args.signal_decoder_checkpoint else None})
    print(json.dumps({name: str(path) for name, path in write_same_world_audit_outputs(result, args.output_dir).items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
