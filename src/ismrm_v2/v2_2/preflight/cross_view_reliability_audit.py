"""Numerical primitives for the v2.2 cross-view reliability preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np
import torch
try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - exercised only in unsupported runtimes
    cKDTree = None

from nesvor.transform import ax_transform_points
from ismrm_v2.sampling.sparse_profile_manifest import LEGACY_SCHEMA_VERSION, SCHEMA_VERSION, validate_sparse_profile_manifest


SMALL_ALL_PAIRS_FALLBACK_MAX_PAIRS = 100_000
OBSERVATION_STACKS = ("sax", "2ch", "4ch")
GEOMETRY_CHECKPOINT_KEY = "group_axisangle_post_stack_init_physical"
GEOMETRY_ABS_TOLERANCE = 1e-5


def _points(name: str, value: np.ndarray) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite [N,3] physical RAS-mm points.")
    return points


def match_lax_to_sax(
    sax_xyz_ras_mm: np.ndarray,
    lax_xyz_ras_mm: np.ndarray,
    *,
    chunk_size: int = 8192,
    reference_chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact nearest SAX anchor for every LAX anchor, in millimetres."""

    sax, lax = _points("sax_xyz_ras_mm", sax_xyz_ras_mm), _points("lax_xyz_ras_mm", lax_xyz_ras_mm)
    if sax.shape[0] == 0:
        raise ValueError("At least one SAX reference anchor is required.")
    if chunk_size < 1 or reference_chunk_size < 1:
        raise ValueError("chunk sizes must be positive.")
    if cKDTree is not None:
        distance_mm, reference_index = cKDTree(sax).query(lax, k=1)
        return np.asarray(reference_index, dtype=np.int64), np.asarray(distance_mm, dtype=np.float64)
    if sax.shape[0] * lax.shape[0] > SMALL_ALL_PAIRS_FALLBACK_MAX_PAIRS:
        raise RuntimeError(
            "SciPy cKDTree is required for production cross-view matching; "
            "the small all-pairs fallback is limited to unit-scale fixtures."
        )
    indices = np.empty(lax.shape[0], dtype=np.int64)
    distances = np.empty(lax.shape[0], dtype=np.float64)
    for start in range(0, lax.shape[0], chunk_size):
        target = lax[start : start + chunk_size]
        best_squared = np.full(target.shape[0], np.inf, dtype=np.float64)
        best_indices = np.zeros(target.shape[0], dtype=np.int64)
        for reference_start in range(0, sax.shape[0], reference_chunk_size):
            reference = sax[reference_start : reference_start + reference_chunk_size]
            squared = np.sum((target[:, None, :] - reference[None, :, :]) ** 2, axis=2)
            local = np.argmin(squared, axis=1)
            candidate = squared[np.arange(target.shape[0]), local]
            replace = candidate < best_squared
            best_squared[replace] = candidate[replace]
            best_indices[replace] = reference_start + local[replace]
        stop = start + target.shape[0]
        indices[start:stop] = best_indices
        distances[start:stop] = np.sqrt(best_squared)
    return indices, distances


def nearest_neighbor_backend() -> str:
    """Describe the exact NN implementation selected by this runtime."""

    return "scipy.spatial.cKDTree" if cKDTree is not None else "guarded small NumPy all-pairs fallback"


def cosine_discrepancy(
    target: np.ndarray,
    reference: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1-cosine for matching 10-weight vectors and flag unusable rows."""

    left, right = np.asarray(target, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 10:
        raise ValueError("target and reference must be matching [N,10] arrays.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    left_norm, right_norm = np.linalg.norm(left, axis=1), np.linalg.norm(right, axis=1)
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1) & (left_norm >= epsilon) & (right_norm >= epsilon)
    result = np.full(left.shape[0], np.nan, dtype=np.float64)
    result[valid] = 1.0 - np.sum(left[valid] * right[valid], axis=1) / (left_norm[valid] * right_norm[valid] + epsilon)
    return result, valid


def _percentile(values: np.ndarray, value: float) -> float | None:
    return None if values.size == 0 else float(np.percentile(values, value))


def timing_statistics(delta_timing9_ms: np.ndarray) -> dict[str, Any]:
    """Summarize signed timing deltas without changing their millisecond convention."""

    delta = np.asarray(delta_timing9_ms, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[1] != 9 or not np.isfinite(delta).all():
        raise ValueError("delta_timing9_ms must be finite [N,9].")
    absolute = np.abs(delta)
    per_pair_abs_mean = absolute.mean(axis=1)
    per_pair_l2 = np.linalg.norm(delta, axis=1)
    return {
        "count": int(delta.shape[0]),
        "per_dimension_mean_abs_ms": absolute.mean(axis=0).tolist() if delta.shape[0] else [None] * 9,
        "per_dimension_median_abs_ms": np.median(absolute, axis=0).tolist() if delta.shape[0] else [None] * 9,
        "per_dimension_p90_abs_ms": np.percentile(absolute, 90, axis=0).tolist() if delta.shape[0] else [None] * 9,
        "per_dimension_p95_abs_ms": np.percentile(absolute, 95, axis=0).tolist() if delta.shape[0] else [None] * 9,
        "per_pair_abs_mean_ms": per_pair_abs_mean.tolist(),
        "per_pair_l2_ms": per_pair_l2.tolist(),
        "per_pair_abs_mean_summary_ms": {"mean": float(per_pair_abs_mean.mean()) if per_pair_abs_mean.size else None, "median": _percentile(per_pair_abs_mean, 50), "p90": _percentile(per_pair_abs_mean, 90), "p95": _percentile(per_pair_abs_mean, 95)},
        "per_pair_l2_summary_ms": {"mean": float(per_pair_l2.mean()) if per_pair_l2.size else None, "median": _percentile(per_pair_l2, 50), "p90": _percentile(per_pair_l2, 90), "p95": _percentile(per_pair_l2, 95)},
    }


def binned_cosine_statistics(
    metric: np.ndarray,
    discrepancy: np.ndarray,
    edges: Sequence[float],
) -> list[dict[str, Any]]:
    """Return transparent half-open bins, with the final bin including its upper edge."""

    metric_array, discrepancy_array, boundaries = np.asarray(metric, dtype=np.float64), np.asarray(discrepancy, dtype=np.float64), np.asarray(edges, dtype=np.float64)
    if metric_array.shape != discrepancy_array.shape or metric_array.ndim != 1:
        raise ValueError("metric and discrepancy must be matching one-dimensional arrays.")
    if boundaries.ndim != 1 or boundaries.size < 2 or np.any(np.diff(boundaries) <= 0):
        raise ValueError("edges must be strictly increasing with at least two values.")
    valid = np.isfinite(metric_array) & np.isfinite(discrepancy_array)
    results: list[dict[str, Any]] = []
    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mask = valid & (metric_array >= lower) & ((metric_array <= upper) if index == boundaries.size - 2 else (metric_array < upper))
        values = discrepancy_array[mask]
        results.append({
            "lower": None if np.isneginf(lower) else float(lower),
            "upper": None if np.isposinf(upper) else float(upper), "count": int(values.size),
            "median_cosine_discrepancy": _percentile(values, 50),
            "p90_cosine_discrepancy": _percentile(values, 90),
        })
    return results


def timing_only_cosine_discrepancy(
    decoder: torch.nn.Module,
    t1_ms: torch.Tensor,
    t2_ms: torch.Tensor,
    b1: torch.Tensor,
    timing_sax_ms: torch.Tensor,
    timing_lax_ms: torch.Tensor,
    *,
    amplitude: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Evaluate a frozen decoder twice with identical tissue parameters and changed timing only."""

    if timing_sax_ms.shape != timing_lax_ms.shape or timing_sax_ms.ndim != 2 or timing_sax_ms.shape[1] != 9:
        raise ValueError("timing inputs must be matching [N,9] tensors.")
    if any(value.shape != timing_sax_ms.shape[:1] for value in (t1_ms, t2_ms, b1)):
        raise ValueError("T1, T2 and B1 must be [N] and align with timing rows.")
    if amplitude is not None and amplitude.shape != timing_sax_ms.shape[:1]:
        raise ValueError("amplitude must be [N] and align with timing rows.")
    with torch.no_grad():
        fingerprint_sax = decoder(t1_ms, t2_ms, b1, timing_sax_ms)
        fingerprint_lax = decoder(t1_ms, t2_ms, b1, timing_lax_ms)
        if amplitude is not None:
            fingerprint_sax = fingerprint_sax * amplitude[:, None]
            fingerprint_lax = fingerprint_lax * amplitude[:, None]
    values, _ = cosine_discrepancy(fingerprint_lax.detach().cpu().numpy(), fingerprint_sax.detach().cpu().numpy(), epsilon=epsilon)
    return values


STACK_NAMES = {0: "sax", 1: "2ch", 2: "4ch"}
COVERAGE_THRESHOLDS_MM = (1.0, 2.0, 3.0, 5.0)


def _summary(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "median": _percentile(finite, 50),
        "p75": _percentile(finite, 75),
        "p90": _percentile(finite, 90),
        "p95": _percentile(finite, 95),
        "p99": _percentile(finite, 99),
        "max": float(finite.max()) if finite.size else None,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _correlations(metric: np.ndarray, discrepancy: np.ndarray) -> dict[str, float | int | None]:
    valid = np.isfinite(metric) & np.isfinite(discrepancy)
    x, y = metric[valid], discrepancy[valid]
    if x.size < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return {"valid_count": int(x.size), "pearson": None, "spearman": None}
    return {
        "valid_count": int(x.size),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1]),
    }


def _quantile_bins(metric: np.ndarray, discrepancy: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(metric) & np.isfinite(discrepancy)
    values = metric[valid]
    if values.size == 0:
        return {"edges": [], "bins": []}
    edges = np.unique(np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0]))
    if edges.size == 1:
        selected = discrepancy[valid]
        return {
            "edges": [float(edges[0]), float(edges[0])],
            "bins": [{"lower": float(edges[0]), "upper": float(edges[0]), "count": int(selected.size), "median_cosine_discrepancy": _percentile(selected, 50), "p90_cosine_discrepancy": _percentile(selected, 90)}],
        }
    return {"edges": edges.tolist(), "bins": binned_cosine_statistics(metric, discrepancy, edges)}


def _dataset_arrays(joint_dataset: Any, group_axisangle_physical: torch.Tensor) -> dict[str, np.ndarray]:
    required = ("xyz", "observed", "group_idx", "stack_idx", "timing", "row_col")
    missing = [name for name in required if not hasattr(joint_dataset, name)]
    if missing:
        raise TypeError(f"joint_dataset lacks required fields: {missing}")
    xyz = torch.as_tensor(joint_dataset.xyz, dtype=torch.float32)
    groups = torch.as_tensor(joint_dataset.group_idx, dtype=torch.long, device=xyz.device)
    poses = torch.as_tensor(group_axisangle_physical, dtype=xyz.dtype, device=xyz.device)
    if poses.ndim != 2 or poses.shape[1] != 6 or groups.numel() != xyz.shape[0] or groups.numel() and int(groups.max()) >= poses.shape[0]:
        raise ValueError("group_axisangle_physical must cover every joint anchor group with [G,6] poses.")
    physical = ax_transform_points(poses[groups], xyz, trans_first=True).detach().cpu().numpy()
    return {
        "physical_xyz": physical,
        "observed": torch.as_tensor(joint_dataset.observed).detach().cpu().numpy(),
        "group_idx": groups.detach().cpu().numpy(),
        "stack_idx": torch.as_tensor(joint_dataset.stack_idx).detach().cpu().numpy(),
        "timing": torch.as_tensor(joint_dataset.timing).detach().cpu().numpy(),
        "row_col": torch.as_tensor(joint_dataset.row_col).detach().cpu().numpy(),
    }


def _timing_only_values(
    decoder: torch.nn.Module | None,
    parameters: dict[str, Any] | None,
    target_indices: np.ndarray,
    reference_indices: np.ndarray,
    timing: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if decoder is None:
        return None, {"timing_only_simulator_status": "not_run", "reason": "no decoder was supplied"}
    if parameters is None:
        return None, {"timing_only_simulator_status": "not_run", "reason": "no anchor-aligned T1/T2/B1/amplitude parameters were supplied"}
    missing = {"t1_ms", "t2_ms", "b1", "amplitude"}.difference(parameters)
    if missing:
        return None, {"timing_only_simulator_status": "not_run", "reason": f"parameter source lacks {sorted(missing)}"}
    count = timing.shape[0]
    decoder_device = next(decoder.parameters(), None)
    device = decoder_device.device if decoder_device is not None else torch.device("cpu")
    tensors = []
    for name in ("t1_ms", "t2_ms", "b1", "amplitude"):
        value = torch.as_tensor(parameters[name], dtype=torch.float32)
        if value.shape != (count,):
            raise ValueError(f"timing-only {name} must be anchor-aligned [N] with N={count}.")
        tensors.append(value[target_indices].to(device))
    timing_sax = torch.as_tensor(timing[reference_indices], dtype=torch.float32, device=device)
    timing_lax = torch.as_tensor(timing[target_indices], dtype=torch.float32, device=device)
    return timing_only_cosine_discrepancy(decoder, *tensors[:3], timing_sax, timing_lax, amplitude=tensors[3], epsilon=epsilon), {
        "timing_only_simulator_status": "run",
        "source": str(parameters.get("source", "explicit anchor-aligned parameters")),
    }


def _timing_only_statistics(stack_payloads: dict[str, dict[str, Any]], status: dict[str, Any]) -> dict[str, Any]:
    """Summarize timing-only discrepancy without treating missing simulation as data."""

    if status["timing_only_simulator_status"] != "run":
        return dict(status)
    stacks: dict[str, Any] = {}
    for stack_name, payload in stack_payloads.items():
        values = payload["timing_only"]
        if values is None:
            stacks[stack_name] = {"count": 0, "valid_count": 0, "invalid_count": 0, **_summary(np.empty(0)), "corr_timing_only_vs_timing_abs_mean_ms": None, "corr_timing_only_vs_timing_l2_ms": None}
            continue
        values = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(values)
        delta = payload["delta_timing9_ms"]
        timing_abs_mean = np.abs(delta).mean(axis=1) if delta.size else np.empty(0)
        timing_l2 = np.linalg.norm(delta, axis=1) if delta.size else np.empty(0)
        stacks[stack_name] = {
            "count": int(values.size), "valid_count": int(valid.sum()), "invalid_count": int((~valid).sum()),
            **_summary(values[valid]),
            "corr_timing_only_vs_timing_abs_mean_ms": _correlations(timing_abs_mean, values),
            "corr_timing_only_vs_timing_l2_ms": _correlations(timing_l2, values),
        }
    return {
        **status,
        "timing_only_parameter_source": "external_anchor_aligned_npz",
        "stacks": stacks,
    }


def _is_l100_case(value: object) -> bool:
    text = str(value).upper()
    return text == "L100" or text.startswith("L100_") or text.startswith("L100-")


def _resolved_observation_paths(observation_paths: dict[str, str | Path]) -> dict[str, Path]:
    if set(observation_paths) != set(OBSERVATION_STACKS):
        raise ValueError("preflight observations must contain exactly sax, 2ch, and 4ch.")
    return {stack: Path(observation_paths[stack]).resolve() for stack in OBSERVATION_STACKS}


def _manifest_observation_path(stack_record: dict[str, Any], field: str, *, stack: str) -> Path:
    observation = stack_record.get(field)
    if not isinstance(observation, dict) or not isinstance(observation.get("path"), str):
        raise ValueError(f"sparse manifest {stack}.{field} is missing its validated observation path.")
    return Path(observation["path"]).resolve()


def _profile_l100_full_retention(manifest: dict[str, Any]) -> None:
    """Prove all profile-v2 stacks retain every ordered source group."""

    stacks = manifest.get("stacks")
    if not isinstance(stacks, dict):
        raise ValueError("L100 profile manifest has no stack records.")
    for stack in OBSERVATION_STACKS:
        record = stacks.get(stack)
        if not isinstance(record, dict):
            raise ValueError(f"L100 profile manifest lacks {stack}.")
        ordered = record.get("ordered_source_group_ids")
        retained = record.get("retained_original_group_ids")
        heldout = record.get("heldout_original_group_ids")
        source_count = record.get("source_count")
        retained_count = record.get("retained_count")
        if (
            record.get("training_stack_present") is not True
            or not isinstance(ordered, list)
            or not isinstance(retained, list)
            or heldout != []
            or source_count != len(ordered)
            or retained_count != len(retained)
            or retained_count != source_count
            or retained != ordered
        ):
            raise ValueError(f"L100 profile manifest does not prove full retention for {stack}.")


def _admit_manifest_observation_mode(manifest: dict[str, Any], observation_paths: dict[str, Path]) -> str:
    """Admit exactly one validated source or derived three-stack observation set."""

    stacks = manifest["stacks"]
    source_paths = {
        stack: _manifest_observation_path(stacks[stack], "source_observation", stack=stack)
        for stack in OBSERVATION_STACKS
    }
    source_match = all(observation_paths[stack] == source_paths[stack] for stack in OBSERVATION_STACKS)
    if manifest["schema_version"] == LEGACY_SCHEMA_VERSION:
        if not source_match:
            raise ValueError("legacy L100 audit inputs must be the manifest source observations.")
        return "source"
    derived_paths = {
        stack: _manifest_observation_path(stacks[stack], "derived_observation", stack=stack)
        for stack in OBSERVATION_STACKS
    }
    derived_match = all(observation_paths[stack] == derived_paths[stack] for stack in OBSERVATION_STACKS)
    if source_match:
        return "source"
    if derived_match:
        return "derived"
    declared_paths = set(source_paths.values()) | set(derived_paths.values())
    if any(path in declared_paths for path in observation_paths.values()):
        raise ValueError("L100 profile audit inputs must be one coherent source or derived observation set, not mixed stacks.")
    raise ValueError("L100 profile audit inputs do not match the manifest source or derived observations.")


def validate_preflight_sparse_manifest(
    manifest_path: str | Path,
    *,
    subject: str,
    requested_case: str,
    observation_paths: dict[str, str | Path],
) -> dict[str, Any]:
    """Validate an L100 manifest against the exact three observation archives in use."""

    if not _is_l100_case(requested_case):
        raise ValueError("v2.2 preflight currently requires requested --case L100.")
    manifest = validate_sparse_profile_manifest(manifest_path, subject=subject, strict=False)
    if manifest.get("subject") != subject:
        raise ValueError(f"sparse manifest subject {manifest.get('subject')!r} disagrees with --subject {subject!r}.")
    resolved_case = manifest.get("case_id") or manifest.get("condition")
    if not _is_l100_case(resolved_case):
        raise ValueError(f"v2.2 preflight requires an L100/full-long-axis manifest, got {resolved_case!r}.")
    paths = _resolved_observation_paths(observation_paths)
    schema = manifest.get("schema_version")
    if schema == SCHEMA_VERSION:
        _profile_l100_full_retention(manifest)
    elif schema == LEGACY_SCHEMA_VERSION:
        for stack in OBSERVATION_STACKS:
            record = manifest.get("stacks", {}).get(stack)
            if not isinstance(record, dict) or record.get("heldout_original_group_ids") != [] or "derived_observation" in record:
                raise ValueError(f"legacy L100 manifest does not prove source identity/full retention for {stack}.")
    else:  # The project validator should have rejected this before admission.
        raise ValueError(f"unsupported sparse manifest schema for v2.2 preflight: {schema!r}.")
    mode = _admit_manifest_observation_mode(manifest, paths)
    return {
        "sparse_manifest_status": "validated",
        "requested_case": requested_case,
        "manifest_resolved_case": str(resolved_case),
        "schema_version": manifest.get("schema_version"),
        "profile_id": manifest.get("profile_id"),
        "case_id": manifest.get("case_id"),
        "fold_id": manifest.get("fold_id"),
        "fold_count": manifest.get("fold_count"),
        "path": str(manifest["_manifest_path"]),
        "sha256": str(manifest["_manifest_sha256"]),
        "l100_full_retention_verified": True,
        "legacy_identity_full_retention": schema == LEGACY_SCHEMA_VERSION,
        "input_observation_mode": mode,
    }


def resolve_preflight_sparse_manifest(
    manifest_path: str | Path | None,
    *,
    subject: str,
    requested_case: str,
    observation_paths: dict[str, str | Path],
) -> dict[str, Any]:
    """Return explicit provenance for either a validated L100 manifest or its absence."""

    if manifest_path is None:
        return {
            "sparse_manifest_status": "not_supplied",
            "requested_case": requested_case,
            "manifest_resolved_case": None,
        }
    return validate_preflight_sparse_manifest(
        manifest_path,
        subject=subject,
        requested_case=requested_case,
        observation_paths=observation_paths,
    )


def run_cross_view_reliability_audit(
    joint_dataset: Any,
    *,
    group_axisangle_physical: torch.Tensor,
    subject: str,
    case_id: str,
    geometry_source: str,
    timing_only_decoder: torch.nn.Module | None = None,
    timing_only_parameters: dict[str, Any] | None = None,
    cosine_epsilon: float = 1e-8,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit fixed post-HB1 geometry; this function neither trains nor changes poses."""

    arrays = _dataset_arrays(joint_dataset, group_axisangle_physical)
    sax_indices = np.flatnonzero(arrays["stack_idx"] == 0)
    if sax_indices.size == 0:
        raise ValueError("The joint dataset has no SAX anchors (stack_idx=0).")
    pairs: list[dict[str, Any]] = []
    stack_payloads: dict[str, dict[str, Any]] = {}
    timing_status: dict[str, Any] = {"timing_only_simulator_status": "not_run", "reason": "no LAX anchors"}
    for stack_code, stack_name in ((1, "2ch"), (2, "4ch")):
        target_indices = np.flatnonzero(arrays["stack_idx"] == stack_code)
        if target_indices.size == 0:
            stack_payloads[stack_name] = {"target_indices": target_indices, "reference_indices": np.empty(0, dtype=np.int64), "distance_mm": np.empty(0), "delta_timing9_ms": np.empty((0, 9)), "cosine": np.empty(0), "valid": np.empty(0, dtype=bool), "timing_only": None}
            continue
        local_reference, distance_mm = match_lax_to_sax(arrays["physical_xyz"][sax_indices], arrays["physical_xyz"][target_indices])
        reference_indices = sax_indices[local_reference]
        delta = arrays["timing"][target_indices] - arrays["timing"][reference_indices]
        cosine, valid = cosine_discrepancy(arrays["observed"][target_indices], arrays["observed"][reference_indices], epsilon=cosine_epsilon)
        timing_only, timing_status = _timing_only_values(timing_only_decoder, timing_only_parameters, target_indices, reference_indices, arrays["timing"], epsilon=cosine_epsilon)
        stack_payloads[stack_name] = {"target_indices": target_indices, "reference_indices": reference_indices, "distance_mm": distance_mm, "delta_timing9_ms": delta, "cosine": cosine, "valid": valid, "timing_only": timing_only}
        for offset, (target_index, reference_index) in enumerate(zip(target_indices, reference_indices)):
            record = {
                "subject": subject, "case_id": case_id, "target_stack": stack_name,
                "target_anchor_index": int(target_index), "target_group_idx": int(arrays["group_idx"][target_index]),
                "reference_stack": "sax", "reference_anchor_index": int(reference_index), "reference_group_idx": int(arrays["group_idx"][reference_index]),
                "distance_mm": float(distance_mm[offset]),
                "timing_abs_mean_ms": float(np.abs(delta[offset]).mean()), "timing_l2_ms": float(np.linalg.norm(delta[offset])),
                "observed_cosine_discrepancy": float(cosine[offset]) if valid[offset] else None,
                "timing_only_cosine_discrepancy": None if timing_only is None or not np.isfinite(timing_only[offset]) else float(timing_only[offset]),
                "valid_observed_pair": bool(valid[offset]),
                "target_row": int(arrays["row_col"][target_index, 0]), "target_col": int(arrays["row_col"][target_index, 1]),
            }
            for dimension in range(9):
                record[f"delta_timing9_ms_{dimension}"] = float(delta[offset, dimension])
                record[f"abs_delta_timing9_ms_{dimension}"] = float(abs(delta[offset, dimension]))
            pairs.append(record)
    coverage: dict[str, Any] = {}
    timing_stats: dict[str, Any] = {}
    fingerprint_stats: dict[str, Any] = {}
    confounding: dict[str, Any] = {}
    for stack_name, payload in stack_payloads.items():
        distances, delta, cosine, valid = payload["distance_mm"], payload["delta_timing9_ms"], payload["cosine"], payload["valid"]
        coverage[stack_name] = {f"<= {threshold:g} mm": int((distances <= threshold).sum()) for threshold in COVERAGE_THRESHOLDS_MM}
        coverage[stack_name]["anchor_count"] = int(distances.size)
        coverage[stack_name]["coverage_fraction"] = {f"<= {threshold:g} mm": (float((distances <= threshold).mean()) if distances.size else None) for threshold in COVERAGE_THRESHOLDS_MM}
        coverage[stack_name]["distance_mm_summary"] = _summary(distances)
        timing_stats[stack_name] = timing_statistics(delta)
        fingerprint_stats[stack_name] = _summary(cosine[valid]) | {"valid_count": int(valid.sum()), "invalid_count": int((~valid).sum())}
        timing_metric = np.abs(delta).mean(axis=1) if delta.size else np.empty(0)
        distance_edges = [0.0, 1.0, 2.0, 3.0, 5.0, np.inf]
        confounding[stack_name] = {
            "cosine_vs_distance": _correlations(distances, cosine),
            "cosine_vs_timing_abs_mean": _correlations(timing_metric, cosine),
            "distance_bins": binned_cosine_statistics(distances, cosine, distance_edges),
            "timing_quantile_bins": _quantile_bins(timing_metric, cosine),
        }
    timing_only_report = _timing_only_statistics(stack_payloads, timing_status)
    manifest = {
        "schema": "ismrm_v2_2/cross_view_reliability_preflight/v1",
        "subject": subject, "case": case_id, "case_id": case_id, "stack_mapping": STACK_NAMES,
        "reference_stack": "sax", "geometry_source": geometry_source,
        "post_hb1_fixed_geometry": geometry_source,
        "nearest_neighbor_implementation": nearest_neighbor_backend(),
        "coverage_thresholds_mm": list(COVERAGE_THRESHOLDS_MM), "cosine_epsilon": cosine_epsilon,
        "timing_difference_definition": "timing_LAX - timing_SAX, in stored physical milliseconds",
        "timing_only_simulator": timing_only_report,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "joint_dataset_qc": getattr(joint_dataset, "qc", None),
        **dict(provenance or {}),
    }
    return {
        "manifest": manifest, "pairs": pairs, "coverage_statistics": coverage,
        "timing_statistics": timing_stats, "fingerprint_statistics": fingerprint_stats,
        "confounding_statistics": confounding, "timing_only_simulator": timing_only_report,
        "summary": {
            "subject": subject, "case_id": case_id,
            "anchor_pairs": {name: int(payload["distance_mm"].size) for name, payload in stack_payloads.items()},
            "coverage_statistics": coverage, "timing_statistics": timing_stats,
            "fingerprint_statistics": fingerprint_stats, "confounding_statistics": confounding,
            "timing_only_simulator": timing_only_report,
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_pairs(path: Path, pairs: list[dict[str, Any]]) -> None:
    base = ["subject", "case_id", "target_stack", "target_anchor_index", "target_group_idx", "reference_stack", "reference_anchor_index", "reference_group_idx", "distance_mm", "timing_abs_mean_ms", "timing_l2_ms", "observed_cosine_discrepancy", "timing_only_cosine_discrepancy", "valid_observed_pair", "target_row", "target_col"]
    fields = base + [f"{prefix}{dimension}" for prefix in ("delta_timing9_ms_", "abs_delta_timing9_ms_") for dimension in range(9)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(result: dict[str, Any], output: Path) -> tuple[dict[str, Path], str | None]:
    try:
        os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib-cache")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # plotting is optional; numerical artifacts have already been written
        return {}, f"{type(error).__name__}: {error}"
    created: dict[str, Path] = {}
    for stack in ("2ch", "4ch"):
        rows = [item for item in result["pairs"] if item["target_stack"] == stack and item["valid_observed_pair"]]
        cosine = np.asarray([item["observed_cosine_discrepancy"] for item in rows], dtype=float)
        distance = np.asarray([item["distance_mm"] for item in rows], dtype=float)
        timing = np.asarray([item["timing_abs_mean_ms"] for item in rows], dtype=float)
        hist = output / f"hist_cosine_{stack}.png"
        figure, axis = plt.subplots(); axis.hist(cosine, bins="auto" if cosine.size else 1); axis.set(xlabel="Observed cosine discrepancy (1 - cosine)", ylabel="Pair count", title=f"{stack}: observed fingerprint discrepancy"); figure.tight_layout(); figure.savefig(hist, dpi=150); plt.close(figure); created[hist.name] = hist
        for metric, label, name in ((distance, "Physical nearest-SAX distance (mm)", "distance"), (timing, "Mean absolute timing difference (ms)", "timing")):
            scatter = output / f"scatter_cosine_vs_{name}_{stack}.png"
            figure, axis = plt.subplots(); axis.scatter(metric, cosine, s=10); axis.set(xlabel=label, ylabel="Observed cosine discrepancy (1 - cosine)", title=f"{stack}: cosine versus {name}"); figure.tight_layout(); figure.savefig(scatter, dpi=150); plt.close(figure); created[scatter.name] = scatter
    return created, None


def write_audit_outputs(result: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write every audit artifact below one caller-provided new or empty directory."""

    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"v2.2 audit output_dir must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest.json": output / "manifest.json", "summary.json": output / "summary.json",
        "matched_pairs.csv": output / "matched_pairs.csv", "coverage_statistics.json": output / "coverage_statistics.json",
        "timing_statistics.json": output / "timing_statistics.json", "fingerprint_statistics.json": output / "fingerprint_statistics.json",
        "cosine_vs_distance.csv": output / "cosine_vs_distance.csv", "cosine_vs_timing.csv": output / "cosine_vs_timing.csv",
        "timing_only_simulator.json": output / "timing_only_simulator.json",
    }
    _write_json(paths["manifest.json"], result["manifest"])
    _write_json(paths["summary.json"], result["summary"])
    _write_pairs(paths["matched_pairs.csv"], result["pairs"])
    _write_json(paths["coverage_statistics.json"], result["coverage_statistics"])
    _write_json(paths["timing_statistics.json"], result["timing_statistics"])
    _write_json(paths["fingerprint_statistics.json"], result["fingerprint_statistics"])
    _write_json(paths["timing_only_simulator.json"], result["timing_only_simulator"])
    distance_rows = [{"target_stack": item["target_stack"], "distance_mm": item["distance_mm"], "observed_cosine_discrepancy": item["observed_cosine_discrepancy"], "valid_observed_pair": item["valid_observed_pair"]} for item in result["pairs"]]
    timing_rows = [{"target_stack": item["target_stack"], "timing_abs_mean_ms": item["timing_abs_mean_ms"], "timing_l2_ms": item["timing_l2_ms"], "observed_cosine_discrepancy": item["observed_cosine_discrepancy"], "valid_observed_pair": item["valid_observed_pair"]} for item in result["pairs"]]
    _write_csv(paths["cosine_vs_distance.csv"], distance_rows, ["target_stack", "distance_mm", "observed_cosine_discrepancy", "valid_observed_pair"])
    _write_csv(paths["cosine_vs_timing.csv"], timing_rows, ["target_stack", "timing_abs_mean_ms", "timing_l2_ms", "observed_cosine_discrepancy", "valid_observed_pair"])
    figures, plotting_error = _write_plots(result, output)
    if plotting_error is not None:
        manifest = dict(result["manifest"]); manifest["plotting_status"] = "not_run"; manifest["plotting_error"] = plotting_error; _write_json(paths["manifest.json"], manifest)
    else:
        manifest = dict(result["manifest"]); manifest["plotting_status"] = "written"; _write_json(paths["manifest.json"], manifest); paths.update(figures)
    return paths


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _joint_anchor_order_sha256(joint_dataset: Any) -> str:
    """Hash the stable stack/group/native-pixel order required by parameter NPZs."""

    identity = np.column_stack((
        torch.as_tensor(joint_dataset.stack_idx).detach().cpu().numpy(),
        torch.as_tensor(joint_dataset.group_idx).detach().cpu().numpy(),
        torch.as_tensor(joint_dataset.row_col).detach().cpu().numpy(),
    )).astype("<i8", copy=False)
    return hashlib.sha256(identity.tobytes()).hexdigest()


def _canonical_cpu_numeric_sha256(value: torch.Tensor | np.ndarray) -> str:
    """Hash a finite numeric array in a shape-aware, little-endian CPU representation."""

    array = torch.as_tensor(value).detach().cpu().numpy()
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("geometry provenance requires a finite numeric array.")
    canonical = np.ascontiguousarray(array, dtype="<f8")
    shape = np.asarray(canonical.shape, dtype="<i8")
    return hashlib.sha256(shape.tobytes() + canonical.tobytes()).hexdigest()


def _geometry_tensor(value: Any, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value).detach().cpu()
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"{name} must be a numeric tensor/array.") from error
    if not np.issubdtype(tensor.numpy().dtype, np.number):
        raise ValueError(f"{name} must be a numeric tensor/array.")
    if not torch.isfinite(tensor.to(dtype=torch.float64)).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _geometry_max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float((left.to(dtype=torch.float64) - right.to(dtype=torch.float64)).abs().max().item())


def _compare_geometry(
    checkpoint_value: torch.Tensor,
    prepared_value: torch.Tensor,
    name: str,
    *,
    tolerance: float,
) -> float:
    if checkpoint_value.shape != prepared_value.shape:
        raise ValueError(f"{name} shape mismatch: checkpoint {tuple(checkpoint_value.shape)} != prepared {tuple(prepared_value.shape)}.")
    maximum = _geometry_max_abs_diff(checkpoint_value, prepared_value)
    if not torch.allclose(checkpoint_value.to(dtype=torch.float64), prepared_value.to(dtype=torch.float64), rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} mismatch: maximum absolute difference {maximum:g} exceeds tolerance {tolerance:g} (rtol=0).")
    return maximum


def load_geometry_checkpoint(
    path: str | Path,
    prepared_dicom_geometry: torch.Tensor | np.ndarray,
    prepared_group_resolution: torch.Tensor | np.ndarray,
    *,
    tolerance: float = GEOMETRY_ABS_TOLERANCE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load and fail-closed validate the frozen Q002 post-HB1 geometry."""

    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"geometry checkpoint does not exist: {checkpoint_path}")
    if tolerance < 0:
        raise ValueError("geometry tolerance must be non-negative.")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError as error:  # pragma: no cover - only unsupported old Torch releases
        raise RuntimeError("CPU-safe geometry checkpoint loading requires torch.load(..., weights_only=False).") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("geometry checkpoint must contain a mapping payload.")

    prepared_dicom = _geometry_tensor(prepared_dicom_geometry, "prepared DICOM geometry")
    prepared_resolution = _geometry_tensor(prepared_group_resolution, "prepared group resolution")
    if prepared_dicom.ndim != 2 or prepared_dicom.shape[1] != 6:
        raise ValueError("prepared DICOM geometry must have shape [G,6].")
    group_count = int(prepared_dicom.shape[0])
    if prepared_resolution.shape != (group_count, 3):
        raise ValueError("prepared group resolution must have shape [G,3] matching the prepared DICOM geometry group count.")

    missing = {GEOMETRY_CHECKPOINT_KEY, "group_axisangle_dicom_physical", "group_resolution_xyz_mm"}.difference(checkpoint)
    if missing:
        raise ValueError(f"geometry checkpoint missing required field(s): {sorted(missing)}")
    post = _geometry_tensor(checkpoint[GEOMETRY_CHECKPOINT_KEY], GEOMETRY_CHECKPOINT_KEY)
    dicom = _geometry_tensor(checkpoint["group_axisangle_dicom_physical"], "checkpoint group_axisangle_dicom_physical")
    resolution = _geometry_tensor(checkpoint["group_resolution_xyz_mm"], "checkpoint group_resolution_xyz_mm")
    for name, value in ((GEOMETRY_CHECKPOINT_KEY, post), ("group_axisangle_dicom_physical", dicom), ("group_resolution_xyz_mm", resolution)):
        if value.shape[0] != group_count:
            raise ValueError(f"geometry checkpoint group count mismatch for {name}: checkpoint {value.shape[0]} != prepared {group_count}.")
    if post.shape != (group_count, 6):
        raise ValueError(f"{GEOMETRY_CHECKPOINT_KEY} must have shape [G,6], got {tuple(post.shape)}.")
    if dicom.shape != (group_count, 6):
        raise ValueError(f"checkpoint group_axisangle_dicom_physical must have shape [G,6], got {tuple(dicom.shape)}.")
    if resolution.shape != (group_count, 3):
        raise ValueError(f"checkpoint group_resolution_xyz_mm must have shape [G,3], got {tuple(resolution.shape)}.")

    dicom_max_abs_diff = _compare_geometry(dicom, prepared_dicom, "DICOM geometry", tolerance=tolerance)
    resolution_max_abs_diff = _compare_geometry(resolution, prepared_resolution, "group resolution", tolerance=tolerance)

    training_space = checkpoint.get("training_space")
    if not isinstance(training_space, Mapping):
        raise ValueError("geometry checkpoint training_space mapping is required for internal consistency validation.")
    training_required = {"group_axisangle_init_physical", "group_axisangle_dicom_physical", "group_resolution_xyz_mm"}
    missing_training = training_required.difference(training_space)
    if missing_training:
        raise ValueError(f"geometry checkpoint training_space missing required field(s): {sorted(missing_training)}")
    training_init = _geometry_tensor(training_space["group_axisangle_init_physical"], "training_space group_axisangle_init_physical")
    training_dicom = _geometry_tensor(training_space["group_axisangle_dicom_physical"], "training_space group_axisangle_dicom_physical")
    training_resolution = _geometry_tensor(training_space["group_resolution_xyz_mm"], "training_space group_resolution_xyz_mm")
    _compare_geometry(training_init, post, "training_space group_axisangle_init_physical", tolerance=tolerance)
    _compare_geometry(training_dicom, dicom, "training_space group_axisangle_dicom_physical", tolerance=tolerance)
    _compare_geometry(training_resolution, resolution, "training_space group_resolution_xyz_mm", tolerance=tolerance)
    if "group_axisangle_init_physical" in checkpoint:
        checkpoint_init = _geometry_tensor(checkpoint["group_axisangle_init_physical"], "checkpoint group_axisangle_init_physical")
        _compare_geometry(checkpoint_init, post, "checkpoint group_axisangle_init_physical", tolerance=tolerance)

    provenance = {
        "path": str(checkpoint_path),
        "sha256": _sha256_file(checkpoint_path),
        "source": f"frozen Q002 checkpoint: {GEOMETRY_CHECKPOINT_KEY}",
        "selected_key": GEOMETRY_CHECKPOINT_KEY,
        "group_count": group_count,
        "post_hb1_geometry_sha256": _canonical_cpu_numeric_sha256(post),
        "dicom_geometry_sha256": _canonical_cpu_numeric_sha256(dicom),
        "prepared_dicom_geometry_sha256": _canonical_cpu_numeric_sha256(prepared_dicom),
        "group_resolution_sha256": _canonical_cpu_numeric_sha256(resolution),
        "prepared_group_resolution_sha256": _canonical_cpu_numeric_sha256(prepared_resolution),
        "dicom_geometry_max_abs_diff": dicom_max_abs_diff,
        "group_resolution_max_abs_diff": resolution_max_abs_diff,
        "geometry_abs_tolerance": tolerance,
        "internal_training_space_consistency": True,
        "trained_pose_key_not_used": "trained_axisangle_train",
    }
    return post.clone(), provenance


def stack_initialization_provenance(stack_initialization: Any) -> dict[str, Any]:
    """Record the existing default HB1 initialization without changing its geometry."""

    post = torch.as_tensor(stack_initialization.post_stack_init_axisangle_physical)
    if post.ndim != 2 or post.shape[1] != 6:
        raise ValueError("post-stack-init geometry must be [G,6] axis-angle poses.")
    return {
        "enabled": True,
        "args_registration": None,
        "source": "existing initialize_group_poses_from_hb1 defaults",
        "post_stack_init_axisangle_physical_sha256": _canonical_cpu_numeric_sha256(post),
        "stack_pose_records": list(stack_initialization.stack_pose_records),
    }


def _git_provenance() -> dict[str, str | None]:
    def query(*command: str) -> str | None:
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"git_sha": query("git", "rev-parse", "HEAD"), "branch": query("git", "branch", "--show-current")}


def _load_timing_only_parameters(path: str | Path, expected_anchors: int, expected_anchor_order_sha256: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"timing-only parameter NPZ does not exist: {source}")
    with np.load(source, allow_pickle=False) as payload:
        required = {"t1_ms", "t2_ms", "b1", "amplitude", "anchor_order_sha256"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"timing-only parameter NPZ lacks: {sorted(missing)}")
        parameters = {name: np.asarray(payload[name]) for name in ("t1_ms", "t2_ms", "b1", "amplitude")}
        identity = np.asarray(payload["anchor_order_sha256"]).item()
    if any(value.shape != (expected_anchors,) for value in parameters.values()):
        raise ValueError(f"timing-only parameters must align with all {expected_anchors} joint anchors in joint-dataset order.")
    if isinstance(identity, bytes):
        identity = identity.decode("ascii")
    if identity != expected_anchor_order_sha256:
        raise ValueError("timing-only anchor_order_sha256 does not match the current JointAnchorDataset order.")
    parameters["source"] = str(source)
    parameters["sha256"] = _sha256_file(source)
    return parameters


def run_from_args(args: argparse.Namespace) -> dict[str, Path]:
    """Load frozen checkpoint geometry and run the diagnostic-only audit CLI."""

    from ismrm_v2.proposed.q002.contracts import cropped_observation_paths
    from ismrm_v2.proposed.q002.joint_dataset import JointAnchorDataset
    from ismrm_v2.proposed.quantitative_point_dataset import QuantPointDataset

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
    inputs = cropped_observation_paths(args.prepared_root, args.subject)
    observation_paths = dict(zip(OBSERVATION_STACKS, inputs, strict=True))
    if not _is_l100_case(args.case):
        raise ValueError("v2.2 Modification 1 preflight currently requires --case L100.")
    scalar_dataset = QuantPointDataset(inputs, device=device)
    joint_dataset = JointAnchorDataset(inputs, scalar_dataset)
    group_axisangle_physical, geometry_checkpoint = load_geometry_checkpoint(
        args.geometry_checkpoint,
        scalar_dataset.group_axisangle_init,
        scalar_dataset.group_resolution_xyz_mm,
    )
    decoder = None
    if args.signal_decoder_checkpoint:
        if not args.protocol_yaml:
            raise ValueError("--signal-decoder-checkpoint requires --protocol-yaml.")
        from ismrm_v2.proposed.signal_decoder.checkpoint_loader import load_frozen_mlp_decoder_for_benchmark
        decoder = load_frozen_mlp_decoder_for_benchmark(args.signal_decoder_checkpoint, scalar_dataset, args.protocol_yaml, device=device)
    elif args.protocol_yaml:
        raise ValueError("--protocol-yaml requires --signal-decoder-checkpoint.")
    anchor_order_sha256 = _joint_anchor_order_sha256(joint_dataset)
    parameters = _load_timing_only_parameters(args.timing_only_parameters_npz, int(joint_dataset.xyz.shape[0]), anchor_order_sha256) if args.timing_only_parameters_npz else None
    if parameters is not None and decoder is None:
        raise ValueError("--timing-only-parameters-npz requires --signal-decoder-checkpoint.")
    sparse_manifest = resolve_preflight_sparse_manifest(
        args.sparse_manifest,
        subject=args.subject,
        requested_case=args.case,
        observation_paths=observation_paths,
    )
    provenance = {
        **_git_provenance(), "prepared_root": str(Path(args.prepared_root).resolve()),
        "prepared_inputs": {str(path): _sha256_file(path) for path in inputs},
        "sparse_manifest": sparse_manifest, "device": str(device),
        "python_version": sys.version.split()[0], "numpy_version": np.__version__, "torch_version": torch.__version__,
        "timing_only_decoder_checkpoint": str(args.signal_decoder_checkpoint) if args.signal_decoder_checkpoint else None,
        "joint_anchor_order_sha256": anchor_order_sha256,
        "timing_only_parameters": {"path": parameters["source"], "sha256": parameters["sha256"], "anchor_order_sha256": anchor_order_sha256} if parameters else None,
        "timing_only_parameter_source": "external_anchor_aligned_npz" if parameters else None,
        "geometry_checkpoint": geometry_checkpoint,
    }
    result = run_cross_view_reliability_audit(
        joint_dataset,
        group_axisangle_physical=group_axisangle_physical,
        subject=args.subject,
        case_id=args.case,
        geometry_source=geometry_checkpoint["source"],
        timing_only_decoder=decoder,
        timing_only_parameters=parameters,
        provenance=provenance,
    )
    return write_audit_outputs(result, args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="CYJ", help="Subject identifier under the prepared root.")
    parser.add_argument("--case", required=True, help="Case identifier recorded in every output artifact.")
    parser.add_argument("--prepared-root", required=True, help="Existing cropped prepared-observations root; never a v2.1 output root.")
    parser.add_argument("--geometry-checkpoint", required=True, help="Frozen Q002 checkpoint containing group_axisangle_post_stack_init_physical; loaded on CPU without model construction.")
    parser.add_argument("--sparse-manifest", help="Optional sparse-manifest provenance record; it does not change the required SAX/2CH/4CH inputs.")
    parser.add_argument("--output-dir", required=True, help="Required new or empty v2.2 output directory.")
    parser.add_argument("--device", default="cpu", help="Torch device; local validation uses cpu, server may use cuda:0.")
    parser.add_argument("--signal-decoder-checkpoint", help="Optional compatible frozen signal-decoder checkpoint for timing-only simulation.")
    parser.add_argument("--protocol-yaml", help="Protocol YAML required with --signal-decoder-checkpoint.")
    parser.add_argument("--timing-only-parameters-npz", help="Optional NPZ with anchor-aligned t1_ms/t2_ms/b1/amplitude [N] and required anchor_order_sha256; N is the full current JointAnchorDataset order.")
    return parser


def main() -> None:
    paths = run_from_args(build_parser().parse_args())
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
