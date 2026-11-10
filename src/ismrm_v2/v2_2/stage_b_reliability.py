"""Detached Stage-B cross-view reliability for the v2.2 Q002 route.

The module intentionally contains the v2.2 policy boundary: raw observed
cosine discrepancy, fixed post-HB1 correspondence, neutral invalid support,
and per-sampled-stack normalization.  It does not alter the generic Q002
objective or any model parameters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ismrm_v2.v2_2.preflight.cross_view_reliability_audit import _dataset_arrays, _joint_anchor_order_sha256
from ismrm_v2.v2_2.preflight.discrepancy import (
    fit_case_joint_robust_normalization,
    normalized_l1_discrepancy,
    raw_cosine_discrepancy,
)
from ismrm_v2.v2_2.preflight.reference_sampling import build_slice_stack_reference, sample_slice_stack


RELIABILITY_SCHEMA = "ismrm_v2_2/stage_b_cross_view_reliability/v1"
DEFAULT_RELIABILITY = {
    "enabled": False,
    "discrepancy": "raw_cosine",
    "reference_stack": "sax",
    "stage": "B_only",
    "gate_quantile": 0.90,
    "gate_power": 3.0,
    "gate_epsilon": 1.0e-6,
    "tau_min": 0.05,
    "w_min_start": 0.15,
    "w_min_end": 0.0,
    "invalid_reference_policy": "neutral",
}


@dataclass(frozen=True)
class ReliabilityConfig:
    enabled: bool = False
    discrepancy: str = "raw_cosine"
    reference_stack: str = "sax"
    stage: str = "B_only"
    gate_quantile: float = 0.90
    gate_power: float = 3.0
    gate_epsilon: float = 1.0e-6
    tau_min: float = 0.05
    w_min_start: float = 0.15
    w_min_end: float = 0.0
    invalid_reference_policy: str = "neutral"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ReliabilityConfig":
        payload = dict(DEFAULT_RELIABILITY)
        if value is not None:
            payload.update(dict(value))
        result = cls(
            enabled=bool(payload["enabled"]),
            discrepancy=str(payload["discrepancy"]),
            reference_stack=str(payload["reference_stack"]),
            stage=str(payload["stage"]),
            gate_quantile=float(payload["gate_quantile"]),
            gate_power=float(payload["gate_power"]),
            gate_epsilon=float(payload["gate_epsilon"]),
            tau_min=float(payload["tau_min"]),
            w_min_start=float(payload["w_min_start"]),
            w_min_end=float(payload["w_min_end"]),
            invalid_reference_policy=str(payload["invalid_reference_policy"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.discrepancy not in {"raw_cosine", "normalized_l1"}:
            raise ValueError("v2.2 reliability discrepancy must be raw_cosine or normalized_l1")
        if self.reference_stack != "sax":
            raise ValueError("v2.2 reliability reference_stack is frozen to sax")
        if self.stage != "B_only":
            raise ValueError("v2.2 reliability stage is frozen to B_only")
        if not 0.0 < self.gate_quantile < 1.0 or self.gate_power <= 0.0 or self.gate_epsilon <= 0.0:
            raise ValueError("invalid v2.2 reliability gate parameters")
        if self.tau_min < 0.0 or not 0.0 <= self.w_min_start <= 1.0 or not 0.0 <= self.w_min_end <= 1.0:
            raise ValueError("invalid v2.2 reliability floor parameters")
        if self.invalid_reference_policy != "neutral":
            raise ValueError("invalid_reference_policy must be neutral")

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "discrepancy": self.discrepancy,
            "reference_stack": self.reference_stack,
            "stage": self.stage,
            "gate_quantile": self.gate_quantile,
            "gate_power": self.gate_power,
            "gate_epsilon": self.gate_epsilon,
            "tau_min": self.tau_min,
            "w_min_start": self.w_min_start,
            "w_min_end": self.w_min_end,
            "invalid_reference_policy": self.invalid_reference_policy,
        }


def stage_b_progress(iteration: int, iterations: int) -> float:
    """Return local Stage-B progress, with iteration zero at the first step."""

    if iterations < 1 or iteration < 0 or iteration >= iterations:
        raise ValueError("Stage-B iteration must lie in [0, iterations).")
    return 1.0 if iterations == 1 else float(iteration) / float(iterations - 1)


def gate_floor(progress: float, config: ReliabilityConfig | Mapping[str, Any]) -> float:
    cfg = config if isinstance(config, ReliabilityConfig) else ReliabilityConfig.from_mapping(config)
    if not 0.0 <= progress <= 1.0:
        raise ValueError("Stage-B progress must lie in [0,1].")
    return float(cfg.w_min_start + (cfg.w_min_end - cfg.w_min_start) * progress)


def tau_from_discrepancy(discrepancy: np.ndarray, valid: np.ndarray, *, config: ReliabilityConfig | Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    cfg = config if isinstance(config, ReliabilityConfig) else ReliabilityConfig.from_mapping(config)
    values = np.asarray(discrepancy, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or mask.shape != values.shape:
        raise ValueError("discrepancy and valid must be matching vectors")
    finite = mask & np.isfinite(values)
    q = float(np.quantile(values[finite], cfg.gate_quantile)) if finite.any() else None
    selected = values[finite]
    tau = max(float(q) if q is not None else 0.0, cfg.tau_min)
    return tau, {"count": int(values.size), "valid_count": int(finite.sum()), "valid_fraction": float(finite.mean()) if values.size else 0.0, "q90": q, "tau": tau, "mean": float(selected.mean()) if selected.size else None, "std": float(selected.std()) if selected.size else None, "median": float(np.quantile(selected, .5)) if selected.size else None, "min": float(selected.min()) if selected.size else None, "max": float(selected.max()) if selected.size else None}


def compute_reliability_gate(
    discrepancy: torch.Tensor,
    reference_valid: torch.Tensor,
    stack_idx: torch.Tensor,
    tau_by_stack: Mapping[int, float],
    progress: float,
    config: ReliabilityConfig | Mapping[str, Any],
) -> torch.Tensor:
    """Compute one detached gate for each complete 10-D anchor fingerprint."""

    cfg = config if isinstance(config, ReliabilityConfig) else ReliabilityConfig.from_mapping(config)
    cfg.validate()
    d = torch.as_tensor(discrepancy).detach()
    valid = torch.as_tensor(reference_valid, dtype=torch.bool, device=d.device).detach()
    stacks = torch.as_tensor(stack_idx, dtype=torch.long, device=d.device).detach()
    if d.ndim != 1 or valid.shape != d.shape or stacks.shape != d.shape:
        raise ValueError("reliability inputs must be matching one-dimensional tensors")
    floor = d.new_tensor(gate_floor(progress, cfg))
    gate = torch.ones_like(d)
    with torch.no_grad():
        for stack, tau in tau_by_stack.items():
            stack_mask = (stacks == int(stack)) & valid & torch.isfinite(d)
            if int(stack) == 0 or not bool(stack_mask.any()):
                continue
            tau_value = d.new_tensor(max(float(tau), cfg.tau_min))
            ratio = d[stack_mask].clamp_min(0.0) / (tau_value + cfg.gate_epsilon)
            gate[stack_mask] = floor + (1.0 - floor) / (1.0 + ratio.pow(cfg.gate_power))
    return gate.detach()


def normalize_reliability_gate(gate: torch.Tensor, stack_idx: torch.Tensor) -> torch.Tensor:
    """Normalize gates independently for each stack represented in a batch."""

    values = torch.as_tensor(gate).detach()
    stacks = torch.as_tensor(stack_idx, dtype=torch.long, device=values.device)
    if values.ndim != 1 or stacks.shape != values.shape or not torch.isfinite(values).all():
        raise ValueError("gate and stack_idx must be matching finite vectors")
    normalized = torch.empty_like(values)
    with torch.no_grad():
        for stack in stacks.unique(sorted=True).tolist():
            mask = stacks == int(stack)
            if not bool(mask.any()):
                raise ValueError("empty stack encountered while normalizing reliability")
            mean = values[mask].mean()
            if not torch.isfinite(mean) or float(mean) <= 0.0:
                raise ValueError("reliability gate mean must be positive")
            normalized[mask] = values[mask] / mean
    return normalized.detach()


def _stack_weights(stack_idx: torch.Tensor, stack_weights: Mapping[int, float], *, dtype: torch.dtype, enabled: bool) -> torch.Tensor:
    if not enabled:
        return torch.ones(stack_idx.shape, dtype=dtype, device=stack_idx.device)
    values = torch.full(stack_idx.shape, float("nan"), dtype=dtype, device=stack_idx.device)
    for stack, weight in stack_weights.items():
        values[stack_idx == int(stack)] = float(weight)
    if not torch.isfinite(values).all():
        raise ValueError("stack weights do not cover every sampled anchor")
    return values


def reliability_weighted_joint_mse(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    stack_idx: torch.Tensor,
    stack_weights: Mapping[int, float],
    gate: torch.Tensor | None,
    *,
    enabled: bool,
    stage: str,
    stack_balance_enabled: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the frozen MSE or normalized Stage-B reliability MSE."""

    if prediction.shape != observed.shape or prediction.ndim != 2 or prediction.shape[1] != 10:
        raise ValueError("prediction/observed must be matching [B,10] tensors")
    stacks = torch.as_tensor(stack_idx, dtype=torch.long, device=prediction.device)
    weights = _stack_weights(stacks, stack_weights, dtype=prediction.dtype, enabled=stack_balance_enabled)
    per_anchor = (prediction - observed).pow(2).mean(dim=1)
    base = (weights * per_anchor).mean()
    diagnostics: dict[str, Any] = {"reliability_enabled": bool(enabled), "reliability_active": False, "reliability_gate": None, "reliability_multiplier": None, "reliability_w_min": None}
    if not enabled or stage != "B" or gate is None:
        return base, diagnostics
    gate_tensor = torch.as_tensor(gate, dtype=prediction.dtype, device=prediction.device).detach()
    if gate_tensor.shape != stacks.shape:
        raise ValueError("reliability gate must align with sampled anchors")
    multiplier = normalize_reliability_gate(gate_tensor, stacks)
    diagnostics.update({"reliability_active": True, "reliability_gate": gate_tensor, "reliability_multiplier": multiplier})
    return (weights * multiplier * per_anchor).mean(), diagnostics


@dataclass
class ReliabilityCache:
    """Compact anchor-indexed discrepancy cache plus immutable provenance."""

    anchor_order_sha256: str
    discrepancy: np.ndarray
    reference_valid: np.ndarray
    stack_idx: np.ndarray
    tau_by_stack: dict[int, float]
    provenance: dict[str, Any] = field(default_factory=dict)
    qc: dict[str, Any] = field(default_factory=dict)
    normalization: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.discrepancy = np.asarray(self.discrepancy, dtype=np.float64)
        self.reference_valid = np.asarray(self.reference_valid, dtype=bool)
        self.stack_idx = np.asarray(self.stack_idx, dtype=np.int64)
        if self.discrepancy.ndim != 1 or self.reference_valid.shape != self.discrepancy.shape or self.stack_idx.shape != self.discrepancy.shape:
            raise ValueError("reliability cache arrays must be matching one-dimensional arrays")
        if not isinstance(self.anchor_order_sha256, str) or len(self.anchor_order_sha256) != 64:
            raise ValueError("reliability cache requires a SHA256 anchor identity")

    @property
    def anchor_count(self) -> int:
        return int(self.discrepancy.size)

    def validate_anchor_order(self, anchor_order_sha256: str) -> None:
        if str(anchor_order_sha256) != self.anchor_order_sha256:
            raise ValueError("reliability cache anchor_order_sha256 disagrees with current JointAnchorDataset")

    def gate_for_indices(self, indices: torch.Tensor, *, progress: float, config: ReliabilityConfig | Mapping[str, Any]) -> torch.Tensor:
        values = torch.as_tensor(self.discrepancy[torch.as_tensor(indices).detach().cpu().numpy()], dtype=torch.float32, device=indices.device)
        valid = torch.as_tensor(self.reference_valid[torch.as_tensor(indices).detach().cpu().numpy()], dtype=torch.bool, device=indices.device)
        stacks = torch.as_tensor(self.stack_idx[torch.as_tensor(indices).detach().cpu().numpy()], dtype=torch.long, device=indices.device)
        return compute_reliability_gate(values, valid, stacks, self.tau_by_stack, progress, config)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": RELIABILITY_SCHEMA,
            "anchor_order_sha256": self.anchor_order_sha256,
            "anchor_count": self.anchor_count,
            "tau_by_stack": {str(key): float(value) for key, value in self.tau_by_stack.items()},
            "provenance": self.provenance,
            "qc": self.qc,
            "normalization": self.normalization,
        }

    def write(self, directory: str | Path) -> dict[str, str]:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        array_path = root / "reliability_cache.npz"
        manifest_path = root / "reliability_manifest.json"
        np.savez_compressed(array_path, discrepancy=self.discrepancy, reference_valid=self.reference_valid, stack_idx=self.stack_idx)
        payload = self.manifest()
        payload["cache_npz"] = array_path.name
        payload["cache_npz_sha256"] = _sha256_file(array_path)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"cache_npz": str(array_path), "manifest": str(manifest_path)}

    @classmethod
    def load(cls, directory: str | Path) -> "ReliabilityCache":
        root = Path(directory)
        manifest = json.loads((root / "reliability_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != RELIABILITY_SCHEMA:
            raise ValueError("unsupported reliability cache schema")
        array_path = root / str(manifest.get("cache_npz", "reliability_cache.npz"))
        if manifest.get("cache_npz_sha256") and _sha256_file(array_path) != manifest["cache_npz_sha256"]:
            raise ValueError("reliability cache NPZ SHA256 mismatch")
        with np.load(array_path, allow_pickle=False) as data:
            return cls(str(manifest["anchor_order_sha256"]), data["discrepancy"], data["reference_valid"], data["stack_idx"], {int(key): float(value) for key, value in manifest.get("tau_by_stack", {}).items()}, dict(manifest.get("provenance", {})), dict(manifest.get("qc", {})), manifest.get("normalization"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry_sha256(value: Any) -> str:
    array = np.asarray(torch.as_tensor(value).detach().cpu(), dtype=np.float64)
    return hashlib.sha256(np.asarray(array.shape, dtype="<i8").tobytes() + np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def build_reliability_cache(
    joint_dataset: Any,
    *,
    group_axisangle_post_stack_init_physical: torch.Tensor,
    sax_observation_path: str | Path | None,
    config: ReliabilityConfig | Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    cache_directory: str | Path | None = None,
) -> ReliabilityCache:
    """Precompute same-world observed SAX support using fixed post-HB1 geometry."""

    cfg = config if isinstance(config, ReliabilityConfig) else ReliabilityConfig.from_mapping(config)
    cfg.validate()
    arrays = _dataset_arrays(joint_dataset, torch.as_tensor(group_axisangle_post_stack_init_physical).detach().cpu())
    stacks = np.asarray(arrays["stack_idx"], dtype=np.int64)
    discrepancy = np.full(stacks.shape, np.nan, dtype=np.float64)
    reference_valid = np.zeros(stacks.shape, dtype=bool)
    tau_by_stack: dict[int, float] = {}
    qc: dict[str, Any] = {"anchor_count": int(stacks.size), "reference_stack": cfg.reference_stack, "target_stacks": {}}
    normalization_payload: dict[str, Any] | None = None
    target_stacks = [int(stack) for stack in np.unique(stacks) if int(stack) != 0] if cfg.enabled else []
    if target_stacks:
        if sax_observation_path is None or not Path(sax_observation_path).is_file():
            raise ValueError("reliability-enabled target stacks require an admitted SAX observation path")
        reference = build_slice_stack_reference(str(sax_observation_path), np.asarray(torch.as_tensor(group_axisangle_post_stack_init_physical).detach().cpu()))
        if cfg.discrepancy == "normalized_l1":
            stack_names = {0: "sax", 1: "2ch", 2: "4ch"}
            observed_by_stack = {
                stack_names[int(stack)]: arrays["observed"][stacks == int(stack)]
                for stack in np.unique(stacks)
                if int(stack) in stack_names and np.count_nonzero(stacks == int(stack))
            }
            normalization = fit_case_joint_robust_normalization(observed_by_stack)
            normalization_payload = normalization.as_dict()
            qc["normalization"] = normalization_payload
        for stack in target_stacks:
            indices = np.flatnonzero(stacks == stack)
            sampled, valid, support = sample_slice_stack(reference, arrays["physical_xyz"][indices])
            if cfg.discrepancy == "raw_cosine":
                raw, raw_valid = raw_cosine_discrepancy(arrays["observed"][indices], sampled, epsilon=1.0e-8)
            else:
                raw, raw_valid = normalized_l1_discrepancy(arrays["observed"][indices], sampled, normalization)
            valid = np.asarray(valid, dtype=bool) & np.asarray(raw_valid, dtype=bool)
            discrepancy[indices] = raw
            reference_valid[indices] = valid
            tau, stats = tau_from_discrepancy(raw, valid, config=cfg)
            tau_by_stack[stack] = tau
            qc["target_stacks"][str(stack)] = {**stats, "invalid_reference_fraction": float((~valid).mean()) if valid.size else 0.0, "support_valid_fraction": float(np.asarray(support["in_plane_valid"], dtype=bool).mean()) if valid.size else 0.0}
    for stack in np.unique(stacks):
        if int(stack) == 0:
            reference_valid[stacks == stack] = True
    payload = dict(provenance or {})
    source_hashes: dict[str, str] = dict(payload.get("observation_sha256", {})) if isinstance(payload.get("observation_sha256"), Mapping) else {}
    if sax_observation_path is not None and Path(sax_observation_path).is_file():
        source_hashes["sax_observation"] = _sha256_file(Path(sax_observation_path))
    payload.update({"anchor_order_sha256": _joint_anchor_order_sha256(joint_dataset), "geometry_key": "group_axisangle_post_stack_init_physical", "post_hb1_geometry_sha256": _geometry_sha256(group_axisangle_post_stack_init_physical), "observation_sha256": source_hashes, "scientific_parameters": cfg.as_dict()})
    cache = ReliabilityCache(_joint_anchor_order_sha256(joint_dataset), discrepancy, reference_valid, stacks, tau_by_stack, payload, qc, normalization_payload)
    if cache_directory is not None:
        cache.write(cache_directory)
    return cache
