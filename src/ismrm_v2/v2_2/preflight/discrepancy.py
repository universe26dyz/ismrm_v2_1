"""Auditable 10-D discrepancy and SRPAN-style gate primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class JointRobustNormalization:
    low: np.ndarray
    high: np.ndarray
    low_quantile: float
    high_quantile: float
    pooled_row_count: int
    pooled_stack_counts: dict[str, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"low": self.low.tolist(), "high": self.high.tolist(), "low_quantile": self.low_quantile, "high_quantile": self.high_quantile, "pooled_row_count": self.pooled_row_count, "pooled_stack_counts": dict(self.pooled_stack_counts or {}), "range_definition": "per-weight pooled actual active case anchors; no clipping"}


def fit_joint_robust_normalization(target: np.ndarray, reference: np.ndarray, *, low_quantile: float = 0.005, high_quantile: float = 0.995) -> JointRobustNormalization:
    left, right = np.asarray(target, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != (10,) or right.shape[1:] != (10,):
        raise ValueError("target and reference must be [N,10].")
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("normalization quantiles must satisfy 0 <= low < high <= 1.")
    pooled = np.vstack((left, right))
    if pooled.shape[0] == 0 or not np.isfinite(pooled).all():
        raise ValueError("normalization inputs must be non-empty and finite.")
    low, high = np.quantile(pooled, [low_quantile, high_quantile], axis=0)
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
        raise ValueError("per-weight robust normalization range is non-finite or degenerate.")
    return JointRobustNormalization(low, high, low_quantile, high_quantile, int(pooled.shape[0]))


def fit_case_joint_robust_normalization(stack_values: dict[str, np.ndarray], *, low_quantile: float = .005, high_quantile: float = .995) -> JointRobustNormalization:
    """Fit one shared normalization from the actual observed active anchors."""
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in stack_values.items() if np.asarray(values).size}
    if not arrays:
        raise ValueError("case normalization requires at least one active observed stack")
    if any(value.ndim != 2 or value.shape[1] != 10 or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("active observed stack fingerprints must be finite [N,10]")
    pooled = np.vstack(list(arrays.values()))
    fitted = fit_joint_robust_normalization(pooled, pooled[:0], low_quantile=low_quantile, high_quantile=high_quantile)
    return JointRobustNormalization(fitted.low, fitted.high, fitted.low_quantile, fitted.high_quantile, fitted.pooled_row_count, {name: int(value.shape[0]) for name, value in arrays.items()})


def _normalized(values: np.ndarray, stats: JointRobustNormalization) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (array - stats.low) / (stats.high - stats.low)


def normalized_l1_discrepancy(target: np.ndarray, reference: np.ndarray, stats: JointRobustNormalization) -> tuple[np.ndarray, np.ndarray]:
    left, right = np.asarray(target, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 10:
        raise ValueError("target and reference must be matching [N,10].")
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    result = np.full(left.shape[0], np.nan, dtype=np.float64)
    result[valid] = np.abs(_normalized(left[valid], stats) - _normalized(right[valid], stats)).mean(axis=1)
    return result, valid


def raw_cosine_discrepancy(target: np.ndarray, reference: np.ndarray, *, epsilon: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    left, right = np.asarray(target, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 10:
        raise ValueError("target and reference must be matching [N,10].")
    if epsilon <= 0:
        raise ValueError("cosine epsilon must be positive.")
    norms = np.linalg.norm(left, axis=1), np.linalg.norm(right, axis=1)
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1) & (norms[0] >= epsilon) & (norms[1] >= epsilon)
    result = np.full(left.shape[0], np.nan, dtype=np.float64)
    result[valid] = 1.0 - np.sum(left[valid] * right[valid], axis=1) / (norms[0][valid] * norms[1][valid] + epsilon)
    return result, valid


def srpan_gate(discrepancy: np.ndarray, *, tau: float, progress: float, power: float = 3.0, epsilon: float = 1e-6, w_min_start: float = 0.15) -> np.ndarray:
    values = np.asarray(discrepancy, dtype=np.float64)
    if tau < 0 or epsilon <= 0 or power <= 0 or not 0 <= progress <= 1 or not 0 <= w_min_start <= 1:
        raise ValueError("invalid SRPAN gate parameters.")
    floor = w_min_start * (1.0 - progress)
    return floor + (1.0 - floor) / (1.0 + (np.maximum(values, 0.0) / (tau + epsilon)) ** power)


def gate_dry_run(discrepancy: np.ndarray, valid: np.ndarray, *, gate_quantile: float = .90, gate_power: float = 3.0, gate_epsilon: float = 1e-6, srpan_tau_min: float = .05) -> dict[str, Any]:
    values, mask = np.asarray(discrepancy, dtype=np.float64), np.asarray(valid, dtype=bool)
    if values.ndim != 1 or mask.shape != values.shape:
        raise ValueError("discrepancy and valid must be matching vectors.")
    if not 0 < gate_quantile < 1:
        raise ValueError("gate quantile must lie in (0,1).")
    finite = mask & np.isfinite(values)
    q90 = float(np.quantile(values[finite], gate_quantile)) if finite.any() else None
    result: dict[str, Any] = {"count": int(values.size), "valid_count": int(finite.sum()), "invalid_reference_fraction": float((~mask).mean()) if values.size else None, "invalid_reference_policy_report": {"neutral_if_invalid": "g=1 for reporting only", "srpan_exclude_if_invalid": "g=0/excluded for reporting only"}, "gate_quantile": gate_quantile, "gate_power": gate_power, "gate_epsilon": gate_epsilon, "srpan_tau_min": srpan_tau_min, "tau_q90": q90, "tau_srpan_literal": max(q90, srpan_tau_min) if q90 is not None else None, "progress": {}}
    for progress in (0.0, .25, .5, .75, 1.0):
        progress_payload: dict[str, Any] = {}
        for name, tau in (("q90_only", q90), ("srpan_literal", result["tau_srpan_literal"])):
            if tau is None:
                progress_payload[name] = {"status": "not_run"}
                continue
            gate = srpan_gate(values[finite], tau=tau, progress=progress, power=gate_power, epsilon=gate_epsilon)
            progress_payload[name] = {"status": "run", "w_min": .15 * (1 - progress), "mean": float(gate.mean()), "std": float(gate.std()), "quantiles": {f"p{q:02d}": float(np.quantile(gate, q / 100)) for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)}, "fraction_g_lt_0.1": float((gate < .1).mean()), "fraction_g_lt_0.25": float((gate < .25).mean()), "fraction_g_lt_0.5": float((gate < .5).mean()), "fraction_g_gt_0.9": float((gate > .9).mean())}
        result["progress"][str(progress)] = progress_payload
    return result
