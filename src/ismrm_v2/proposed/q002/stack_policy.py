"""Resolve joint-anchor loss weights without changing historical scalar weighting."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def resolve_stack_policy(scalar: Any, joint: Any, config: Mapping[str, Any]) -> tuple[dict[int, float], dict[str, Any]]:
    enabled = bool(config.get("enabled", True))
    policy = config.get("policy", "equal_total_stack" if enabled else "off")
    if policy not in ("off", "equal_total_stack", "slice_proportional_total") or enabled != (policy != "off"):
        raise ValueError("inconsistent or unknown stack balance policy")
    historical = {int(k): float(v) for k, v in scalar.validate_balanced_samples().items()}
    counts = {int(s): int((joint.stack_idx == s).sum().item()) for s in joint.stack_idx.unique().tolist()}
    groups = Counter(int(group["stack_idx"]) for group in joint.qc["groups"].values())
    if set(counts) != set(historical) or set(counts) != set(groups) or any(v <= 0 for v in counts.values()):
        raise ValueError("stack support differs between scalar, joint and group datasets")
    if policy == "off":
        weights = {s: 1.0 for s in counts}
        source, formula, normalization = "constant", "1", "none"
    elif policy == "equal_total_stack":
        weights = historical
        source, formula, normalization = "scalar_dataset.validate_balanced_samples", "total_scalar / (number_of_stacks * scalar_count_s)", "scalar-count weighted mean is 1"
    else:
        # Preserve the historical joint-anchor-weighted average, not its generally
        # different scalar-weighted average, so data loss has comparable scale.
        scale = sum(counts[s] * historical[s] for s in counts) / sum(groups.values())
        weights = {s: scale * groups[s] / counts[s] for s in counts}
        source, formula, normalization = "joint_anchor_count and retained_group_count", "w_s = scale * retained_groups_s / joint_count_s", "scale = sum(joint_count_s * historical_weight_s) / sum(retained_groups_s)"
    masses = {s: counts[s] * weights[s] for s in counts}
    total = sum(masses.values())
    qc = {"stack_balance_enabled": enabled, "stack_balance_policy": policy, "stack_weight_source": source, "stack_weight_formula": formula, "stack_weight_normalization": normalization, "retained_slice_count_by_stack": dict(groups), "joint_anchor_count_by_stack": counts, "resolved_stack_weights": weights, "expected_weighted_fraction_by_stack": {s: masses[s] / total for s in counts}}
    return weights, qc
