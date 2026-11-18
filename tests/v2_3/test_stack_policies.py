from types import SimpleNamespace

import pytest
import torch

from ismrm_v2.proposed.q002.contracts import build_q002_route_config
from ismrm_v2.proposed.q002.stack_policy import resolve_stack_policy


def test_stack_policies_preserve_historical_joint_scale():
    scalar = SimpleNamespace(validate_balanced_samples=lambda: {0: 0.5, 1: 1.5, 2: 3.0})
    joint = SimpleNamespace(stack_idx=torch.tensor([0] * 60 + [1] * 20 + [2] * 10), qc={"groups": {str(i): {"stack_idx": s} for i, s in enumerate([0]*12 + [1]*7 + [2]*5)}})
    equal, eq_qc = resolve_stack_policy(scalar, joint, {"enabled": True})
    assert equal == scalar.validate_balanced_samples()
    off, _ = resolve_stack_policy(scalar, joint, {"enabled": False})
    assert set(off.values()) == {1.0}
    weights, qc = resolve_stack_policy(scalar, joint, {"enabled": True, "policy": "slice_proportional_total"})
    mass = [n * weights[s] for s, n in enumerate((60, 20, 10))]
    assert [m / mass[0] for m in mass] == pytest.approx([1, 7/12, 5/12])
    assert sum(mass) == pytest.approx(sum(n * equal[s] for s, n in enumerate((60, 20, 10))))
    assert qc["expected_weighted_fraction_by_stack"][0] == pytest.approx(.5)
    assert eq_qc["stack_weight_source"] == "scalar_dataset.validate_balanced_samples"
    with pytest.raises(ValueError):
        resolve_stack_policy(scalar, joint, {"enabled": False, "policy": "slice_proportional_total"})


@pytest.mark.parametrize("name,parent,policy", [
    ("B1_R0_STACK_OFF", "B1_R0", "off"),
    ("B1_R0_STACK_SLICE", "B1_R0", "slice_proportional_total"),
    ("B1_R0_HGNLL_STACK_SLICE", "B1_R0_HGNLL", "slice_proportional_total"),
    ("B1_R0_HGNLL_DETACH_STACK_SLICE", "B1_R0_HGNLL_DETACH", "slice_proportional_total"),
])
def test_routes_inherit_parent_science(name, parent, policy):
    def route(x):
        return build_q002_route_config({"training": {}, "decoder": {}}, cropped_prepared_root="cropped", experiment_id=x, anchor_batch_size=640)
    child, old = route(name), route(parent)
    assert child["route"]["stack_balance"]["policy"] == policy
    assert child["route"]["sampling"] == old["route"]["sampling"]
    assert child["route"]["cross_view_reliability"] == old["route"]["cross_view_reliability"]
    assert child["route"].get("heteroscedastic") == old["route"].get("heteroscedastic")
    assert old["route"]["stack_balance"] == {"enabled": True}
