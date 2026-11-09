from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ismrm_v2.proposed.q002.run_q002_reconstruction import FORMAL_SPARSE_EXPERIMENT_IDS, validate_formal_sparse_reconstruction_dependencies
from ismrm_v2.proposed.q002.contracts import B0_R0_EXPERIMENT_ID, B0_R1_EXPERIMENT_ID, B1_R0_EXPERIMENT_ID, B1_R1_EXPERIMENT_ID, build_q002_route_config
from ismrm_v2.proposed.q002.training import joint_vector_mse, reliability_sample_coverage
from ismrm_v2.v2_2.stage_b_reliability import (
    ReliabilityCache,
    ReliabilityConfig,
    build_reliability_cache,
    compute_reliability_gate,
    normalize_reliability_gate,
    reliability_weighted_joint_mse,
    stage_b_progress,
    tau_from_discrepancy,
)


def _inputs():
    prediction = torch.arange(40, dtype=torch.float32).reshape(4, 10) / 10
    observed = prediction + torch.tensor([[1.0] + [0.0] * 9, [0.0, 1.0] + [0.0] * 8, [0.0, 0.0, 1.0] + [0.0] * 7, [0.0, 0.0, 0.0, 1.0] + [0.0] * 6])
    stacks = torch.tensor([0, 0, 1, 1])
    return prediction, observed, stacks


def _cfg(enabled=True):
    return ReliabilityConfig.from_mapping({"enabled": enabled})


def test_r0_is_exact_frozen_joint_vector_mse():
    prediction, observed, stacks = _inputs()
    baseline = joint_vector_mse(prediction, observed, stacks, {0: 1.0, 1: 2.0})
    result, diagnostics = reliability_weighted_joint_mse(prediction, observed, stacks, {0: 1.0, 1: 2.0}, None, enabled=False, stage="B")
    assert torch.equal(result, baseline)
    assert diagnostics["reliability_active"] is False


def test_all_one_gate_matches_baseline():
    prediction, observed, stacks = _inputs()
    baseline = joint_vector_mse(prediction, observed, stacks, {0: 1.0, 1: 2.0})
    result, diagnostics = reliability_weighted_joint_mse(prediction, observed, stacks, {0: 1.0, 1: 2.0}, torch.ones(4), enabled=True, stage="B")
    assert torch.allclose(result, baseline, rtol=0, atol=0)
    assert torch.allclose(diagnostics["reliability_multiplier"], torch.ones(4))


def test_stack_normalization_preserves_each_stack_mean():
    gate = torch.tensor([1.0, .5, .2, .8, .7, .3])
    stacks = torch.tensor([0, 0, 1, 1, 2, 2])
    normalized = normalize_reliability_gate(gate, stacks)
    for stack in stacks.unique():
        assert torch.allclose(normalized[stacks == stack].mean(), torch.tensor(1.0))


def test_invalid_reference_is_neutral_and_detached():
    discrepancy = torch.tensor([0.0, 10.0], requires_grad=True)
    valid = torch.tensor([True, False])
    stacks = torch.tensor([1, 1])
    gate = compute_reliability_gate(discrepancy, valid, stacks, {1: .05}, 0.0, _cfg())
    assert gate[1].item() == 1.0
    assert not gate.requires_grad


def test_reliability_loss_detaches_gate():
    prediction, observed, stacks = _inputs()
    prediction = prediction.requires_grad_()
    gate = torch.tensor([.2, .8, .4, .6], requires_grad=True)
    loss, diagnostics = reliability_weighted_joint_mse(prediction, observed, stacks, {0: 1.0, 1: 2.0}, gate, enabled=True, stage="B")
    loss.backward()
    assert gate.grad is None
    assert not diagnostics["reliability_multiplier"].requires_grad


def test_stage_a_is_noop_and_schedule_is_local():
    prediction, observed, stacks = _inputs()
    baseline = joint_vector_mse(prediction, observed, stacks, {0: 1.0, 1: 2.0})
    result, _ = reliability_weighted_joint_mse(prediction, observed, stacks, {0: 1.0, 1: 2.0}, torch.tensor([.1, .2, .3, .4]), enabled=True, stage="A")
    assert torch.equal(result, baseline)
    progress = [stage_b_progress(i, 4000) for i in (0, 1, 3999)]
    assert progress == sorted(progress)
    assert progress[0] == 0.0 and progress[-1] == 1.0


def test_tau_uses_q90_and_floor():
    tau, stats = tau_from_discrepancy(np.array([.01, .02, .1, .2]), np.ones(4, dtype=bool), config=_cfg())
    assert stats["q90"] == pytest.approx(.17)
    assert tau == pytest.approx(.17)
    tau_floor, _ = tau_from_discrepancy(np.array([.01, .02]), np.ones(2, dtype=bool), config=_cfg())
    assert tau_floor == .05


def test_cache_anchor_identity_fails_closed(tmp_path):
    cache = ReliabilityCache("a" * 64, np.array([np.nan]), np.array([True]), np.array([0]), {}, {})
    paths = cache.write(tmp_path)
    loaded = ReliabilityCache.load(tmp_path)
    assert json.loads((tmp_path / "reliability_manifest.json").read_text())["schema"].endswith("/v1")
    assert paths["cache_npz"].endswith("reliability_cache.npz")
    with pytest.raises(ValueError, match="anchor_order_sha256"):
        loaded.validate_anchor_order("b" * 64)


def test_sax_only_reliability_is_neutral_noop():
    class SaxOnly:
        xyz = torch.zeros((2, 3))
        observed = torch.ones((2, 10))
        group_idx = torch.zeros(2, dtype=torch.long)
        stack_idx = torch.zeros(2, dtype=torch.long)
        timing = torch.zeros((2, 9))
        row_col = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)

    cache = build_reliability_cache(SaxOnly(), group_axisangle_post_stack_init_physical=torch.zeros((1, 6)), sax_observation_path=None, config=_cfg(), provenance={})
    gate = cache.gate_for_indices(torch.tensor([0, 1]), progress=0.0, config=_cfg())
    assert cache.tau_by_stack == {}
    assert torch.equal(gate, torch.ones(2))


def test_route_switches_are_independent():
    base = {"training": {"stage_a_iterations": 2000, "stage_b_iterations": 4000, "batch_size": 640, "psf_samples": 8, "seed": 20260911}, "decoder": {}}
    routes = {}
    for name in (B0_R0_EXPERIMENT_ID, B1_R0_EXPERIMENT_ID, B0_R1_EXPERIMENT_ID, B1_R1_EXPERIMENT_ID):
        routes[name] = build_q002_route_config(base, cropped_prepared_root="cropped", experiment_id=name, anchor_batch_size=640)["route"]
    assert routes[B0_R0_EXPERIMENT_ID]["stack_balance"]["enabled"] is False
    assert routes[B1_R0_EXPERIMENT_ID]["stack_balance"]["enabled"] is True
    assert routes[B0_R1_EXPERIMENT_ID]["cross_view_reliability"]["enabled"] is True
    assert routes[B1_R1_EXPERIMENT_ID]["cross_view_reliability"]["enabled"] is True


def test_all_formal_routes_use_sparse_admission_and_missing_pins_fail_closed():
    expected_formal = {"Q002S640_B6_roi_same_anchor_mse", "Q004S640_K1_B6_roi_same_anchor_mse", "Q005S640_K4_B6_roi_same_anchor_mse", B0_R0_EXPERIMENT_ID, B1_R0_EXPERIMENT_ID, B0_R1_EXPERIMENT_ID, B1_R1_EXPERIMENT_ID}
    assert expected_formal.issubset(set(FORMAL_SPARSE_EXPERIMENT_IDS))
    from argparse import Namespace
    for experiment_id in (B0_R0_EXPERIMENT_ID, B1_R0_EXPERIMENT_ID, B0_R1_EXPERIMENT_ID, B1_R1_EXPERIMENT_ID):
        args = Namespace(experiment_id=experiment_id, sparse_manifest=None)
        with pytest.raises(ValueError, match="exact sparse manifest"):
            validate_formal_sparse_reconstruction_dependencies(args)


def test_target_only_reliability_coverage_ignores_sax_and_handles_no_target():
    coverage = reliability_sample_coverage(torch.tensor([0, 0, 1, 1]), torch.tensor([True, True, True, False]))
    assert coverage["reliability_valid_fraction"] == pytest.approx(.75)
    assert coverage["reliability_invalid_reference_fraction"] == pytest.approx(.25)
    assert coverage["reliability_target_valid_fraction"] == pytest.approx(.5)
    assert coverage["reliability_target_invalid_reference_fraction"] == pytest.approx(.5)
    no_target = reliability_sample_coverage(torch.tensor([0, 0]), torch.tensor([True, True]))
    assert no_target["reliability_target_valid_fraction"] is None
    assert no_target["reliability_target_invalid_reference_fraction"] is None
