from __future__ import annotations

import math

import pytest
import torch

from ismrm_v2.proposed.q002.contracts import (
    B1_R0_EXPERIMENT_ID,
    B1_R0_HGNLL_EXPERIMENT_ID,
    B1_R1_EXPERIMENT_ID,
    B1_R1_HGNLL_EXPERIMENT_ID,
    build_q002_route_config,
)
from ismrm_v2.proposed.q002.run_q002_reconstruction import FORMAL_SPARSE_EXPERIMENT_IDS
from ismrm_v2.proposed.q002.training import joint_vector_mse
from ismrm_v2.v2_3.hgnll import (
    HGNLLConfig,
    HGNLL_LOG_VARIANCE_MAX,
    JointFingerprintVarianceHead,
    aggregate_log_variance,
    hgnll_objective,
)


def _base():
    return {"training": {"stage_a_iterations": 2000, "stage_b_iterations": 4000, "batch_size": 640, "psf_samples": 8, "seed": 20260911}, "decoder": {}}


def test_hgnll_config_is_explicit_and_frozen():
    cfg = HGNLLConfig.from_mapping({"enabled": True})
    assert cfg.distribution == "gaussian"
    assert cfg.output == "per_weight_log_variance"
    assert cfg.log_variance_min == -10.0
    assert cfg.log_variance_max == pytest.approx(math.log(0.35**2))
    assert cfg.initialization == "upper_bound_minus_epsilon"
    with pytest.raises(ValueError):
        HGNLLConfig.from_mapping({"enabled": True, "distribution": "laplace"})


def test_head_outputs_ten_log_variances_with_safe_initialization_and_gradients():
    cfg = HGNLLConfig.from_mapping({"enabled": True})
    head = JointFingerprintVarianceHead(4, cfg)
    latent = torch.randn(6, 4, requires_grad=True)
    output = head(latent)
    assert output.shape == (6, 10)
    assert torch.allclose(output, torch.full_like(output, HGNLL_LOG_VARIANCE_MAX - cfg.initialization_epsilon))
    output.sum().backward()
    assert latent.grad is not None
    assert head.network[-1].weight.grad is not None


def test_psf_log_mean_exp_variance_matches_definition():
    values = torch.tensor([[[0.0] * 10, [math.log(4.0)] * 10]])
    result = aggregate_log_variance(values)
    assert torch.allclose(result, torch.full((1, 10), math.log(2.5)))


def test_clamp_bounds_and_fractions_are_reported():
    cfg = HGNLLConfig.from_mapping({"enabled": True})
    prediction = torch.zeros(4, 10)
    observed = torch.zeros_like(prediction)
    raw = torch.cat((torch.full((1, 10), -20.0), torch.full((1, 10), 1.0), torch.full((2, 10), -3.0)))
    _, diagnostics = hgnll_objective(prediction, observed, raw, torch.tensor([0, 0, 1, 1]), {0: 1.0, 1: 1.0}, config=cfg)
    assert diagnostics["clamped_logvar_min"] == pytest.approx(-10.0)
    assert diagnostics["clamped_logvar_max"] == pytest.approx(cfg.log_variance_max)
    assert diagnostics["clamp_low_fraction"] == pytest.approx(.25)
    assert diagnostics["clamp_high_fraction"] == pytest.approx(.25)
    assert diagnostics["sigma_p10"] <= diagnostics["sigma_p50"] <= diagnostics["sigma_p90"]


def test_hgnll_matches_complete_anchor_formula_and_stack_balance():
    cfg = HGNLLConfig.from_mapping({"enabled": True})
    prediction = torch.ones(2, 10, requires_grad=True)
    observed = torch.zeros(2, 10)
    logvar = torch.full((2, 10), -3.0)
    stacks = torch.tensor([0, 1])
    loss, diagnostics = hgnll_objective(prediction, observed, logvar, stacks, {0: 1.0, 1: 2.0}, config=cfg)
    per_anchor = 0.5 * (math.exp(3.0) + (-3.0))
    expected = (per_anchor * 1.0 + per_anchor * 2.0) / 2.0
    assert loss.item() == pytest.approx(expected)
    assert diagnostics["joint_signal_mse"].item() == pytest.approx(1.0)
    loss.backward()
    assert prediction.grad is not None


def test_hgnll_reliability_multiplier_acts_on_complete_anchor_after_weight_reduction():
    cfg = HGNLLConfig.from_mapping({"enabled": True})
    prediction = torch.ones(2, 10)
    observed = torch.zeros(2, 10)
    logvar = torch.zeros(2, 10)
    stacks = torch.tensor([0, 1])
    baseline, _ = hgnll_objective(prediction, observed, logvar, stacks, {0: 1.0, 1: 1.0}, config=cfg)
    weighted, _ = hgnll_objective(prediction, observed, logvar, stacks, {0: 1.0, 1: 1.0}, reliability_multiplier=torch.tensor([0.5, 1.5]), config=cfg)
    assert weighted.item() == pytest.approx(baseline.item())


def test_stage_a_can_remain_plain_mse_monitor_and_old_mse_is_unchanged():
    prediction = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    observed = prediction + 1
    stacks = torch.tensor([0, 1])
    assert joint_vector_mse(prediction, observed, stacks, {0: 1.0, 1: 2.0}).item() == pytest.approx(1.5)


def test_hgnll_routes_are_new_and_old_routes_are_unchanged():
    r0 = build_q002_route_config(_base(), cropped_prepared_root="cropped", experiment_id=B1_R0_EXPERIMENT_ID, anchor_batch_size=640)["route"]
    r1 = build_q002_route_config(_base(), cropped_prepared_root="cropped", experiment_id=B1_R1_EXPERIMENT_ID, anchor_batch_size=640)["route"]
    h0 = build_q002_route_config(_base(), cropped_prepared_root="cropped", experiment_id=B1_R0_HGNLL_EXPERIMENT_ID, anchor_batch_size=640)["route"]
    h1 = build_q002_route_config(_base(), cropped_prepared_root="cropped", experiment_id=B1_R1_HGNLL_EXPERIMENT_ID, anchor_batch_size=640)["route"]
    assert "heteroscedastic" not in r0 and "heteroscedastic" not in r1
    assert r0["stack_balance"] == {"enabled": True} and r0["cross_view_reliability"]["enabled"] is False
    assert r1["stack_balance"] == {"enabled": True} and r1["cross_view_reliability"]["discrepancy"] == "raw_cosine"
    assert h0["heteroscedastic"]["enabled"] is True and h0["cross_view_reliability"]["enabled"] is False
    assert h1["heteroscedastic"]["enabled"] is True and h1["cross_view_reliability"]["enabled"] is True
    assert h1["cross_view_reliability"]["discrepancy"] == "raw_cosine"


def test_hgnll_routes_are_formally_sparse_admitted_with_batch_640():
    assert B1_R0_HGNLL_EXPERIMENT_ID in FORMAL_SPARSE_EXPERIMENT_IDS
    assert B1_R1_HGNLL_EXPERIMENT_ID in FORMAL_SPARSE_EXPERIMENT_IDS
    for route_id in (B1_R0_HGNLL_EXPERIMENT_ID, B1_R1_HGNLL_EXPERIMENT_ID):
        route = build_q002_route_config(_base(), cropped_prepared_root="cropped", experiment_id=route_id, anchor_batch_size=640)["route"]
        assert route["sampling"]["anchor_batch_size"] == 640
        assert route["sampling"]["shared_training_psf_samples"] == 8
