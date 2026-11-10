from __future__ import annotations

import math
import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ismrm_v2.proposed.q002.contracts import (
    B1_R0_EXPERIMENT_ID,
    B1_R0_HGNLL_EXPERIMENT_ID,
    B1_R1_EXPERIMENT_ID,
    B1_R1_HGNLL_EXPERIMENT_ID,
    build_q002_route_config,
)
from ismrm_v2.proposed.q002.run_q002_reconstruction import FORMAL_SPARSE_EXPERIMENT_IDS
from ismrm_v2.proposed.q002.training import joint_vector_mse
from ismrm_v2.proposed.q002.training import build_q002_optimizer
from ismrm_v2.proposed.reconstruction.orchestration import load_checkpoint, save_checkpoint
from ismrm_v2.proposed.rigid_psf.rigid_psf_forward import GroupRigidPSF, TradQuantitativeForward
from ismrm_v2.v2_2.stage_b_reliability import ReliabilityConfig, compute_reliability_gate, normalize_reliability_gate
from ismrm_v2.v2_3.hgnll import (
    HGNLLConfig,
    HGNLL_INITIAL_RAW_PARAMETER_BIAS,
    HGNLL_LOG_VARIANCE_INITIAL,
    HGNLL_LOG_VARIANCE_MAX,
    HGNLL_SIGMA_INITIAL,
    HGNLL_SIGMA_MAX,
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
    assert cfg.sigma_max == pytest.approx(1.0)
    assert cfg.log_variance_max == pytest.approx(0.0)
    assert cfg.sigma_initial == pytest.approx(0.5)
    assert cfg.log_variance_initial == pytest.approx(math.log(0.25))
    assert cfg.raw_parameter_initial_bias == pytest.approx(HGNLL_INITIAL_RAW_PARAMETER_BIAS)
    assert cfg.initialization == "inverse_sigmoid_of_sigma_initial"
    with pytest.raises(ValueError):
        HGNLLConfig.from_mapping({"enabled": True, "distribution": "laplace"})


def test_head_outputs_ten_log_variances_with_safe_initialization_and_gradients():
    cfg = HGNLLConfig.from_mapping({"enabled": True})
    head = JointFingerprintVarianceHead(4, cfg)
    latent = torch.randn(6, 4, requires_grad=True)
    raw = head(latent)
    output = head.bounded_log_variance_from_raw(raw)
    assert output.shape == (6, 10)
    assert torch.allclose(output, torch.full_like(output, HGNLL_LOG_VARIANCE_INITIAL), atol=1e-6)
    raw.sum().backward()
    assert latent.grad is not None
    assert head.network[-1].weight.grad is not None
    assert HGNLL_SIGMA_INITIAL == 0.5 and HGNLL_SIGMA_MAX == 1.0


def test_psf_log_mean_exp_variance_matches_definition():
    values = torch.tensor([[[0.0] * 10, [math.log(4.0)] * 10]])
    result = aggregate_log_variance(values)
    assert torch.allclose(result, torch.full((1, 10), math.log(2.5)))


def test_smooth_bounds_and_proximity_diagnostics_are_reported():
    cfg = HGNLLConfig.from_mapping({"enabled": True})
    prediction = torch.zeros(4, 10)
    observed = torch.zeros_like(prediction)
    raw_parameter = torch.cat((torch.full((1, 10), -20.0), torch.full((1, 10), 20.0), torch.zeros((2, 10))))
    head = JointFingerprintVarianceHead(1, cfg)
    bounded = head.bounded_log_variance_from_raw(raw_parameter)
    _, diagnostics = hgnll_objective(prediction, observed, bounded, torch.tensor([0, 0, 1, 1]), {0: 1.0, 1: 1.0}, raw_variance_parameter=raw_parameter, config=cfg)
    assert diagnostics["bounded_logvar_min"] >= -10.0
    assert diagnostics["bounded_logvar_max"] <= 0.0
    assert torch.all(bounded >= cfg.log_variance_min) and torch.all(bounded <= cfg.log_variance_max)
    assert diagnostics["fraction_sigma_gt_0_95_sigma_max"] > 0
    assert diagnostics["fraction_sigma_gt_0_99_sigma_max"] > 0
    assert diagnostics["sigma_p10"] <= diagnostics["sigma_p50"] <= diagnostics["sigma_p90"]


def test_formal_variance_mapping_has_no_hard_clamp():
    source = inspect.getsource(JointFingerprintVarianceHead.bounded_log_variance_from_raw)
    assert "clamp" not in source


def test_smooth_variance_mapping_has_gradients_toward_both_bounds():
    cfg = HGNLLConfig.from_mapping({"enabled": True})
    head = JointFingerprintVarianceHead(1, cfg)
    z = torch.full((1, 10), -10.0, requires_grad=True)
    bounded = head.bounded_log_variance_from_raw(z)
    upper_loss, _ = hgnll_objective(torch.full((1, 10), 10.0), torch.zeros(1, 10), bounded, torch.tensor([0]), {0: 1.0}, config=cfg)
    upper_grad = torch.autograd.grad(upper_loss, z, retain_graph=True)[0]
    z_lower = torch.full((1, 10), 10.0, requires_grad=True)
    lower_bounded = head.bounded_log_variance_from_raw(z_lower)
    lower_loss, _ = hgnll_objective(torch.zeros(1, 10), torch.zeros(1, 10), lower_bounded, torch.tensor([0]), {0: 1.0}, config=cfg)
    lower_grad = torch.autograd.grad(lower_loss, z_lower)[0]
    assert torch.all(upper_grad < 0)  # descent increases log variance
    assert torch.all(lower_grad > 0)  # descent decreases log variance
    assert torch.all(upper_grad.abs() > 0) and torch.all(lower_grad.abs() > 0)


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
    assert diagnostics["joint_signal_mse"].item() == pytest.approx(1.5)
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
    assert h1["heteroscedastic"]["variance_policy_version"] == "multimap_hgnll_smooth_sigma1_v1"
    assert h1["heteroscedastic"]["sigma_max"] == 1.0
    assert h1["heteroscedastic"]["sigma_max_reason"] == "pre_existing_v2_2_mse_residual_scale_audit"
    assert "initialization_epsilon" not in h1["heteroscedastic"]


def test_hgnll_routes_are_formally_sparse_admitted_with_batch_640():
    assert B1_R0_HGNLL_EXPERIMENT_ID in FORMAL_SPARSE_EXPERIMENT_IDS
    assert B1_R1_HGNLL_EXPERIMENT_ID in FORMAL_SPARSE_EXPERIMENT_IDS
    for route_id in (B1_R0_HGNLL_EXPERIMENT_ID, B1_R1_HGNLL_EXPERIMENT_ID):
        route = build_q002_route_config(_base(), cropped_prepared_root="cropped", experiment_id=route_id, anchor_batch_size=640)["route"]
        assert route["sampling"]["anchor_batch_size"] == 640
        assert route["sampling"]["shared_training_psf_samples"] == 8


class _FakeINR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding = nn.Linear(2, 2)
        self.network = nn.Linear(2, 2)
        self.config = SimpleNamespace(latent_features=2)


class _FakeRigid(nn.Module):
    def __init__(self):
        super().__init__()
        self.axisangle = nn.Parameter(torch.zeros(1, 6))


class _FakeModel(nn.Module):
    def __init__(self, with_head: bool):
        super().__init__()
        self.inr = _FakeINR()
        self.rigid_psf = _FakeRigid()
        self.variance_head = None
        self.joint_variance_head = JointFingerprintVarianceHead(2, HGNLLConfig.from_mapping({"enabled": True})) if with_head else None


def _learning_rates():
    return {"encoding": 1e-3, "network": 1e-3, "weight_decay": 0.0, "rigid": 1e-3, "variance": 2e-3}


def test_hgnll_optimizer_has_one_variance_group_and_updates_it():
    cfg = HGNLLConfig.from_mapping({"enabled": True})
    model = _FakeModel(True)
    optimizer = build_q002_optimizer(model, _learning_rates(), False, hgnll_config=cfg)
    variance_groups = [group for group in optimizer.param_groups if group["name"] == "variance"]
    assert len(variance_groups) == 1
    assert variance_groups[0]["lr"] == pytest.approx(2e-3)
    before = [parameter.detach().clone() for parameter in model.joint_variance_head.parameters()]
    loss = model.joint_variance_head(torch.ones(4, 2)).pow(2).mean()
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.joint_variance_head.parameters()))
    off = build_q002_optimizer(_FakeModel(False), _learning_rates(), False, hgnll_config=HGNLLConfig())
    assert all(group["name"] != "variance" for group in off.param_groups)


class _DummyQuantitative(nn.Module):
    def forward_with_latent(self, xyz, *, include_latent=True):
        latent = torch.cat((xyz, xyz[:, :1]), dim=1)
        return {"t1_ms": torch.full((xyz.shape[0],), 1000.0, device=xyz.device), "t2_ms": torch.full((xyz.shape[0],), 50.0, device=xyz.device), "b1": torch.ones(xyz.shape[0], device=xyz.device), "amplitude": torch.ones(xyz.shape[0], device=xyz.device), "latent": latent}

    def forward(self, xyz):
        return self.forward_with_latent(xyz)


class _DummySimulator(nn.Module):
    def forward(self, t1, t2, b1, timing, protocol, normalize=True):
        return torch.stack(tuple((t1 + t2 + b1 + timing[:, index % 9]) * 0.001 for index in range(10)), dim=1)


def test_same_precomputed_psf_world_samples_preserve_mean_prediction():
    rigid = GroupRigidPSF(torch.zeros(1, 6), torch.ones(1, 3))
    forward = TradQuantitativeForward(_DummyQuantitative(), _DummySimulator(), rigid, object())
    head = JointFingerprintVarianceHead(4, HGNLLConfig.from_mapping({"enabled": True}))
    xyz = torch.zeros(3, 3); groups = torch.zeros(3, dtype=torch.long); timing = torch.ones(3, 9)
    world = torch.arange(72, dtype=torch.float32).reshape(3, 8, 3) / 10.0
    mean = forward.forward_fingerprint(xyz, groups, timing, 8, world_samples=world)
    hgnll_mean, logvar = forward.forward_fingerprint_with_log_variance(xyz, groups, timing, 8, head, world_samples=world)
    assert torch.allclose(mean, hgnll_mean, rtol=0.0, atol=1e-7)
    assert logvar.shape == (3, 10)


def test_hgnll_stage_a_is_active_without_reliability_and_stage_b_uses_raw_cosine_gate():
    cfg = HGNLLConfig.from_mapping({"enabled": True})
    prediction = torch.ones(4, 10); observed = torch.zeros_like(prediction); logvar = torch.full_like(prediction, -3.0); stacks = torch.tensor([0, 0, 1, 1])
    stage_a, a_diag = hgnll_objective(prediction, observed, logvar, stacks, {0: 1.0, 1: 1.0}, config=cfg)
    rel_cfg = ReliabilityConfig.from_mapping({"enabled": True})
    gate = compute_reliability_gate(torch.tensor([0.0, 0.0, 0.1, 0.2]), torch.ones(4, dtype=torch.bool), stacks, {1: .1}, 1.0, rel_cfg)
    multiplier = normalize_reliability_gate(gate, stacks)
    stage_b, b_diag = hgnll_objective(prediction, observed, logvar, stacks, {0: 1.0, 1: 1.0}, reliability_multiplier=multiplier, config=cfg)
    assert stage_a.item() != 0.0 and stage_b.item() != 0.0
    assert torch.equal(a_diag["hgnll_reliability_multiplier"], torch.ones(4))
    assert torch.allclose(b_diag["hgnll_reliability_multiplier"], multiplier)


@dataclass
class _Protocol:
    tr_ms: float = 5.0
    vps: int = 10
    fa_deg: tuple = (1.0,)
    ti_ms: tuple = (1.0,)
    t2prep_ms: tuple = (1.0,)
    n_ramp_up: int = 1


class _FakeSpace:
    group_axisangle_dicom_physical = torch.zeros(1, 6)
    group_axisangle_init_physical = torch.zeros(1, 6)
    group_axisangle_init_train = torch.zeros(1, 6)

    def state_dict(self):
        return {}


class _FakeDataset:
    group_resolution_xyz_mm = torch.ones(1, 3)


def test_hgnll_checkpoint_round_trip_preserves_head_and_provenance(tmp_path):
    cfg = HGNLLConfig.from_mapping({"enabled": True})
    model = _FakeModel(True)
    config = {"heteroscedastic": cfg.as_dict(), "training": {"learning_rates": _learning_rates()}}
    path = tmp_path / "hgnll.pt"
    save_checkpoint(path, model, config, _Protocol(), _FakeDataset(), _FakeSpace(), 20260911, {}, {})
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    rebuilt = _FakeModel(True)
    load_checkpoint(path, rebuilt, torch.device("cpu"))
    for left, right in zip(model.joint_variance_head.parameters(), rebuilt.joint_variance_head.parameters()):
        assert torch.equal(left, right)
    assert checkpoint["hgnll_config"] == cfg.as_dict()


def test_hgnll_decomposition_is_exact_after_stack_and_reliability_weighting():
    cfg = HGNLLConfig.from_mapping({"enabled": True})
    prediction = torch.tensor([[1.0] * 10, [2.0] * 10, [0.5] * 10, [1.5] * 10])
    observed = torch.zeros_like(prediction); logvar = torch.tensor([[-3.0] * 10, [-2.5] * 10, [-3.5] * 10, [-2.8] * 10]); stacks = torch.tensor([0, 0, 1, 1])
    total, diagnostics = hgnll_objective(prediction, observed, logvar, stacks, {0: 1.0, 1: 2.0}, reliability_multiplier=torch.tensor([.5, 1.5, .7, 1.3]), config=cfg)
    assert torch.allclose(diagnostics["hgnll_residual_term"] + diagnostics["hgnll_log_variance_term"], total, rtol=1e-6, atol=1e-7)


def test_old_routes_have_no_hgnll_provenance_and_nl1_stays_unchanged():
    for route_id in (B1_R0_EXPERIMENT_ID, B1_R1_EXPERIMENT_ID, "B1_R1_NL1"):
        route = build_q002_route_config(_base(), cropped_prepared_root="cropped", experiment_id=route_id, anchor_batch_size=640)["route"]
        assert "heteroscedastic" not in route
    h0 = build_q002_route_config(_base(), cropped_prepared_root="cropped", experiment_id=B1_R0_HGNLL_EXPERIMENT_ID, anchor_batch_size=640)["route"]
    h1 = build_q002_route_config(_base(), cropped_prepared_root="cropped", experiment_id=B1_R1_HGNLL_EXPERIMENT_ID, anchor_batch_size=640)["route"]
    assert h0["parent_experiment"] == B1_R0_EXPERIMENT_ID
    assert h1["parent_experiment"] == B1_R1_EXPERIMENT_ID
