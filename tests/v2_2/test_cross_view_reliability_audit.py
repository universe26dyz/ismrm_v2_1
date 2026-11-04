import numpy as np
import json
import subprocess
import sys
import torch

from ismrm_v2.v2_2.preflight.cross_view_reliability_audit import (
    binned_cosine_statistics,
    cosine_discrepancy,
    match_lax_to_sax,
    timing_only_cosine_discrepancy,
    timing_statistics,
    run_cross_view_reliability_audit,
    write_audit_outputs,
)


def test_match_lax_to_sax_recovers_exact_and_known_nearest_reference():
    sax = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    lax = np.array([[0.0, 0.0, 0.0], [3.1, 0.0, 0.0]])

    reference_index, distance_mm = match_lax_to_sax(sax, lax, chunk_size=1)

    assert reference_index.tolist() == [0, 1]
    assert np.allclose(distance_mm, [0.0, 0.9])


def test_cosine_discrepancy_is_amplitude_invariant_and_flags_degenerate_rows():
    left = np.array([[1.0] * 10, [1.0] + [0.0] * 9, [0.0] * 10, [np.nan] + [0.0] * 9])
    right = np.array([[2.0] * 10, [0.0, 1.0] + [0.0] * 8, [1.0] * 10, [1.0] * 10])

    discrepancy, valid = cosine_discrepancy(left, right, epsilon=1e-8)

    assert valid.tolist() == [True, True, False, False]
    assert np.isclose(discrepancy[0], 0.0)
    assert np.isclose(discrepancy[1], 1.0)
    assert np.isnan(discrepancy[2:]).all()


def test_timing_statistics_and_bins_have_deterministic_empty_behavior():
    delta = np.array([[0.0] * 9, [2.0] * 9, [-4.0] * 9])

    summary = timing_statistics(delta)
    bins = binned_cosine_statistics(
        metric=np.array([0.5, 2.5]),
        discrepancy=np.array([0.1, 0.9]),
        edges=[0.0, 1.0, 2.0, np.inf],
    )

    assert summary["count"] == 3
    assert np.allclose(summary["per_dimension_mean_abs_ms"], [2.0] * 9)
    assert np.allclose(summary["per_pair_abs_mean_ms"], [0.0, 2.0, 4.0])
    assert [entry["count"] for entry in bins] == [1, 0, 1]
    assert bins[1]["median_cosine_discrepancy"] is None


class _FrozenDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grad_enabled: list[bool] = []
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(self, t1, t2, b1, timing):
        self.grad_enabled.append(torch.is_grad_enabled())
        self.calls.append((t1.detach().clone(), t2.detach().clone(), b1.detach().clone(), timing.detach().clone()))
        raw = torch.cat((timing[:, :1], torch.ones((timing.shape[0], 9), device=timing.device)), dim=1)
        return raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)


def test_timing_only_simulation_holds_parameters_fixed_and_uses_no_grad():
    decoder = _FrozenDecoder()
    t1 = torch.tensor([1000.0, 1200.0])
    t2 = torch.tensor([50.0, 60.0])
    b1 = torch.tensor([0.9, 1.0])
    timing_a = torch.zeros((2, 9))
    timing_b = timing_a.clone(); timing_b[1, 0] = 2.0

    discrepancy = timing_only_cosine_discrepancy(decoder, t1, t2, b1, timing_a, timing_b)

    assert np.isclose(discrepancy[0], 0.0)
    assert discrepancy[1] > 0.0
    assert decoder.grad_enabled == [False, False]
    for before, after in zip(decoder.calls[0][:3], decoder.calls[1][:3]):
        assert torch.equal(before, after)


class _JointFixture:
    def __init__(self) -> None:
        self.xyz = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.2, 0.0, 0.0], [3.5, 0.0, 0.0], [4.1, 0.0, 0.0]])
        self.observed = torch.tensor([[1.0] * 10, [1.0] * 10, [2.0] * 10, [1.0] + [0.0] * 9, [0.0] + [1.0] * 9])
        self.group_idx = torch.tensor([0, 0, 1, 1, 2])
        self.stack_idx = torch.tensor([0, 0, 1, 1, 2])
        self.timing = torch.tensor([[0.0] * 9, [1.0] * 9, [2.0] * 9, [3.0] * 9, [4.0] * 9])
        self.row_col = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]])
        self.qc = {"source_sha256": {"synthetic": "fixture"}}


def test_synthetic_audit_writes_only_explicit_output_root(tmp_path):
    result = run_cross_view_reliability_audit(
        _JointFixture(),
        group_axisangle_physical=torch.zeros((3, 6)),
        subject="SYN",
        case_id="L100",
        geometry_source="synthetic fixed HB1",
    )
    output = tmp_path / "v2_2_preflight"
    paths = write_audit_outputs(result, output)

    assert {pair["target_stack"] for pair in result["pairs"]} == {"2ch", "4ch"}
    assert result["coverage_statistics"]["2ch"]["<= 1 mm"] == 2
    assert result["fingerprint_statistics"]["2ch"]["valid_count"] == 2
    assert result["summary"]["timing_statistics"]["2ch"]["count"] == 2
    assert result["confounding_statistics"]["2ch"]["distance_bins"][-1]["upper"] is None
    assert all(path.is_relative_to(output) for path in paths.values())
    assert (output / "matched_pairs.csv").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["reference_stack"] == "sax"
    assert manifest["post_hb1_fixed_geometry"] == "synthetic fixed HB1"
    assert manifest["timing_only_simulator"]["timing_only_simulator_status"] == "not_run"


def test_module_cli_help_exposes_explicit_isolated_output_arguments():
    completed = subprocess.run(
        [sys.executable, "-m", "ismrm_v2.v2_2.preflight.cross_view_reliability_audit", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--prepared-root" in completed.stdout
    assert "--output-dir" in completed.stdout
