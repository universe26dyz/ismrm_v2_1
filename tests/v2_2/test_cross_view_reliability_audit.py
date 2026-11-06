import numpy as np
import json
import inspect
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch

import ismrm_v2.v2_2.preflight.cross_view_reliability_audit as audit
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


def test_match_lax_to_sax_uses_spatial_index_for_large_fixture(monkeypatch):
    calls = {"query": 0}

    class _RecordingTree:
        def __init__(self, points):
            self.points = np.asarray(points)

        def query(self, points, k):
            calls["query"] += 1
            assert k == 1
            assert self.points.shape == (4096, 3)
            assert np.asarray(points).shape == (2048, 3)
            return np.zeros(2048), np.zeros(2048, dtype=np.int64)

    monkeypatch.setattr(audit, "cKDTree", _RecordingTree)
    reference_index, distance_mm = match_lax_to_sax(np.zeros((4096, 3)), np.zeros((2048, 3)))

    assert calls == {"query": 1}
    assert reference_index.dtype == np.int64
    assert distance_mm.dtype == np.float64
    assert audit.nearest_neighbor_backend() == "scipy.spatial.cKDTree"


def test_match_lax_to_sax_rejects_large_all_pairs_without_spatial_index(monkeypatch):
    monkeypatch.setattr(audit, "cKDTree", None)

    with np.testing.assert_raises_regex(RuntimeError, "SciPy cKDTree"):
        match_lax_to_sax(np.zeros((400, 3)), np.zeros((400, 3)))


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

    scaled = timing_only_cosine_discrepancy(decoder, t1, t2, b1, timing_a, timing_b, amplitude=torch.tensor([2.0, 3.0]))
    assert np.allclose(scaled, discrepancy)


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
    assert np.isclose(result["coverage_statistics"]["2ch"]["distance_mm_summary"]["median"], 0.35)
    assert np.isclose(result["coverage_statistics"]["2ch"]["distance_mm_summary"]["p90"], 0.47)
    assert np.isclose(result["coverage_statistics"]["2ch"]["distance_mm_summary"]["max"], 0.5)
    assert result["summary"]["coverage_statistics"]["2ch"]["distance_mm_summary"] == result["coverage_statistics"]["2ch"]["distance_mm_summary"]
    assert result["fingerprint_statistics"]["2ch"]["valid_count"] == 2
    assert result["summary"]["timing_statistics"]["2ch"]["count"] == 2
    assert result["confounding_statistics"]["2ch"]["distance_bins"][-1]["upper"] is None
    assert all(path.is_relative_to(output) for path in paths.values())
    assert (output / "matched_pairs.csv").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["reference_stack"] == "sax"
    assert manifest["post_hb1_fixed_geometry"] == "synthetic fixed HB1"
    assert manifest["timing_only_simulator"]["timing_only_simulator_status"] == "not_run"


def test_timing_only_outputs_have_per_stack_summaries_and_correlations(tmp_path):
    result = run_cross_view_reliability_audit(
        _JointFixture(),
        group_axisangle_physical=torch.zeros((3, 6)),
        subject="SYN",
        case_id="L100",
        geometry_source="synthetic fixed HB1",
        timing_only_decoder=_FrozenDecoder(),
        timing_only_parameters={"t1_ms": np.full(5, 1000.0), "t2_ms": np.full(5, 50.0), "b1": np.ones(5), "amplitude": np.ones(5)},
    )
    output = tmp_path / "timing_only"
    write_audit_outputs(result, output)

    payload = result["timing_only_simulator"]
    assert payload["timing_only_simulator_status"] == "run"
    assert payload["timing_only_parameter_source"] == "external_anchor_aligned_npz"
    assert payload["stacks"]["2ch"]["count"] == 2
    assert payload["stacks"]["2ch"]["valid_count"] == 2
    assert "corr_timing_only_vs_timing_abs_mean_ms" in payload["stacks"]["2ch"]
    assert result["summary"]["timing_only_simulator"] == payload
    assert json.loads((output / "timing_only_simulator.json").read_text()) == payload
    assert json.loads((output / "summary.json").read_text())["timing_only_simulator"] == payload


def test_timing_only_parameter_npz_requires_exact_joint_anchor_order(tmp_path):
    parameter_file = tmp_path / "parameters.npz"
    expected_hash = "a" * 64
    np.savez(
        parameter_file,
        t1_ms=np.array([1000.0, 1100.0]),
        t2_ms=np.array([50.0, 55.0]),
        b1=np.array([0.9, 1.0]),
        amplitude=np.array([1.0, 2.0]),
        anchor_order_sha256=np.array(expected_hash),
    )

    parameters = audit._load_timing_only_parameters(parameter_file, 2, expected_hash)

    assert parameters["t1_ms"].shape == (2,)
    np.savez(parameter_file, t1_ms=np.array([1000.0, 1100.0]), t2_ms=np.array([50.0, 55.0]), b1=np.array([0.9, 1.0]), anchor_order_sha256=np.array(expected_hash))
    with np.testing.assert_raises_regex(ValueError, "amplitude"):
        audit._load_timing_only_parameters(parameter_file, 2, expected_hash)
    np.savez(
        parameter_file,
        t1_ms=np.array([1000.0, 1100.0]),
        t2_ms=np.array([50.0, 55.0]),
        b1=np.array([0.9, 1.0]),
        amplitude=np.array([1.0, 2.0]),
        anchor_order_sha256=np.array("b" * 64),
    )
    with np.testing.assert_raises_regex(ValueError, "anchor_order_sha256"):
        audit._load_timing_only_parameters(parameter_file, 2, expected_hash)


def _observation_paths(root):
    return {stack: root / stack / "observations.npz" for stack in ("sax", "2ch", "4ch")}


def _profile_l100_manifest(tmp_path):
    source_paths = _observation_paths(tmp_path / "source")
    derived_paths = _observation_paths(tmp_path / "derived")
    stacks = {}
    for stack in ("sax", "2ch", "4ch"):
        stacks[stack] = {
            "source_observation": {"path": str(source_paths[stack]), "sha256": "a" * 64},
            "derived_observation": {"path": str(derived_paths[stack]), "sha256": "b" * 64},
            "training_stack_present": True,
            "ordered_source_group_ids": [0, 1],
            "retained_original_group_ids": [0, 1],
            "heldout_original_group_ids": [],
            "source_count": 2,
            "retained_count": 2,
            "heldout_count": 0,
        }
    return {
        "schema_version": "phase4r_sparse_profile_v2", "subject": "CYJ", "profile_id": "L100",
        "case_id": "L100", "fold_id": None, "fold_count": 1,
        "source_prepared_root": str(tmp_path / "source"), "prepared_root": str(tmp_path / "derived"),
        "stacks": stacks, "_manifest_path": str(tmp_path / "derived" / "sparse_manifest.json"), "_manifest_sha256": "b" * 64,
    }, source_paths, derived_paths


def test_preflight_sparse_manifest_accepts_legacy_l100_source_inputs(monkeypatch, tmp_path):
    source_paths = _observation_paths(tmp_path / "source")
    manifest = {
        "schema_version": "phase4r_sparse_v1", "subject": "CYJ", "condition": "L100",
        "profile_id": "legacy_group_id_modulo_L100", "case_id": "L100_fold0", "fold_id": 0, "fold_count": 1,
        "source_prepared_root": str(tmp_path / "source"), "prepared_root": str(tmp_path / "source"),
        "stacks": {
            stack: {"source_observation": {"path": str(path), "sha256": "a" * 64}, "retained_original_group_ids": [0, 1], "heldout_original_group_ids": []}
            for stack, path in source_paths.items()
        },
        "_manifest_path": str(tmp_path / "legacy_manifest.json"), "_manifest_sha256": "a" * 64,
    }
    monkeypatch.setattr(audit, "validate_sparse_profile_manifest", lambda *args, **kwargs: dict(manifest))

    provenance = audit.validate_preflight_sparse_manifest(tmp_path / "legacy_manifest.json", subject="CYJ", requested_case="L100", observation_paths=source_paths)

    assert provenance["input_observation_mode"] == "source"
    assert provenance["l100_full_retention_verified"] is True
    assert provenance["legacy_identity_full_retention"] is True


def test_preflight_sparse_manifest_accepts_profile_l100_source_inputs(monkeypatch, tmp_path):
    manifest, source_paths, _derived_paths = _profile_l100_manifest(tmp_path)
    monkeypatch.setattr(audit, "validate_sparse_profile_manifest", lambda *args, **kwargs: dict(manifest))

    provenance = audit.validate_preflight_sparse_manifest(tmp_path / "derived" / "sparse_manifest.json", subject="CYJ", requested_case="L100", observation_paths=source_paths)

    assert provenance["input_observation_mode"] == "source"
    assert provenance["l100_full_retention_verified"] is True


def test_preflight_sparse_manifest_accepts_profile_l100_derived_inputs(monkeypatch, tmp_path):
    manifest, _source_paths, derived_paths = _profile_l100_manifest(tmp_path)
    monkeypatch.setattr(audit, "validate_sparse_profile_manifest", lambda *args, **kwargs: dict(manifest))

    provenance = audit.validate_preflight_sparse_manifest(tmp_path / "derived" / "sparse_manifest.json", subject="CYJ", requested_case="L100", observation_paths=derived_paths)

    assert provenance["input_observation_mode"] == "derived"


def test_preflight_sparse_manifest_rejects_mixed_profile_observation_modes(monkeypatch, tmp_path):
    manifest, source_paths, derived_paths = _profile_l100_manifest(tmp_path)
    mixed_paths = dict(source_paths); mixed_paths["4ch"] = derived_paths["4ch"]
    monkeypatch.setattr(audit, "validate_sparse_profile_manifest", lambda *args, **kwargs: dict(manifest))

    with np.testing.assert_raises_regex(ValueError, "coherent"):
        audit.validate_preflight_sparse_manifest(tmp_path / "derived" / "sparse_manifest.json", subject="CYJ", requested_case="L100", observation_paths=mixed_paths)


def test_preflight_sparse_manifest_rejects_profile_l100_with_heldout_groups(monkeypatch, tmp_path):
    manifest, source_paths, _derived_paths = _profile_l100_manifest(tmp_path)
    manifest["stacks"]["2ch"]["heldout_original_group_ids"] = [1]
    manifest["stacks"]["2ch"]["retained_original_group_ids"] = [0]
    manifest["stacks"]["2ch"]["retained_count"] = 1
    manifest["stacks"]["2ch"]["heldout_count"] = 1
    monkeypatch.setattr(audit, "validate_sparse_profile_manifest", lambda *args, **kwargs: dict(manifest))

    with np.testing.assert_raises_regex(ValueError, "full retention"):
        audit.validate_preflight_sparse_manifest(tmp_path / "derived" / "sparse_manifest.json", subject="CYJ", requested_case="L100", observation_paths=source_paths)


def test_preflight_sparse_manifest_rejects_wrong_subject(monkeypatch, tmp_path):
    manifest, source_paths, _derived_paths = _profile_l100_manifest(tmp_path)
    manifest["subject"] = "OTHER"
    monkeypatch.setattr(audit, "validate_sparse_profile_manifest", lambda *args, **kwargs: dict(manifest))

    with np.testing.assert_raises_regex(ValueError, "subject"):
        audit.validate_preflight_sparse_manifest(tmp_path / "derived" / "sparse_manifest.json", subject="CYJ", requested_case="L100", observation_paths=source_paths)


def test_stack_initialization_provenance_hashes_post_hb1_geometry_deterministically():
    initialization = SimpleNamespace(
        post_stack_init_axisangle_physical=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        stack_pose_records=({"prepared_input": "sax/observations.npz", "group_count": 1},),
    )
    altered = SimpleNamespace(
        post_stack_init_axisangle_physical=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 7.0]]),
        stack_pose_records=initialization.stack_pose_records,
    )

    provenance = audit.stack_initialization_provenance(initialization)

    assert provenance["post_stack_init_axisangle_physical_sha256"] == audit.stack_initialization_provenance(initialization)["post_stack_init_axisangle_physical_sha256"]
    assert provenance["post_stack_init_axisangle_physical_sha256"] != audit.stack_initialization_provenance(altered)["post_stack_init_axisangle_physical_sha256"]
    assert provenance["stack_pose_records"] == [{"prepared_input": "sax/observations.npz", "group_count": 1}]


def test_sparse_manifest_provenance_records_not_supplied_status(tmp_path):
    provenance = audit.resolve_preflight_sparse_manifest(
        None,
        subject="CYJ",
        requested_case="L100",
        observation_paths=_observation_paths(tmp_path),
    )

    assert provenance == {
        "sparse_manifest_status": "not_supplied",
        "requested_case": "L100",
        "manifest_resolved_case": None,
    }


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


def _geometry_checkpoint(tmp_path, *, post=None, dicom=None, resolution=None, training_space=None, trained=None):
    post = torch.tensor([[0.0, 0.1, 0.2, 1.0, 2.0, 3.0], [0.3, 0.4, 0.5, 4.0, 5.0, 6.0]]) if post is None else torch.as_tensor(post)
    dicom = torch.tensor([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 4.0, 5.0, 6.0]]) if dicom is None else torch.as_tensor(dicom)
    resolution = torch.tensor([[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]]) if resolution is None else torch.as_tensor(resolution)
    if training_space is None:
        training_space = {
            "group_axisangle_init_physical": post.clone(),
            "group_axisangle_dicom_physical": dicom.clone(),
            "group_resolution_xyz_mm": resolution.clone(),
        }
    payload = {
        "group_axisangle_post_stack_init_physical": post,
        "group_axisangle_dicom_physical": dicom,
        "group_axisangle_init_physical": post.clone(),
        "group_resolution_xyz_mm": resolution,
        "training_space": training_space,
        "trained_axisangle_train": torch.full_like(post, 99.0) if trained is None else torch.as_tensor(trained),
    }
    path = tmp_path / "geometry_checkpoint.pt"
    torch.save(payload, path)
    return path, post, dicom, resolution


def test_geometry_checkpoint_selects_post_hb1_geometry_and_records_provenance(tmp_path):
    checkpoint, post, dicom, resolution = _geometry_checkpoint(tmp_path)

    selected, provenance = audit.load_geometry_checkpoint(checkpoint, dicom, resolution)

    assert torch.equal(selected, post)
    assert not torch.equal(selected, torch.full_like(post, 99.0))
    assert provenance["selected_key"] == "group_axisangle_post_stack_init_physical"
    assert provenance["source"] == "frozen Q002 checkpoint: group_axisangle_post_stack_init_physical"
    assert provenance["group_count"] == 2
    assert provenance["internal_training_space_consistency"] is True
    assert provenance["dicom_geometry_max_abs_diff"] == 0.0
    assert provenance["group_resolution_max_abs_diff"] == 0.0
    assert provenance == audit.load_geometry_checkpoint(checkpoint, dicom, resolution)[1]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("group_axisangle_post_stack_init_physical"), "group_axisangle_post_stack_init_physical"),
        (lambda payload: payload.__setitem__("group_axisangle_post_stack_init_physical", torch.zeros((2, 5))), "shape"),
        (lambda payload: payload.__setitem__("group_axisangle_post_stack_init_physical", torch.tensor([[float("nan")] * 6, [0.0] * 6])), "finite"),
    ],
)
def test_geometry_checkpoint_fails_closed_for_missing_bad_shape_or_nonfinite_post_geometry(tmp_path, mutation, message):
    checkpoint, _post, dicom, resolution = _geometry_checkpoint(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mutation(payload)
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=message):
        audit.load_geometry_checkpoint(checkpoint, dicom, resolution)


def test_geometry_checkpoint_fails_closed_for_dataset_geometry_mismatches(tmp_path):
    checkpoint, _post, dicom, resolution = _geometry_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="DICOM geometry mismatch"):
        audit.load_geometry_checkpoint(checkpoint, dicom + 1e-4, resolution)
    with pytest.raises(ValueError, match="group resolution mismatch"):
        audit.load_geometry_checkpoint(checkpoint, dicom, resolution + 1e-4)


def test_geometry_checkpoint_fails_closed_for_group_count_and_training_space_inconsistency(tmp_path):
    checkpoint, _post, dicom, resolution = _geometry_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="group count"):
        audit.load_geometry_checkpoint(checkpoint, torch.cat((dicom, dicom[:1])), torch.cat((resolution, resolution[:1])))

    inconsistent = {
        "group_axisangle_init_physical": torch.zeros((2, 6)),
        "group_axisangle_dicom_physical": dicom,
        "group_resolution_xyz_mm": resolution,
    }
    checkpoint, _post, dicom, resolution = _geometry_checkpoint(tmp_path, training_space=inconsistent)
    with pytest.raises(ValueError, match="training_space.*group_axisangle_init_physical"):
        audit.load_geometry_checkpoint(checkpoint, dicom, resolution)


def test_formal_cli_requires_geometry_checkpoint_and_does_not_rerun_hb1():
    parser = audit.build_parser()
    assert parser.parse_args(["--case", "L100", "--prepared-root", "prepared", "--output-dir", "out", "--geometry-checkpoint", "checkpoint.pt"]).geometry_checkpoint == "checkpoint.pt"
    with pytest.raises(SystemExit):
        parser.parse_args(["--case", "L100", "--prepared-root", "prepared", "--output-dir", "out"])
    assert "initialize_group_poses_from_hb1" not in inspect.getsource(audit.run_from_args)

    completed = subprocess.run(
        [sys.executable, "-m", "ismrm_v2.v2_2.preflight.cross_view_reliability_audit", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert "--geometry-checkpoint" in completed.stdout
