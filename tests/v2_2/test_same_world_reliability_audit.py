import numpy as np
import json
import torch
from nesvor.transform import RigidTransform, ax_transform_points

from ismrm_v2.v2_2.preflight.discrepancy import (
    fit_joint_robust_normalization,
    normalized_l1_discrepancy,
    srpan_gate,
)
from ismrm_v2.v2_2.preflight.reference_sampling import (
    build_regular_grid_reference,
    build_slice_stack_reference,
    sample_regular_grid_trilinear,
    sample_slice_stack,
)
from ismrm_v2.v2_2.preflight.export_timing_only_anchor_parameters import export_anchor_parameters
import ismrm_v2.v2_2.preflight.same_world_reliability_audit as same_world


def test_trilinear_sampling_recovers_ten_channel_linear_field():
    axes = (np.array([0.0, 1.0]), np.array([0.0, 2.0]), np.array([0.0, 3.0]))
    points = np.array([[x, y, z] for z in axes[2] for y in axes[1] for x in axes[0]])
    values = np.column_stack([2.0 * points[:, 0] + (k + 1) * points[:, 1] - 0.5 * points[:, 2] for k in range(10)])
    grid = build_regular_grid_reference(points, values)
    sampled, valid, _ = sample_regular_grid_trilinear(grid, np.array([[0.25, 0.5, 1.5], [9.0, 0.0, 0.0]]))
    expected = np.array([[2.0 * .25 + (k + 1) * .5 - .5 * 1.5 for k in range(10)], [np.nan] * 10])
    assert valid.tolist() == [True, False]
    assert np.allclose(sampled[0], expected[0])
    assert np.isnan(sampled[1]).all()


def test_normalized_l1_and_gate_are_explicit_and_monotone():
    target = np.ones((2, 10))
    reference = np.vstack((target[0], np.full(10, 3.0)))
    stats = fit_joint_robust_normalization(target, reference, low_quantile=0.0, high_quantile=1.0)
    values, valid = normalized_l1_discrepancy(target, reference, stats)
    assert valid.tolist() == [True, True]
    assert np.isclose(values[0], 0.0)
    gate = srpan_gate(np.array([0.0, 1.0, 2.0]), tau=1.0, progress=0.5)
    assert np.isclose(gate[0], 1.0)
    assert np.all(np.diff(gate) < 0)


def test_same_world_geometry_transform_is_cpu_safe():
    points = np.array([[0., 0., 0.], [1., 0., 0.]])
    values = np.ones((2, 10))
    grid = build_regular_grid_reference(points, values, axes=(np.array([0., 1.]), np.array([0.]), np.array([0.])))
    sampled, valid, _ = sample_regular_grid_trilinear(grid, torch.tensor([[1., 0., 0.] ]).numpy())
    assert valid[0]
    assert np.allclose(sampled[0], 1.0)


def test_nesvor_transform_inversion_matches_world_ras_sampling_convention():
    pose = torch.tensor([[0.2, -0.1, 0.3, 4.0, -2.0, 1.5]], dtype=torch.float64)
    local = torch.tensor([[1.2, -0.7, 2.1], [-0.5, 0.4, -1.0]], dtype=torch.float64)
    world = ax_transform_points(pose, local, trans_first=True)
    recovered = ax_transform_points(RigidTransform(pose, trans_first=True).inv().axisangle(trans_first=True), world, trans_first=True)
    assert torch.allclose(recovered, local, atol=1e-8, rtol=0)


def test_oblique_parallel_slice_stack_brackets_planes_and_matches_linear_field(tmp_path):
    height, width, groups = 2, 2, 2
    images = np.zeros((groups * 10, height, width), dtype=np.float32)
    masks = np.ones_like(images, dtype=bool)
    group_idx = np.repeat(np.arange(groups), 10)
    weight_idx = np.tile(np.arange(10), groups)
    stack_idx = np.zeros(groups * 10, dtype=np.int64)
    spacing = np.ones((groups * 10, 2), dtype=np.float64)
    timing = np.zeros((groups * 10, 9), dtype=np.float64)
    for group, z in enumerate((0.0, 2.0)):
        for weight in range(10):
            row = group * 10 + weight
            for pixel_row in range(height):
                for pixel_col in range(width):
                    x = pixel_col - .5; y = pixel_row - .5
                    images[row, pixel_row, pixel_col] = 2 * x + 3 * y + z + weight
    path = tmp_path / "sax.npz"
    np.savez(path, images=images, masks=masks, group_idx=group_idx, weight_idx=weight_idx, stack_idx=stack_idx, pixel_spacing_rc_mm=spacing, timing9_ms=timing)
    poses = np.zeros((groups, 6), dtype=np.float64); poses[1, 5] = 2.0
    reference = build_slice_stack_reference(path, poses)
    sampled, valid, qc = sample_slice_stack(reference, np.array([[.0, .0, 1.0], [10., 0., 1.0]]))
    assert valid.tolist() == [True, False]
    assert np.allclose(sampled[0], np.arange(10) + 1.0)
    assert qc["orientation_spread_deg"] == 0.0
    assert qc["median_inter_plane_angle_deg"] == 0.0


def test_timing_parameter_export_records_frozen_geometry_and_anchor_order(tmp_path):
    class Dataset:
        xyz = torch.zeros((2, 3))
        stack_idx = torch.tensor([0, 1]); group_idx = torch.tensor([0, 1]); row_col = torch.tensor([[0, 0], [1, 1]])
    checkpoint = tmp_path / "model.pt"; checkpoint.write_bytes(b"model")
    geometry = tmp_path / "geometry.pt"; geometry.write_bytes(b"geometry")
    output = export_anchor_parameters(tmp_path / "timing.npz", t1_ms=np.array([1., 2.]), t2_ms=np.array([3., 4.]), b1=np.array([.9, 1.]), amplitude=np.array([2., 3.]), joint_dataset=Dataset(), reconstruction_checkpoint=checkpoint, geometry_checkpoint=geometry, geometry_provenance={"selected_key": "group_axisangle_post_stack_init_physical", "post_hb1_geometry_sha256": "a" * 64})
    with np.load(output, allow_pickle=False) as values:
        assert values["t1_ms"].tolist() == [1., 2.]
        assert values["amplitude"].tolist() == [2., 3.]
        assert len(values["anchor_order_sha256"].item()) == 64
    sidecar = output.with_suffix(".json")
    assert json.loads(sidecar.read_text())["trained_pose_key_not_used"] == "trained_axisangle_train"


def test_legacy_sparse_l50_uses_derived_observation_not_source(monkeypatch, tmp_path):
    source = tmp_path / "source"; derived = tmp_path / "derived"
    for base in (source, derived):
        for stack in ("sax", "2ch", "4ch"):
            path = base / stack / "observations.npz"; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"fixture")
    manifest = {"schema_version": "phase4r_sparse_v1", "case_id": "L50_fold0", "condition": "L50", "prepared_root": str(derived), "stacks": {stack: {"retained_original_group_ids": [0], "source_observation": {"path": str(source / stack / "observations.npz")}, "derived_observation": {"path": str(derived / stack / "observations.npz")}} for stack in ("sax", "2ch", "4ch")}}
    manifest_path = tmp_path / "manifest.json"; manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr("ismrm_v2.sampling.sparse_profile_manifest.validate_sparse_profile_manifest", lambda *args, **kwargs: manifest | {"_manifest_path": str(manifest_path), "_manifest_sha256": "a" * 64})
    paths, provenance = same_world._manifest_active_inputs(manifest_path, subject="S", case_id="L50_fold0", prepared_root=derived)
    assert all(str(path).startswith(str(derived)) for path in paths)
    assert provenance["input_observation_mode"] == "derived"
    assert all(item["mode"] == "derived" for item in provenance["observation_records"].values())
