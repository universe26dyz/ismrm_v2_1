import nibabel as nib
import numpy as np
import pytest
import torch

from ismrm_v2.evaluation.v2_3_spatial_uncertainty_geometry_audit import pose_difference, native_world_grid, sample_volume, sigma_maps, coverage_maps, correlations, error_change


def test_pose_difference_and_same_volume_different_pose():
    a = np.eye(4)[:3]; b = a.copy(); b[0, 3] = 1
    delta = pose_difference(a, b)
    assert delta['rotation_deg'] == pytest.approx(0) and delta['translation_mm'] == pytest.approx(1)
    volume = np.broadcast_to(np.arange(4.)[:, None, None], (4, 3, 2)).copy()
    image = nib.Nifti1Image(volume, np.eye(4)); world = native_world_grid(np.eye(4), (2, 2)); learned = world.copy(); learned[..., 0] += 1
    original, original_valid = sample_volume(image, world); shifted, learned_valid = sample_volume(image, learned)
    assert original_valid.all() and learned_valid.all() and np.allclose(shifted - original, 1)
    bad = a.copy(); bad[0, 0] = 2
    with pytest.raises(ValueError): pose_difference(a, bad)


def test_sigma_map_actual_head_and_dimensions():
    class INR(torch.nn.Module):
        def forward_with_latent(self, xyz, include_latent): return {'latent': xyz}
    class Head(torch.nn.Module):
        def bounded_log_variance(self, latent): return torch.arange(10, device=latent.device).expand(len(latent), -1) * 0
    class Model(torch.nn.Module):
        def __init__(self): super().__init__(); self.parameter = torch.nn.Parameter(torch.zeros(1)); self.inr = INR(); self.joint_variance_head = Head()
    class Space:
        def physical_ras_to_train(self, x): return x
    maps = sigma_maps(Model(), Space(), np.zeros((2, 3, 3)))
    assert set(maps) == {'sigma_mean_across_weights', 'sigma_p90_across_weights', 'sigma_max_across_weights'}
    assert all(x.shape == (2, 3) and np.all(x == 1) for x in maps.values())
    with pytest.raises(ValueError): sigma_maps(Model(), Space(), np.full((1, 1, 3), np.nan))


def test_coverage_correlations_and_error_change():
    world = np.array([[[0., 0., 0.], [2., 0., 0.]]])
    support = {'sax': np.array([[0., 0., 0.]]), '2ch': np.array([[0., 0., 1.]]), '4ch': np.array([[0., 0., 4.]])}
    cov = coverage_maps(world, support, 'sax')
    assert cov['nearest_other_view_distance_mm'][0, 0] == pytest.approx(1)
    assert cov['other_view_count_within_1mm'][0, 0] == 1 and cov['other_view_count_within_5mm'][0, 0] == 2
    assert correlations([1, 2, 3], [3, 2, 1])['spearman'] == pytest.approx(-1)
    assert correlations([1, 1], [2, 3])['pearson'] is None
    assert error_change([0., 0.], [2., 1.], [1., 3.]).tolist() == [-1., 2.]
    with pytest.raises(ValueError): correlations([1, np.nan], [1, 2])


from test_three_variant_native import training_fixture


def test_spatial_pipeline_stratifies_rois_and_writes_outputs(tmp_path, monkeypatch, training_fixture):
    import json
    from types import SimpleNamespace
    from scipy.ndimage import binary_erosion
    from ismrm_v2.fit_first.contracts import sha256
    import ismrm_v2.evaluation.v2_3_spatial_uncertainty_geometry_audit as module
    sparse, payload, roots = training_fixture
    payload['stacks']['sax']['heldout_original_group_ids'] = [1]
    payload['stacks']['4ch']['heldout_original_group_ids'] = [1]
    monkeypatch.setattr(module, 'validate_sparse_profile_manifest', lambda *a, **kw: payload)
    monkeypatch.setattr(module, '_sparse_contract', lambda *a: {})
    def fake_authority(root, *args):
        i = roots.index(root)
        return {'kind': 'mse' if i == 0 else 'hgnll', 'detach': i == 2, 'policy': 'equal_total_stack', 'scientific': {}, 'hgnll': {'variance_detach_shared_latent': i == 2}}
    monkeypatch.setattr(module, 'read_authority', fake_authority)
    monkeypatch.setattr(module, 'verify_policy_weights', lambda *a: {})
    paths = []
    for stack, idx in (('sax', 0), ('4ch', 2)):
        path = tmp_path / 'prepared' / stack / 'observations.npz'; path.parent.mkdir(parents=True)
        np.savez(path, group_idx=np.zeros(10, int), stack_idx=np.full(10, idx)); paths.append(path)
    monkeypatch.setattr(module, 'profiled_observation_paths', lambda *a: paths)
    monkeypatch.setattr(module, 'QuantPointDataset', lambda *a, **kw: object())
    joint = SimpleNamespace(group_idx=torch.tensor([0, 1]), stack_idx=torch.tensor([0, 2]), xyz=torch.tensor([[0., 0., 0.], [0., 0., 1.]]))
    monkeypatch.setattr(module, 'JointAnchorDataset', lambda *a: joint)
    class INR(torch.nn.Module):
        def forward_with_latent(self, xyz, include_latent): return {'latent': xyz}
    class Head(torch.nn.Module):
        def bounded_log_variance(self, latent): return torch.zeros((len(latent), 10))
    class Rigid(torch.nn.Module):
        def __init__(self): super().__init__(); self.axisangle = torch.nn.Parameter(torch.zeros((2, 6)))
        def transform_local_to_ras(self, xyz, groups): return xyz
    class Model(torch.nn.Module):
        def __init__(self): super().__init__(); self.rigid_psf = Rigid(); self.inr = INR(); self.joint_variance_head = Head()
    class Space:
        group_axisangle_dicom_physical = torch.zeros((2, 6))
        def axisangle_train_to_physical(self, x): return x
        def physical_ras_to_train(self, x): return x
        def train_ras_to_physical(self, x): return x
        def local_to_train(self, x): return x
    monkeypatch.setattr(module, '_load_model', lambda *a, **kw: (Model(), Space(), torch))
    monkeypatch.setattr(module, '_plot', lambda output, *a, **kw: (output.parent.mkdir(parents=True, exist_ok=True), output.write_bytes(b'plot')))
    for root in roots: (root / 'final_rigid_poses.json').write_text(json.dumps({'matrix_physical': [np.eye(4)[:3].tolist()] * 2}))
    roi = tmp_path / 'roi'; roi.mkdir(); outer = np.zeros((2, 12, 12), bool); outer[:, 2:10, 2:10] = True
    inner = np.zeros_like(outer); inner[:, 5:7, 5:7] = True
    full = outer & ~inner; core = np.stack([binary_erosion(x, structure=np.ones((3, 3), bool)) for x in full]); archive = roi / 'sax_myocardium_masks.npz'
    np.savez(archive, group_idx=np.arange(2), outer_filled=outer, inner_filled=inner, blood_pool=inner, myocardium_full=full, myocardium_core_legacy=core, myocardium_core_1px=core, landmark_labels=np.zeros_like(outer))
    (roi / 'manifest.json').write_text(json.dumps({'schema': 'exact_native_myocardium_bundle/v2', 'status': 'PASS', 'subject_id': 'S', 'outputs': {archive.name: sha256(archive)}}))
    def cache(n, affines): return {'group_idx': np.arange(n), 't1_ms': np.full((n, 12, 12), 100.), 't2_ms': np.full((n, 12, 12), 50.), 'affine_lps_rc': affines, 'pixel_spacing_rc_mm': np.ones((n, 2)), 'slice_thickness_mm': np.ones(n), 'source_support': np.ones((n, 12, 12), bool), 'provenance': {}}
    lps = np.diag([-1., -1., 1., 1.]); sax = np.repeat(lps[None], 2, axis=0); sax[1, 2, 3] = 1
    target = np.array([[1., 0., 0., 0.], [0., 0., 1., 3.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
    caches = {'sax': cache(2, sax), '2ch': cache(1, (lps @ target)[None]), '4ch': cache(2, np.repeat((lps @ target)[None], 2, axis=0))}
    monkeypatch.setattr(module, '_load_all_verified_native_cache_groups', lambda root, path, stack: caches[stack])
    native = tmp_path / 'native'; native.mkdir()
    out = module.run(mse_root=roots[0], h10_root=roots[1], detach_root=roots[2], prepared_root=tmp_path/'prepared', r0_native_maps_root=native, sparse_manifest=sparse, roi_bundle=roi, output=tmp_path/'result')
    import csv
    for filename in ('pose_per_group.csv', 'pose_summary.csv', 'learned_vs_native_pose_metrics.csv', 'spatial_sigma_summary.csv', 'coverage_summary.csv', 'residual_sigma_coverage_correlations.csv', 'h10_vs_detach_change_summary.csv'):
        assert (out / filename).is_file()
    with (out / 'learned_vs_native_pose_metrics.csv').open() as handle: records = list(csv.DictReader(handle))
    assert {r['pose'] for r in records} == {'learned', 'native'}
    assert {r['role'] for r in records} == {'observed', 'heldout'}
    assert {r['roi'] for r in records} == {'myocardium_core_1px', 'blood_pool', 'lv_outer_filled'}
    with (out / 'residual_sigma_coverage_correlations.csv').open() as handle: corr = list(csv.DictReader(handle))
    assert {'myocardium_core_1px', 'myocardium_full', 'peri_myocardium_ring', 'blood_pool'} <= {r['roi'] for r in corr}
    assert {'torch_hgnll10', 'torch_hgnll10_detach'} == {r['method'] for r in corr}
    assert {'observed', 'heldout'} == {r['role'] for r in corr}
    assert (out / 'maps' / 'sax' / 'group000.npz').is_file()


def test_training_space_roundoff_is_bounded():
    from ismrm_v2.evaluation.direct_native_endpoint_audit import _validate_rebuilt_training_space
    from types import SimpleNamespace
    keys = ("center_ras_mm", "spatial_scaling", "physical_bbox_ras_mm", "bbox_train", "group_resolution_xyz_mm", "group_axisangle_init_physical", "group_axisangle_init_train")
    saved = {key: torch.tensor([1.0]) for key in keys}
    current = {key: value.clone() for key, value in saved.items()}
    rebuilt = SimpleNamespace(state_dict=lambda: current)
    current["center_ras_mm"] += 2e-5
    with pytest.raises(ValueError): _validate_rebuilt_training_space(rebuilt, saved, torch)
    _validate_rebuilt_training_space(rebuilt, saved, torch, geometry_roundoff_atol=3.2e-5)
    current["center_ras_mm"] += 1e-3
    with pytest.raises(ValueError): _validate_rebuilt_training_space(rebuilt, saved, torch, geometry_roundoff_atol=3.2e-5)
    current["center_ras_mm"] = saved["center_ras_mm"].clone()
    current["group_axisangle_init_physical"] += 2e-5
    with pytest.raises(ValueError): _validate_rebuilt_training_space(rebuilt, saved, torch, geometry_roundoff_atol=3.2e-5)
