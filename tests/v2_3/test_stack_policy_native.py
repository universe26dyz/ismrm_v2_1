import json
import numpy as np
import pytest
from test_three_variant_native import training_fixture

from ismrm_v2.evaluation.v2_3_stack_policy_native_metrics import common_support, paired_deltas


def test_common_support_and_delta_sign():
    ref = np.array([1., 2., 3.]); pred = {'a': ref.copy(), 'b': ref + 1}
    valid = {'a': np.ones(3, bool), 'b': np.array([True, False, True])}
    assert common_support(ref, pred, valid, np.ones(3, bool)).tolist() == [True, False, True]
    stat = {'methods': {'h10_equal': {'rmse': 10, 'mae': 4, 'nrmse': .5}, 'h10_slice': {'rmse': 8, 'mae': 5, 'nrmse': .4}, 'detach_equal': None, 'detach_slice': None}}
    assert paired_deltas(stat)['h10_slice_vs_h10_equal']['rmse'] == pytest.approx(20)
    assert paired_deltas(stat)['h10_slice_vs_h10_equal']['mae'] == pytest.approx(-25)


def test_native_pipeline_roles_and_common_support(tmp_path, monkeypatch, training_fixture):
    import ismrm_v2.evaluation.v2_3_stack_policy_native_metrics as module
    from ismrm_v2.fit_first.contracts import sha256
    from scipy.ndimage import binary_erosion
    sparse, payload, roots = training_fixture
    payload['prepared_root'] = str(tmp_path / 'prepared')
    payload['stacks']['sax']['heldout_original_group_ids'] = [1]
    payload['stacks']['4ch']['heldout_original_group_ids'] = [1]
    names = list(module.MODES['mse'])
    methods = dict(zip(names, roots))
    monkeypatch.setattr(module, 'validate_sparse_profile_manifest', lambda *a, **kw: payload)
    monkeypatch.setattr(module, 'profiled_observation_paths', lambda *a: [])
    monkeypatch.setattr(module, 'QuantPointDataset', lambda *a, **kw: None)
    monkeypatch.setattr(module, 'JointAnchorDataset', lambda *a: None)
    monkeypatch.setattr(module, 'verify_policy_weights', lambda *a: {})
    monkeypatch.setattr(module, 'read_authority', lambda root, *a: {'route': module.MODES['mse'][names[roots.index(root)]], 'scientific': {}, 'kind': 'mse'})
    roiroot = tmp_path / 'roi'; roiroot.mkdir()
    outer = np.zeros((2, 12, 12), bool); outer[:, 2:10, 2:10] = True
    inner = np.zeros_like(outer); inner[:, 5:7, 5:7] = True
    full = outer & ~inner; core = np.stack([binary_erosion(x, structure=np.ones((3, 3), bool)) for x in full])
    archive = roiroot / 'sax_myocardium_masks.npz'
    np.savez(archive, group_idx=np.arange(2), outer_filled=outer, inner_filled=inner, blood_pool=inner, myocardium_full=full, myocardium_core_legacy=core, myocardium_core_1px=core, landmark_labels=np.zeros_like(outer))
    (roiroot / 'manifest.json').write_text(json.dumps({'schema': 'exact_native_myocardium_bundle/v2', 'status': 'PASS', 'subject_id': 'S', 'outputs': {archive.name: sha256(archive)}}))
    def cache(n, affines):
        return {'group_idx': np.arange(n), 't1_ms': np.full((n, 12, 12), 100.), 't2_ms': np.full((n, 12, 12), 50.), 'affine_lps_rc': affines, 'pixel_spacing_rc_mm': np.ones((n, 2)), 'slice_thickness_mm': np.ones(n), 'source_support': np.ones((n, 12, 12), bool), 'provenance': {}}
    lps = np.diag([-1., -1., 1., 1.]); sax = np.repeat(lps[None], 2, axis=0); sax[1, 2, 3] = 1
    target = np.array([[1., 0., 0., 0.], [0., 0., 1., 3.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
    caches = {'sax': cache(2, sax), '2ch': cache(1, (lps @ target)[None]), '4ch': cache(2, np.repeat((lps @ target)[None], 2, axis=0))}
    monkeypatch.setattr(module, '_load_all_verified_native_cache_groups', lambda root, path, stack: caches[stack])
    monkeypatch.setattr(module, 'render_plane', lambda ref, pred, support, roi, p, output, **kw: {'path': str(output)})
    native = tmp_path / 'native'; native.mkdir()
    out = module.run(mode='mse', roots=methods, r0_native_maps_root=native, sparse_manifest=sparse, roi_bundle=roiroot, output=tmp_path / 'result')
    data = json.loads((out / 'aggregate_metrics.json').read_text())['partitions']['t1']
    assert data['observed']['common_native']['methods'][names[0]]['bias'] == 10
    assert data['heldout']['common_native']['methods'][names[0]]['bias'] == 20
    assert data['stack_sax']['myocardium_core_1px']['support_voxels'] > 0
    assert data['observed_sax']['common_native']['support_voxels'] > 0
    assert data['heldout_2ch']['myocardium_core_1px']['support_voxels'] > 0
    assert data['heldout_4ch']['myocardium_core_1px']['support_voxels'] > 0
