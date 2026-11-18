import torch
import pytest
from types import SimpleNamespace
from ismrm_v2.evaluation.v2_3_stack_balance_audit import contribution_rows


def test_contributions_use_scalar_weights_and_joint_counts():
    scalar = SimpleNamespace(stack_idx=torch.tensor([0]*100 + [1]*50), validate_balanced_samples=lambda: {0: .75, 1: 1.5})
    joint = SimpleNamespace(stack_idx=torch.tensor([0]*8 + [1]*2), qc={'groups': {'0': {'stack_idx': 0, 'union_valid_count': 10}, '1': {'stack_idx': 0, 'union_valid_count': 1}, '2': {'stack_idx': 1, 'union_valid_count': 5}}})
    sparse = {'stacks': {'sax': {'retained_original_group_ids': [0, 1], 'heldout_original_group_ids': []}, '2ch': {'retained_original_group_ids': [0], 'heldout_original_group_ids': [1]}, '4ch': {'retained_original_group_ids': [], 'heldout_original_group_ids': [0]}}}
    rows = contribution_rows(scalar, joint, sparse, {'sax': 0, '2ch': 1})
    assert rows[0]['scalar_sample_count'] == 100 and rows[0]['joint_anchor_count'] == 8
    assert rows[0]['current_stack_weight'] == .75 and rows[0]['current_expected_weighted_fraction'] == 2/3
    assert rows[1]['current_expected_weighted_fraction'] == 1/3 and rows[1]['heldout_slice_count'] == 1
    assert rows[2]['current_stack_weight'] is None


def test_all_policies_and_group_validation():
    import pytest
    scalar = SimpleNamespace(stack_idx=torch.tensor([0]*100 + [1]*50), validate_balanced_samples=lambda: {0: .75, 1: 1.5})
    joint = SimpleNamespace(stack_idx=torch.tensor([0]*8 + [1]*2), qc={'groups': {'0': {'stack_idx': 0, 'union_valid_count': 10}, '1': {'stack_idx': 1, 'union_valid_count': 5}}})
    sparse = {'stacks': {'sax': {'retained_original_group_ids': [0], 'heldout_original_group_ids': [1]}, '2ch': {'retained_original_group_ids': [0], 'heldout_original_group_ids': []}, '4ch': {'retained_original_group_ids': [], 'heldout_original_group_ids': [0]}}}
    rows = contribution_rows(scalar, joint, sparse, {'sax': 0, '2ch': 1})
    assert rows[0]['off_policy_weight'] == 1 and rows[0]['off_policy_weighted_fraction'] == pytest.approx(.8)
    assert rows[0]['historical_effective_weighted_fraction'] == pytest.approx(2/3)
    assert rows[0]['slice_policy_weighted_fraction'] == pytest.approx(.5)
    assert rows[1]['slice_policy_weighted_fraction'] == pytest.approx(.5)
    assert rows[0]['heldout_slice_count'] == 1 and rows[0]['joint_to_scalar_ratio'] == pytest.approx(.08)
    sparse['stacks']['sax']['retained_original_group_ids'].append(2)
    with pytest.raises(ValueError, match='retained group'):
        contribution_rows(scalar, joint, sparse, {'sax': 0, '2ch': 1})


def test_full_audit_real_datasets_and_manifest(tmp_path):
    import json
    import numpy as np
    from pathlib import Path
    from ismrm_v2.sampling.sparse import build_sparse_profile_condition
    from ismrm_v2.evaluation.v2_3_stack_balance_audit import run
    groups = np.repeat(np.arange(4), 10)
    weights = np.tile(np.arange(10), 4)
    affine = np.repeat(np.eye(4)[None], len(groups), axis=0)
    affine[:, 2, 2] = 5.
    affine[:, 2, 3] = -groups
    source = tmp_path / 'source'
    for stack_id, stack in enumerate(('sax', '2ch', '4ch')):
        path = source / 'S' / stack / 'observations.npz'
        path.parent.mkdir(parents=True)
        masks = np.ones((len(groups), 3, 3), bool)
        for i, w in enumerate(weights):
            masks[i].flat[(w + stack_id) % (2 + stack_id)] = False
        np.savez_compressed(path, images=np.ones_like(masks, dtype=float), masks=masks,
            group_idx=groups, weight_idx=weights, stack_idx=np.full(len(groups), stack_id),
            acquisition_time_ms=np.zeros(len(groups)), timing9_ms=np.tile(np.arange(9), (len(groups), 1)),
            affine_lps_rc=affine, pixel_spacing_rc_mm=np.ones((len(groups), 2)),
            slice_thickness_mm=np.full(len(groups), 5.), tr_ms=np.full(len(groups), 2.61), vps=np.full(len(groups), 87))
    def audit(path):
        with np.load(path) as data:
            ids = np.unique(data['group_idx'])
            positions = [float(data['affine_lps_rc'][np.flatnonzero(data['group_idx'] == group)[0], 2, 3]) for group in ids]
        return {'regular_gap': True, 'gap_mm': abs(positions[1] - positions[0]) if len(ids) > 1 else 0., 'group_count': len(ids)}
    profile = Path(__file__).resolve().parents[2] / 'configs/sparse_profiles/S2CH50_4CH100.json'
    result = build_sparse_profile_condition(source, 'S', profile, 0, tmp_path / 'profile', git_sha='a'*40, audit_stack=audit)
    sparse = Path(result['manifest_path'])
    output = run(sparse.parent, sparse, tmp_path / 'result')
    manifest = json.loads((output / 'manifest.json').read_text())
    rows = json.loads((output / 'stack_contribution_audit.json').read_text())['stacks']
    assert (output / 'stack_contribution_audit.csv').is_file()
    assert manifest['subject'] == 'S' and len(manifest['prepared_inputs']) == 3
    for row in rows:
        assert row['scalar_sample_count'] == row['retained_slice_count'] * 80
        assert row['joint_anchor_count'] == row['retained_slice_count'] * (7 - row['stack_idx'])
        assert row['historical_equal_stack_weight'] == pytest.approx(sum(r['scalar_sample_count'] for r in rows) / (3 * row['scalar_sample_count']))
        assert row['slice_policy_weighted_fraction'] == pytest.approx(row['retained_slice_count'] / sum(r['retained_slice_count'] for r in rows))
