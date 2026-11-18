import copy
import json

import pytest
import torch

from ismrm_v2.fit_first.contracts import sha256
from ismrm_v2.proposed.q002.contracts import build_q002_route_config
from ismrm_v2.proposed.q002.training import q002_training_controls
from ismrm_v2.evaluation.v2_3_stack_policy_authority import ROUTES, read_authority, compare_authorities


def make_root(tmp_path, name, sparse, sparse_path):
    root = tmp_path / name; root.mkdir()
    training = {'stage_a_iterations': 2000, 'stage_b_iterations': 4000, 'batch_size': 640, 'psf_samples': 8, 'seed': 20260911}
    route = build_q002_route_config({'training': training, 'decoder': {}}, cropped_prepared_root='cropped', experiment_id=name, anchor_batch_size=640)['route']
    kind, policy, detach = ROUTES[name]
    cfg = {'training': training, 'decoder': {}, 'stack_balance': copy.deepcopy(route['stack_balance'])}
    m = {'experiment_id': name, 'route': route, 'subject_id': 'S', 'case_id': 'L50', 'profile_id': 'p', 'sparse_manifest': {'sha256': sha256(sparse_path)},
         'prepared_inputs_sha256': {'/data/sax/observations.npz': 'input'}, 'controls': q002_training_controls(training, anchor_batch_size=640),
         'b6_model_sha256': 'b6', 'approved_decoder_sha256': 'decoder', 'protocol_sha256': 'protocol', 'dependency_sha256': {'B6_model': 'b6', 'frozen_decoder': 'decoder', 'protocol': 'protocol'}}
    if name not in ('B1_R0', 'B1_R0_HGNLL', 'B1_R0_HGNLL_DETACH'):
        qc = {'stack_balance_enabled': policy != 'off', 'stack_balance_policy': policy, 'stack_weight_source': 'source', 'stack_weight_formula': 'formula', 'stack_weight_normalization': 'norm', 'retained_slice_count_by_stack': {'0': 1}, 'joint_anchor_count_by_stack': {'0': 10}, 'resolved_stack_weights': {'0': 1}, 'expected_weighted_fraction_by_stack': {'0': 1}}
        cfg['stack_policy_qc'] = qc; m.update(qc); m['stack_weights'] = qc['resolved_stack_weights']
    checkpoint = {'resolved_config': cfg, 'model_state': {}}
    if kind == 'hgnll':
        h = route['heteroscedastic']; m.update({'loss': 'HGNLL', 'hgnll': h}); checkpoint['hgnll_config'] = h; cfg['heteroscedastic'] = h
        for key, shape in (('joint_variance_head.network.0.weight', (32, 32)), ('joint_variance_head.network.0.bias', (32,)), ('joint_variance_head.network.2.weight', (10, 32)), ('joint_variance_head.network.2.bias', (10,))): checkpoint['model_state'][key] = torch.zeros(shape)
    torch.save(checkpoint, root / 'model.pt'); m['q002_checkpoint_sha256'] = sha256(root / 'model.pt')
    for filename in ('experiment_manifest.json', 'q002_route_manifest.json'): (root / filename).write_text(json.dumps(m))
    (root / 'run.log').write_text('USE_TORCH = True\ndeterministic_algorithms = True\nCUBLAS_WORKSPACE_CONFIG = :4096:8\n')
    return root


def test_comparability_and_fail_closed(tmp_path):
    sparse = {'subject': 'S', 'case_id': 'L50', 'profile_id': 'p', 'stacks': {'sax': {'retained_original_group_ids': [0], 'derived_observation': {'sha256': 'input'}}, '2ch': {}, '4ch': {}}}
    sparse_path = tmp_path / 'sparse.json'; sparse_path.write_text(json.dumps(sparse))
    roots = {name: make_root(tmp_path, name, sparse, sparse_path) for name in ROUTES}
    authorities = {name: read_authority(root, sparse, sparse_path) for name, root in roots.items()}
    assert compare_authorities([authorities[x] for x in ('B1_R0', 'B1_R0_STACK_OFF', 'B1_R0_STACK_SLICE')])['status'] == 'PASS'
    assert compare_authorities([authorities[x] for x in ('B1_R0_HGNLL', 'B1_R0_HGNLL_STACK_SLICE', 'B1_R0_HGNLL_DETACH_STACK_SLICE')])['status'] == 'PASS'
    altered = copy.deepcopy(authorities['B1_R0_STACK_SLICE']); altered['scientific']['controls']['seed'] = 0
    with pytest.raises(ValueError, match='scientific settings'): compare_authorities([authorities['B1_R0'], altered])
    path = roots['B1_R0_STACK_OFF'] / 'model.pt'
    ck = torch.load(path, weights_only=False); ck['resolved_config']['stack_balance']['policy'] = 'equal_total_stack'; torch.save(ck, path)
    for filename in ('experiment_manifest.json', 'q002_route_manifest.json'):
        file = roots['B1_R0_STACK_OFF'] / filename; m = json.loads(file.read_text()); m['q002_checkpoint_sha256'] = sha256(path); file.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='checkpoint explicit stack policy'): read_authority(roots['B1_R0_STACK_OFF'], sparse, sparse_path)


def test_recompute_policy_from_actual_joint_counts():
    from types import SimpleNamespace
    from ismrm_v2.proposed.q002.stack_policy import resolve_stack_policy
    from ismrm_v2.evaluation.v2_3_stack_policy_authority import verify_policy_weights, _json_value
    scalar = SimpleNamespace(validate_balanced_samples=lambda: {0: .75, 1: 1.5})
    joint = SimpleNamespace(stack_idx=torch.tensor([0] * 8 + [1] * 2), qc={'groups': {'a': {'stack_idx': 0}, 'b': {'stack_idx': 1}}})
    _, qc = resolve_stack_policy(scalar, joint, {'enabled': True, 'policy': 'slice_proportional_total'})
    authority = {'policy': 'slice_proportional_total', 'stack_policy_qc': _json_value(qc)}
    assert verify_policy_weights(authority, scalar, joint)['expected_weighted_fraction_by_stack'][0] == pytest.approx(.5)
    authority['stack_policy_qc']['resolved_stack_weights']['0'] = 123
    with pytest.raises(ValueError, match='prepared-data counts'): verify_policy_weights(authority, scalar, joint)
