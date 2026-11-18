"""Fail-closed identity and scientific comparison of stack-policy runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch

from ismrm_v2.fit_first.contracts import sha256
from ismrm_v2.evaluation.v2_3_h10_detach_native_metrics import _backend_provenance, _resolve_run_log
from ismrm_v2.v2_3.hgnll import HGNLLConfig
from ismrm_v2.proposed.q002.stack_policy import resolve_stack_policy

ROUTES = {
    'B1_R0': ('mse', 'equal_total_stack', False),
    'B1_R0_STACK_OFF': ('mse', 'off', False),
    'B1_R0_STACK_SLICE': ('mse', 'slice_proportional_total', False),
    'B1_R0_HGNLL': ('hgnll', 'equal_total_stack', False),
    'B1_R0_HGNLL_DETACH': ('hgnll', 'equal_total_stack', True),
    'B1_R0_HGNLL_STACK_SLICE': ('hgnll', 'slice_proportional_total', False),
    'B1_R0_HGNLL_DETACH_STACK_SLICE': ('hgnll', 'slice_proportional_total', True),
}
LEGACY = {'B1_R0', 'B1_R0_HGNLL', 'B1_R0_HGNLL_DETACH'}
CONTROLS = {'stage_a_iterations': 2000, 'stage_b_iterations': 4000, 'anchor_batch_size': 640,
            'weights_per_anchor': 10, 'training_psf_samples': 8, 'seed': 20260911}


def _json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def read_authority(root, sparse, sparse_path):
    root = Path(root)
    first = json.loads((root / 'experiment_manifest.json').read_text())
    if first != json.loads((root / 'q002_route_manifest.json').read_text()):
        raise ValueError('training manifests disagree')
    route_id = first.get('experiment_id')
    if route_id not in ROUTES or first.get('route', {}).get('experiment_id') != route_id:
        raise ValueError('unknown or contradictory route identity')
    kind, policy, detach = ROUTES[route_id]
    route = first['route']
    if (first.get('subject_id'), first.get('case_id'), first.get('profile_id')) != (sparse.get('subject'), sparse.get('case_id'), sparse.get('profile_id')):
        raise ValueError('sparse identity mismatch')
    if first.get('sparse_manifest', {}).get('sha256') != sha256(sparse_path):
        raise ValueError('sparse manifest mismatch')
    expected_inputs = {s: item.get('derived_observation', item.get('source_observation'))['sha256'] for s, item in sparse['stacks'].items() if item.get('retained_original_group_ids')}
    inputs = {}
    for path, digest in first.get('prepared_inputs_sha256', {}).items():
        name = Path(path).parent.name
        if name in inputs or name not in ('sax', '2ch', '4ch'):
            raise ValueError('ambiguous prepared stack identity')
        inputs[name] = digest
    if not inputs or inputs != expected_inputs:
        raise ValueError('prepared input hashes disagree with sparse manifest')
    controls = first.get('controls', {})
    if any(first.get(k, v) != v for k, v in CONTROLS.items()):
        raise ValueError('manifest contradicts fixed training controls')
    if any(controls.get(k) != v for k, v in CONTROLS.items()):
        raise ValueError('training controls mismatch')
    sampling = route.get('sampling', {})
    if (sampling.get('anchor_batch_size'), sampling.get('weights_per_anchor'), sampling.get('shared_training_psf_samples')) != (640, 10, 8):
        raise ValueError('sampling mismatch')
    cosine = route.get('fingerprint_cosine', {'enabled': False})
    if not isinstance(cosine, Mapping) or cosine.get('enabled', False) is not False or first.get('fingerprint_cosine_enabled', False) is not False:
        raise ValueError('fingerprint cosine must be off for stack-policy ablations')
    reliability = route.get('cross_view_reliability', {})
    if first.get('cross_view_reliability_enabled', False) is not False:
        raise ValueError('manifest reliability must be off')
    if reliability.get('enabled') is not False:
        raise ValueError('reliability must be off')
    balance = route.get('stack_balance', {})
    if balance.get('enabled') is not (policy != 'off'):
        raise ValueError('route stack balance mismatch')
    if route_id in LEGACY:
        if first.get('stack_balance_enabled', True) is not True:
            raise ValueError('historical manifest stack balance mismatch')
        if balance.get('policy', 'equal_total_stack') != 'equal_total_stack':
            raise ValueError('historical route policy mismatch')
    elif balance.get('policy') != policy or first.get('stack_balance_policy') != policy or first.get('stack_balance_enabled') is not (policy != 'off'):
        raise ValueError('explicit route policy mismatch')
    checkpoint_path = root / 'model.pt'
    if first.get('q002_checkpoint_sha256') != sha256(checkpoint_path):
        raise ValueError('checkpoint hash mismatch')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('resolved_config')
    if not isinstance(config, Mapping):
        raise ValueError('checkpoint lacks resolved configuration')
    ck_balance = config.get('stack_balance', {})
    if ck_balance.get('enabled') is not (policy != 'off'):
        raise ValueError('checkpoint stack balance mismatch')
    if route_id in LEGACY:
        if ck_balance.get('policy', 'equal_total_stack') != 'equal_total_stack':
            raise ValueError('historical checkpoint policy mismatch')
    else:
        if ck_balance.get('policy') != policy or not isinstance(config.get('stack_policy_qc'), Mapping):
            raise ValueError('checkpoint explicit stack policy missing or mismatched')
        qc = config['stack_policy_qc']
        if _json_value(first.get('stack_weights')) != _json_value(qc.get('resolved_stack_weights')):
            raise ValueError('manifest stack weights disagree with checkpoint')
        for key in ('stack_balance_enabled', 'stack_balance_policy', 'stack_weight_source', 'stack_weight_formula', 'stack_weight_normalization', 'retained_slice_count_by_stack', 'joint_anchor_count_by_stack', 'resolved_stack_weights', 'expected_weighted_fraction_by_stack'):
            if key not in first or key not in qc or _json_value(first[key]) != _json_value(qc[key]):
                raise ValueError(f'checkpoint and manifest stack metadata mismatch: {key}')
    hgnll = None
    if kind == 'hgnll':
        h = route.get('heteroscedastic')
        if not isinstance(h, Mapping) or first.get('hgnll') != h or first.get('loss') != 'HGNLL':
            raise ValueError('HGNLL route metadata mismatch')
        hgnll = HGNLLConfig.from_mapping(h, required=True).as_dict()
        if hgnll['variance_detach_shared_latent'] is not detach or hgnll['variance_output_dim'] != 10:
            raise ValueError('HGNLL policy mismatch')
        if HGNLLConfig.from_mapping(checkpoint.get('hgnll_config'), required=True).as_dict() != hgnll or HGNLLConfig.from_mapping(config.get('heteroscedastic'), required=True).as_dict() != hgnll:
            raise ValueError('checkpoint HGNLL policy mismatch')
        state = checkpoint.get('model_state', {})
        for key, shape in (('joint_variance_head.network.0.weight', (32, 32)), ('joint_variance_head.network.0.bias', (32,)), ('joint_variance_head.network.2.weight', (10, 32)), ('joint_variance_head.network.2.bias', (10,))):
            if key not in state or tuple(state[key].shape) != shape:
                raise ValueError('checkpoint HGNLL head shape mismatch')
    elif route.get('heteroscedastic', {}).get('enabled', False) is not False or config.get('heteroscedastic', {}).get('enabled', False) is not False or checkpoint.get('hgnll_config', {}).get('enabled', False) is not False or first.get('loss', 'MSE').upper() not in ('MSE', 'MSE_ONLY'):
        raise ValueError('MSE route unexpectedly enables HGNLL')
    dependencies = first.get('dependency_sha256', {})
    for key, field in (('B6_model', 'b6_model_sha256'), ('frozen_decoder', 'approved_decoder_sha256'), ('protocol', 'protocol_sha256')):
        if not first.get(field) or dependencies.get(key) != first[field]:
            raise ValueError('scientific dependency mismatch')
    backend = _backend_provenance(_resolve_run_log(root), True)
    comparable_config = dict(config)
    for key in ('stack_balance', 'stack_policy_qc', 'heteroscedastic'):
        comparable_config.pop(key, None)
    scientific = {'subject': first['subject_id'], 'case': first['case_id'], 'profile': first['profile_id'],
                  'sparse_sha256': sha256(sparse_path), 'prepared': inputs, 'dependencies': dependencies,
                  'controls': controls, 'sampling': sampling, 'reliability': reliability,
                  'config': _json_value(comparable_config)}
    return {'root': str(root.resolve()), 'route': route_id, 'kind': kind, 'policy': policy,
            'detach': detach, 'hgnll': hgnll, 'scientific': scientific, 'backend': backend,
            'checkpoint_sha256': sha256(checkpoint_path), 'stack_policy_qc': _json_value(config.get('stack_policy_qc')) if route_id not in LEGACY else None}


def compare_authorities(authorities):
    if len(authorities) < 2:
        raise ValueError('at least two runs are required')
    if any(a['scientific'] != authorities[0]['scientific'] for a in authorities[1:]):
        raise ValueError('scientific settings differ beyond the permitted policy fields')
    if len({a['kind'] for a in authorities}) != 1:
        raise ValueError('mixed MSE and HGNLL is not a stack-policy ablation')
    if authorities[0]['kind'] == 'hgnll':
        policies = []
        for authority in authorities:
            h = dict(authority['hgnll'])
            h.pop('variance_detach_shared_latent')
            policies.append(h)
        if any(h != policies[0] for h in policies[1:]):
            raise ValueError('HGNLL policies differ beyond latent detach')
    return {'status': 'PASS', 'allowed_differences': ['stack_balance_policy', 'variance_detach_shared_latent' if authorities[0]['kind'] == 'hgnll' else 'none'], 'runs': authorities}


def verify_policy_weights(authority, scalar, joint):
    """Re-resolve weights from the verified prepared data, never trust stored QC alone."""
    policy = authority['policy']
    _, actual = resolve_stack_policy(scalar, joint, {'enabled': policy != 'off', 'policy': policy})
    if authority['stack_policy_qc'] is not None and _json_value(actual) != authority['stack_policy_qc']:
        raise ValueError('checkpoint stack policy metadata disagrees with prepared-data counts and weights')
    return actual
