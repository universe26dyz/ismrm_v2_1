"""Native-space stack-policy comparisons with shared support and strict authority."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from ismrm_v2.evaluation.evaluation import _metrics, PSNR_DATA_RANGES_MS, reproject_volume_to_native_plane
from ismrm_v2.evaluation.final_cyj_reporting import _load_all_verified_native_cache_groups
from ismrm_v2.evaluation.final_native_grid_metrics import native_ras_affine
from ismrm_v2.evaluation.aha.cyj11_myocardial_roi_metrics import PRIMARY_ROI, _load_scmr_masks
from ismrm_v2.evaluation.aha.paired_final_pose_lv_metrics import load_outer_filled_masks
from ismrm_v2.evaluation.aha.common_native_all_stack_roi_metrics import ALL_ROIS, load_legacy_blood_pool_masks, stack_sax_masks_as_volume, project_sax_mask_volume
from ismrm_v2.evaluation.v2_3_h10_detach_native_metrics import _clean, _derived_groups, _observed_plane, _record, _validate_roi_bundle_manifest
from ismrm_v2.evaluation.v2_3_stack_policy_authority import read_authority, compare_authorities, verify_policy_weights
from ismrm_v2.proposed.q002.contracts import profiled_observation_paths
from ismrm_v2.proposed.q002.joint_dataset import JointAnchorDataset
from ismrm_v2.proposed.quantitative_point_dataset import QuantPointDataset
from ismrm_v2.sampling.sparse_profile_manifest import validate_sparse_profile_manifest

STACKS = ('sax', '2ch', '4ch')
PARAMETERS = ('t1', 't2')
MODES = {
    'mse': {'mse_stack_off': 'B1_R0_STACK_OFF', 'mse_stack_equal': 'B1_R0', 'mse_stack_slice': 'B1_R0_STACK_SLICE'},
    'hgnll_off': {'h10_equal': 'B1_R0_HGNLL', 'h10_off': 'B1_R0_HGNLL_STACK_OFF', 'detach_equal': 'B1_R0_HGNLL_DETACH', 'detach_off': 'B1_R0_HGNLL_DETACH_STACK_OFF'},
    'hgnll': {'h10_equal': 'B1_R0_HGNLL', 'h10_slice': 'B1_R0_HGNLL_STACK_SLICE', 'detach_equal': 'B1_R0_HGNLL_DETACH', 'detach_slice': 'B1_R0_HGNLL_DETACH_STACK_SLICE'},
}
LABELS = {'h10_off': 'H10 + SB-OFF', 'detach_off': 'Detach + SB-OFF', 'mse_stack_off': 'MSE + SB-OFF', 'mse_stack_equal': 'MSE + SB-EQUAL', 'mse_stack_slice': 'MSE + SB-SLICE', 'h10_equal': 'H10 + SB-EQUAL', 'h10_slice': 'H10 + SB-SLICE', 'detach_equal': 'Detach + SB-EQUAL', 'detach_slice': 'Detach + SB-SLICE'}
PAIRS = {'h10_slice_vs_h10_equal': ('h10_equal', 'h10_slice'), 'detach_slice_vs_detach_equal': ('detach_equal', 'detach_slice'), 'detach_slice_vs_h10_slice': ('h10_slice', 'detach_slice')}


def common_support(reference, predictions, valid, source):
    shape = np.asarray(reference).shape
    if set(predictions) != set(valid) or not predictions or np.asarray(source).shape != shape:
        raise ValueError('prediction, validity and source support must match')
    support = np.asarray(source, bool) & np.isfinite(reference)
    for name in predictions:
        if np.asarray(predictions[name]).shape != shape or np.asarray(valid[name]).shape != shape:
            raise ValueError('native arrays must have identical shape')
        support &= np.asarray(valid[name], bool) & np.isfinite(predictions[name])
    return support


def statistics(reference, predictions, support, parameter):
    count = int(np.count_nonzero(support))
    return {'support_voxels': count, 'methods': {name: _metrics(reference, value, support, psnr_data_range=PSNR_DATA_RANGES_MS[parameter]) if count else None for name, value in predictions.items()}}


OFF_PAIRS = {'h10_off_vs_h10_equal': ('h10_equal', 'h10_off'), 'detach_off_vs_detach_equal': ('detach_equal', 'detach_off'), 'detach_off_vs_h10_off': ('h10_off', 'detach_off')}


def paired_deltas(stat, pairs=PAIRS):
    rows = {}
    for name, (baseline, candidate) in pairs.items():
        a, b = stat['methods'].get(baseline), stat['methods'].get(candidate)
        if a is None or b is None:
            rows[name] = None
            continue
        rows[name] = {metric: (100. * (a[metric] - b[metric]) / a[metric] if a.get(metric) is not None and b.get(metric) is not None and np.isfinite(a[metric]) and np.isfinite(b[metric]) and a[metric] != 0 else None) for metric in ('rmse', 'mae', 'nrmse')}
    return rows


def render_plane(reference, predictions, support, cardiac_roi, parameter, output, *, title=''):
    import matplotlib.pyplot as plt
    from ismrm_v2.evaluation.visualization import _crop_bounds, colorbars
    from ismrm_v2.evaluation.scmr.quality_control import DISPLAY_RANGES_MS, RESIDUAL_RANGES_MS
    r0, r1, c0, c1 = _crop_bounds(np.asarray(cardiac_roi, bool), margin=3)
    sl = np.s_[r0:r1, c0:c1]; key = parameter.upper(); limit = RESIDUAL_RANGES_MS[key][1]
    fig, axes = plt.subplots(2, len(predictions) + 1, figsize=(3*(len(predictions)+1), 6), constrained_layout=True)
    fig.suptitle(title); bars = colorbars()
    for ax, (label, value) in zip(axes[0], [('Native 2D', reference)] + [(LABELS[name], x) for name, x in predictions.items()]):
        image = ax.imshow(np.ma.masked_where(~support[sl], np.asarray(value)[sl]), origin='lower', interpolation='nearest', cmap=bars.mapping[key], vmin=DISPLAY_RANGES_MS[key][0], vmax=DISPLAY_RANGES_MS[key][1]); ax.set_title(label); ax.axis('off')
    fig.colorbar(image, ax=list(axes[0]), label=f'{key} (ms)', shrink=.8); axes[1, 0].axis('off')
    for ax, (name, value) in zip(axes[1, 1:], predictions.items()):
        delta = np.asarray(value) - reference
        residual = ax.imshow(np.ma.masked_where(~support[sl], delta[sl]), origin='lower', interpolation='nearest', cmap='coolwarm', vmin=-limit, vmax=limit); ax.set_title(LABELS[name] + ' − native'); ax.axis('off')
    fig.colorbar(residual, ax=list(axes[1, 1:]), label=f'Δ{key} (ms)', shrink=.8)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=160, facecolor='white'); plt.close(fig)
    return {'path': str(output), 'map_range_ms': list(DISPLAY_RANGES_MS[key]), 'residual_range_ms': [-limit, limit]}


def run(*, mode, roots, r0_native_maps_root, sparse_manifest, roi_bundle, output):
    if mode not in MODES or set(roots) != set(MODES[mode]):
        raise ValueError('mode requires its exact declared method roots')
    roots = {name: Path(path) for name, path in roots.items()}; sparse_path = Path(sparse_manifest)
    sparse = validate_sparse_profile_manifest(sparse_path, strict=False)
    group_mapping = _derived_groups(sparse); subject, case = sparse['subject'], sparse['case_id']
    output, roi_bundle = Path(output).resolve(), Path(roi_bundle)
    if output.exists(): raise FileExistsError(output)
    if any(output.is_relative_to(p.resolve()) for p in [*roots.values(), Path(r0_native_maps_root), roi_bundle]): raise ValueError('output must be outside inputs')
    authorities = {name: read_authority(root, sparse, sparse_path) for name, root in roots.items()}
    if any(authorities[name]['route'] != expected for name, expected in MODES[mode].items()): raise ValueError('method identity mismatch')
    comparison = compare_authorities(list(authorities.values()))
    prepared = profiled_observation_paths(sparse['prepared_root'], subject, sparse_path)
    scalar = QuantPointDataset(prepared, device='cpu'); joint = JointAnchorDataset(prepared, scalar)
    resolved_policy = {name: verify_policy_weights(authority, scalar, joint) for name, authority in authorities.items()}
    images = {name: {p: nib.load(str(root / f'{p.upper()}_3D.nii.gz')) for p in PARAMETERS} for name, root in roots.items()}
    mappings = {name: {stack: root / 'evaluation' / 'mapping_central_no_psf' / f't1_t2_native_plane_{stack}.npz' for stack in STACKS if group_mapping[stack]} for name, root in roots.items()}
    for stack_map in mappings.values():
        for path in stack_map.values():
            if not path.is_file(): raise FileNotFoundError(path)
    caches = {s: _load_all_verified_native_cache_groups(r0_native_maps_root, sparse_path, s) for s in STACKS}; sax = caches['sax']
    sax_groups, sax_shape = np.asarray(sax['group_idx'], int), tuple(map(int, sax['t1_ms'].shape[1:]))
    archive, roi_provenance = _validate_roi_bundle_manifest(roi_bundle, subject)
    myocardium = _load_scmr_masks(roi_bundle, expected_groups=sax_groups, expected_shape=sax_shape)
    outer = load_outer_filled_masks(roi_bundle, expected_groups=sax_groups, expected_shape=sax_shape)
    blood = load_legacy_blood_pool_masks(roi_bundle, expected_groups=sax_groups, expected_shape=sax_shape, outer_filled=outer)
    volumes, mask_affine = stack_sax_masks_as_volume(sax, {'lv_outer_filled': outer, 'blood_pool': blood, **myocardium})
    roles = {}
    for stack in STACKS:
        retained, heldout = set(group_mapping[stack]), set(sparse['stacks'][stack]['heldout_original_group_ids'])
        groups = set(map(int, caches[stack]['group_idx']))
        if retained & heldout or retained | heldout != groups: raise ValueError(f'{stack}: sparse roles do not partition native groups')
        roles.update({(stack, group): 'observed' if group in retained else 'heldout' for group in groups})
    parts = ('all', 'observed', 'heldout', 'observed_sax', 'stack_sax', 'stack_2ch', 'stack_4ch', 'heldout_2ch', 'heldout_4ch')
    pool = {p: {part: {roi: {n: [] for n in ('reference', *roots)} for roi in ('common_native', *ALL_ROIS)} for part in parts} for p in PARAMETERS}
    output.parent.mkdir(parents=True, exist_ok=True); stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '.', dir=output.parent))
    try:
        rows, figures = [], []
        for stack in STACKS:
            cache = caches[stack]
            for i, group in enumerate(cache['group_idx']):
                group = int(group); role = roles[(stack, group)]; affine = native_ras_affine(cache['affine_lps_rc'][i]); shape = tuple(cache['t1_ms'][i].shape)
                rois = {name: project_sax_mask_volume(volume, mask_affine, affine, shape) for name, volume in volumes.items()}
                observed = {name: _observed_plane(mappings[name][stack], group_mapping[stack][group], shape) for name in roots} if role == 'observed' else {}
                for p in PARAMETERS:
                    ref = np.asarray(cache[p + '_ms'][i], float); pred, valid = {}, {}
                    for name in roots:
                        if role == 'observed': pred[name], valid[name] = observed[name][p]
                        else: pred[name], valid[name] = reproject_volume_to_native_plane(images[name][p].get_fdata(), images[name][p].affine, affine, shape)
                    common = common_support(ref, pred, valid, cache['source_support'][i]); masks = {'common_native': common, **{name: common & mask for name, mask in rois.items()}}
                    rows.append({'stack': stack, 'group': group, 'role': role, 'parameter': p, 'domains': {domain: statistics(ref, pred, mask, p) for domain, mask in masks.items()}})
                    for domain, mask in masks.items():
                        if not mask.any(): continue
                        destinations = ['all', role, f'stack_{stack}'] + (['observed_sax'] if role == 'observed' and stack == 'sax' else []) + ([f'heldout_{stack}'] if role == 'heldout' and stack in ('2ch', '4ch') else [])
                        for part in destinations:
                            pool[p][part][domain]['reference'].append(ref[mask])
                            for name in roots: pool[p][part][domain][name].append(pred[name][mask])
                    if common.any() and rois['lv_outer_filled'].any():
                        fig = render_plane(ref, pred, common, rois['lv_outer_filled'], p, stage / 'figures' / stack / f'group{group:03d}_{p}.png', title=f'{subject} {case} {stack} {group} {p.upper()} ({role})')
                        fig['path'] = str(Path(fig['path']).relative_to(stage)); figures.append(fig)
        aggregates, deltas = {}, []
        for p in PARAMETERS:
            aggregates[p] = {}
            for part, domains in pool[p].items():
                aggregates[p][part] = {}
                for domain, values in domains.items():
                    if values['reference']:
                        ref = np.concatenate(values['reference']); predictions = {n: np.concatenate(values[n]) for n in roots}
                        stat = statistics(ref, predictions, np.ones(ref.shape, bool), p)
                    else: stat = {'support_voxels': 0, 'methods': {n: None for n in roots}}
                    aggregates[p][part][domain] = stat
                    if mode in ('hgnll', 'hgnll_off'): deltas.append({'parameter': p, 'partition': part, 'domain': domain, 'support_voxels': stat['support_voxels'], 'comparisons': paired_deltas(stat, OFF_PAIRS if mode == 'hgnll_off' else PAIRS)})
        manifest = {'schema': 'v2_3_stack_policy_native/v1', 'mode': mode, 'subject': subject, 'case': case, 'primary_roi': PRIMARY_ROI, 'comparability_audit': comparison, 'verified_stack_policies': resolved_policy, 'sparse_manifest': _record(sparse_path), 'roi_authority': {'manifest': _record(roi_bundle / 'manifest.json'), 'archive': _record(archive), 'provenance': roi_provenance}, 'native_reference': {s: caches[s]['provenance'] for s in STACKS}, 'observed_mapping': {n: {s: _record(path) for s, path in paths.items()} for n, paths in mappings.items()}, 'mapping_provenance_limit': 'mapping files have no independent checkpoint digest', 'geometry_policy': {'observed': 'per-method final learned pose native mapping', 'heldout': 'central volume on original native plane'}, 'support': 'reference source support and finite reference and all method valid finite predictions', 'figures': figures}
        for filename, data in (('per_plane_metrics.json', {'planes': rows}), ('aggregate_metrics.json', {'primary_roi': PRIMARY_ROI, 'partitions': aggregates}), ('paired_deltas.json', {'positive_is_better': True, 'rows': deltas}), ('manifest.json', manifest)):
            (stage / filename).write_text(json.dumps(_clean(data), indent=2, allow_nan=False) + '\n')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True); raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=tuple(MODES), required=True)
    parser.add_argument('--method', action='append', required=True, help='method_id=training_root; repeat for each method in the selected mode')
    for name in ('r0-native-maps-root', 'sparse-manifest', 'roi-bundle', 'output'): parser.add_argument('--' + name, required=True)
    args = vars(parser.parse_args()); pairs = [item.split('=', 1) for item in args.pop('method')]
    if any(len(pair) != 2 for pair in pairs) or len(dict(pairs)) != len(pairs): parser.error('method arguments must be unique method_id=training_root pairs')
    print(run(roots=dict(pairs), **args))


if __name__ == '__main__': main()
