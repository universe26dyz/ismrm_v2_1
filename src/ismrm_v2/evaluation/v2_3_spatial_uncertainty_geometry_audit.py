"""Zero-training native-plane pose, uncertainty and cross-view geometry audit."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from scipy.stats import pearsonr, spearmanr

from ismrm_v2.evaluation.direct_native_endpoint_audit import _load_model, _sparse_contract, direct_native_world_grid
from ismrm_v2.evaluation.evaluation import _metrics, _world_sample, PSNR_DATA_RANGES_MS
from ismrm_v2.evaluation.final_cyj_reporting import _load_all_verified_native_cache_groups
from ismrm_v2.evaluation.final_native_grid_metrics import native_ras_affine
from ismrm_v2.evaluation.aha.cyj11_myocardial_roi_metrics import _load_scmr_masks
from ismrm_v2.evaluation.aha.paired_final_pose_lv_metrics import load_outer_filled_masks
from ismrm_v2.evaluation.aha.common_native_all_stack_roi_metrics import load_legacy_blood_pool_masks, stack_sax_masks_as_volume, project_sax_mask_volume
from ismrm_v2.evaluation.v2_3_h10_detach_native_metrics import _derived_groups, _record, _validate_roi_bundle_manifest
from ismrm_v2.evaluation.v2_3_stack_policy_authority import read_authority, compare_authorities, verify_policy_weights
from ismrm_v2.proposed.q002.contracts import profiled_observation_paths
from ismrm_v2.proposed.q002.joint_dataset import JointAnchorDataset
from ismrm_v2.proposed.quantitative_point_dataset import QuantPointDataset
from ismrm_v2.sampling.sparse_profile_manifest import validate_sparse_profile_manifest
from ismrm_v2.v2_2.preflight.cross_view_reliability_audit import match_lax_to_sax
from nesvor.transform import RigidTransform, ax_transform_points

STACKS = ('sax', '2ch', '4ch')
METHODS = ('torch_mse', 'torch_hgnll10', 'torch_hgnll10_detach')
THRESHOLDS = (1, 2, 3, 5)


def pose_difference(first, second):
    a, b = np.asarray(first, float), np.asarray(second, float)
    if a.shape != (3, 4) or b.shape != (3, 4) or not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('rigid poses must be finite 3x4 matrices')
    ah, bh = np.eye(4), np.eye(4); ah[:3] = a; bh[:3] = b
    for matrix in (ah, bh):
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4): raise ValueError('pose rotation is not rigid')
    relative = bh @ np.linalg.inv(ah)
    return {'rotation_deg': float(np.degrees(np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1)))), 'translation_mm': float(np.linalg.norm(relative[:3, 3])), 'center_displacement_mm': float(np.linalg.norm(b[:3, 3] - a[:3, 3]))}


def native_world_grid(affine, shape):
    row, col = np.indices(shape, dtype=float)
    world = np.asarray(affine, float) @ np.vstack((row.ravel(), col.ravel(), np.zeros(row.size), np.ones(row.size)))
    return world[:3].T.reshape(*shape, 3)


def sample_volume(image, physical_world):
    world = np.asarray(physical_world, float)
    if world.ndim != 3 or world.shape[-1] != 3 or not np.isfinite(world).all(): raise ValueError('world grid must be finite [H,W,3]')
    homogeneous = np.concatenate((np.moveaxis(world, -1, 0), np.ones((1, *world.shape[:2]))), axis=0)
    return _world_sample(image.get_fdata(), image.affine, homogeneous, order=1)


def sigma_maps(model, space, physical_world):
    world = np.asarray(physical_world, float)
    if world.ndim != 3 or world.shape[-1] != 3 or not np.isfinite(world).all(): raise ValueError('sigma world grid must be finite [H,W,3]')
    if getattr(model, 'joint_variance_head', None) is None: raise ValueError('model has no trained joint variance head')
    device = next(model.parameters()).device; chunks = []
    points = torch.as_tensor(world.reshape(-1, 3), dtype=torch.float32, device=device)
    with torch.no_grad():
        for chunk in points.split(65536):
            train = space.physical_ras_to_train(chunk)
            latent = model.inr.forward_with_latent(train, include_latent=True)['latent']
            lv = model.joint_variance_head.bounded_log_variance(latent)
            if lv.ndim != 2 or lv.shape != (len(chunk), 10) or not torch.isfinite(lv).all(): raise ValueError('variance head must emit finite per-weight log variance')
            chunks.append(torch.exp(.5 * lv).cpu().numpy())
    sigma = np.concatenate(chunks).reshape(*world.shape[:2], 10)
    return {'sigma_mean_across_weights': sigma.mean(axis=-1), 'sigma_p90_across_weights': np.percentile(sigma, 90, axis=-1), 'sigma_max_across_weights': sigma.max(axis=-1)}


def coverage_maps(world, supports, target_stack):
    points = np.asarray(world, float)
    if points.ndim != 3 or points.shape[-1] != 3 or not np.isfinite(points).all() or target_stack not in STACKS: raise ValueError('invalid target world coordinates or stack')
    distances = {}; shape = points.shape[:2]
    for stack in STACKS:
        reference = np.asarray(supports.get(stack, np.empty((0, 3))), float)
        if reference.ndim != 2 or reference.shape[1] != 3 or not np.isfinite(reference).all(): raise ValueError('invalid view support coordinates')
        distances[stack] = match_lax_to_sax(reference, points.reshape(-1, 3))[1].reshape(shape) if len(reference) else None
    other = [distances[s] for s in STACKS if s != target_stack and distances[s] is not None]
    if not other: raise ValueError('at least one other view is required to audit coverage')
    result = {f'distance_to_nearest_{s}_mm': distances[s] for s in STACKS}
    result['nearest_other_view_distance_mm'] = np.minimum.reduce(other)
    for threshold in THRESHOLDS: result[f'other_view_count_within_{threshold}mm'] = sum((d <= threshold).astype(np.int8) for d in other)
    return result


def correlations(x, y):
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('correlation input must be matching finite arrays')
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0: return {'n': len(x), 'pearson': None, 'spearman': None, 'status': 'insufficient_or_constant'}
    return {'n': len(x), 'pearson': float(pearsonr(x, y).statistic), 'spearman': float(spearmanr(x, y).statistic), 'status': 'complete'}


def error_change(reference, h10, detach):
    ref, h10, detach = (np.asarray(x, float) for x in (reference, h10, detach))
    if ref.shape != h10.shape or ref.shape != detach.shape or not np.isfinite(ref).all() or not np.isfinite(h10).all() or not np.isfinite(detach).all(): raise ValueError('error change requires matching finite arrays')
    return np.abs(detach - ref) - np.abs(h10 - ref)


def _write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _summary(values):
    x = np.asarray(values, float)
    if not len(x) or not np.isfinite(x).all(): raise ValueError('summary requires finite nonempty data')
    return {'mean': float(x.mean()), 'median': float(np.median(x)), 'p90': float(np.percentile(x, 90)), 'p95': float(np.percentile(x, 95)), 'max': float(x.max())}


def _plot(output, reference, h10, detach, sigma, coverage, delta, support, parameter, title):
    import matplotlib.pyplot as plt
    from ismrm_v2.evaluation.scmr.quality_control import DISPLAY_RANGES_MS, RESIDUAL_RANGES_MS
    key = parameter.upper(); lim = RESIDUAL_RANGES_MS[key][1]; vmax = DISPLAY_RANGES_MS[key][1]
    panels = [('Native reference', reference, 'viridis', 0, vmax), ('H10', h10, 'viridis', 0, vmax), ('Detach', detach, 'viridis', 0, vmax), ('H10 residual', h10-reference, 'coolwarm', -lim, lim), ('Detach residual', detach-reference, 'coolwarm', -lim, lim), ('H10 pointwise sigma mean', sigma, 'magma', 0, 1), ('Nearest other view (mm)', coverage, 'viridis', 0, 5), ('Δ absolute error', delta, 'coolwarm', -lim, lim)]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True); fig.suptitle(title)
    for ax, (label, values, cmap, vmin, upper) in zip(axes.flat, panels):
        im = ax.imshow(np.ma.masked_where(~support, values), origin='lower', interpolation='nearest', cmap=cmap, vmin=vmin, vmax=upper); ax.set_title(label); ax.axis('off'); fig.colorbar(im, ax=ax, shrink=.7)
    output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=150, facecolor='white'); plt.close(fig)


def run(*, mse_root, h10_root, detach_root, prepared_root, r0_native_maps_root, sparse_manifest, roi_bundle, output, device='cpu'):
    roots = dict(zip(METHODS, map(Path, (mse_root, h10_root, detach_root))))
    sparse_path = Path(sparse_manifest); sparse = validate_sparse_profile_manifest(sparse_path, strict=False)
    subject, case = sparse['subject'], sparse['case_id']; mapping = _derived_groups(sparse)
    output, roi_bundle = Path(output).resolve(), Path(roi_bundle)
    if output.exists(): raise FileExistsError(output)
    if any(output.is_relative_to(p.resolve()) for p in [*roots.values(), Path(prepared_root), Path(r0_native_maps_root), roi_bundle]): raise ValueError('output must be outside input roots')
    authority = {name: read_authority(roots[name], sparse, sparse_path) for name in METHODS}
    mse, h10, detach = (authority[name] for name in METHODS)
    if (mse['kind'], h10['kind'], detach['kind']) != ('mse', 'hgnll', 'hgnll') or h10['detach'] is not False or detach['detach'] is not True:
        raise ValueError('spatial audit requires MSE, H10 and H10-detach identities')
    if len({mse['policy'], h10['policy'], detach['policy']}) != 1:
        raise ValueError('spatial audit methods must share stack policy')
    if any(a['scientific'] != mse['scientific'] for a in (h10, detach)):
        raise ValueError('spatial audit scientific settings do not match')
    detach_comparison = compare_authorities([h10, detach])
    comparison = {'status': 'PASS', 'shared_stack_policy': mse['policy'], 'allowed_differences': ['MSE vs HGNLL', 'variance_detach_shared_latent']}
    prepared_paths = profiled_observation_paths(prepared_root, subject, sparse_path)
    scalar = QuantPointDataset(prepared_paths, device='cpu'); joint = JointAnchorDataset(prepared_paths, scalar)
    verified_policies = {name: verify_policy_weights(item, scalar, joint) for name, item in authority.items()}
    sparse_contract = _sparse_contract(sparse_path)
    models, spaces, images, matrices = {}, {}, {}, {}
    offsets = {}; cursor = 0
    for path in prepared_paths:
        stack = path.parent.name
        with np.load(path, allow_pickle=False) as data:
            groups = np.unique(data['group_idx']); unique_stack = np.unique(data['stack_idx'])
        if len(unique_stack) != 1 or len(groups) != len(mapping[stack]): raise ValueError('prepared stack identity or group count mismatch')
        offsets[stack] = (cursor, int(unique_stack[0])); cursor += len(groups)
    if len({idx for _, idx in offsets.values()}) != len(offsets): raise ValueError('stack indices are not unique')
    from nesvor.inr import models as nesvor_models
    previous_backend = nesvor_models.USE_TORCH
    nesvor_models.USE_TORCH = True
    try:
        for name, root in roots.items():
            models[name], spaces[name], _ = _load_model(root / 'model.pt', prepared_paths, device, sparse_contract=sparse_contract)
            model, space = models[name], spaces[name]
            axisangle = space.axisangle_train_to_physical(model.rigid_psf.axisangle.detach().cpu())
            matrix = RigidTransform(axisangle, trans_first=True).matrix(trans_first=True).detach().cpu().numpy()
            declared = np.asarray(json.loads((root / 'final_rigid_poses.json').read_text())['matrix_physical'], float)
            if matrix.shape != (cursor, 3, 4) or declared.shape != matrix.shape or not np.allclose(matrix, declared, rtol=0, atol=1e-4): raise ValueError(f'{name}: exported final poses disagree with checkpoint')
            matrices[name] = matrix
            images[name] = {p: nib.load(str(root / f'{p.upper()}_3D.nii.gz')) for p in ('t1', 't2')}
    finally:
        nesvor_models.USE_TORCH = previous_backend
    if not torch.allclose(spaces[METHODS[1]].group_axisangle_dicom_physical, spaces[METHODS[2]].group_axisangle_dicom_physical, rtol=0, atol=1e-5): raise ValueError('original pose provenance disagrees between H10 and detach')
    original = spaces[METHODS[1]].group_axisangle_dicom_physical
    physical = ax_transform_points(original[joint.group_idx], joint.xyz, trans_first=True).detach().cpu().numpy()
    supports = {stack: physical[joint.stack_idx.numpy() == idx] for stack, (_, idx) in offsets.items()}
    if not np.isfinite(physical).all(): raise ValueError('joint support contains nonfinite physical coordinates')
    pose_rows = []
    for stack, (offset, _) in offsets.items():
        for original_group, derived in sorted(mapping[stack].items()):
            group = offset + derived
            for label, a, b in (('mse_vs_h10', METHODS[0], METHODS[1]), ('h10_vs_detach', METHODS[1], METHODS[2]), ('mse_vs_detach', METHODS[0], METHODS[2])):
                pose_rows.append({'subject': subject, 'stack': stack, 'group': original_group, 'comparison': label, **pose_difference(matrices[a][group], matrices[b][group])})
    pose_summary = []
    for stack in STACKS:
        for label in ('mse_vs_h10', 'h10_vs_detach', 'mse_vs_detach'):
            selected = [r for r in pose_rows if r['stack'] == stack and r['comparison'] == label]
            if selected:
                for metric in ('rotation_deg', 'translation_mm', 'center_displacement_mm'):
                    pose_summary.append({'subject': subject, 'stack': stack, 'comparison': label, 'metric': metric, 'n': len(selected), **_summary([r[metric] for r in selected])})
    caches = {s: _load_all_verified_native_cache_groups(r0_native_maps_root, sparse_path, s) for s in STACKS}; sax = caches['sax']
    sax_groups, sax_shape = np.asarray(sax['group_idx'], int), tuple(map(int, sax['t1_ms'].shape[1:]))
    archive, roi_provenance = _validate_roi_bundle_manifest(roi_bundle, subject)
    myocardium = _load_scmr_masks(roi_bundle, expected_groups=sax_groups, expected_shape=sax_shape)
    outer = load_outer_filled_masks(roi_bundle, expected_groups=sax_groups, expected_shape=sax_shape)
    blood = load_legacy_blood_pool_masks(roi_bundle, expected_groups=sax_groups, expected_shape=sax_shape, outer_filled=outer)
    volumes, mask_affine = stack_sax_masks_as_volume(sax, {'lv_outer_filled': outer, 'blood_pool': blood, **myocardium})
    for stack in STACKS:
        retained, held = set(mapping[stack]), set(sparse['stacks'][stack]['heldout_original_group_ids'])
        if retained & held or retained | held != set(map(int, caches[stack]['group_idx'])): raise ValueError('sparse roles do not partition native groups')
    output.parent.mkdir(parents=True, exist_ok=True); stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '.', dir=output.parent))
    try:
        pose_metrics, sigma_rows, coverage_rows, corr_rows, change_rows, figures, map_records = [], [], [], [], [], [], []
        for stack in STACKS:
            cache = caches[stack]
            for index, raw_group in enumerate(cache['group_idx']):
                group = int(raw_group); role = 'observed' if group in mapping[stack] else 'heldout'
                affine = native_ras_affine(cache['affine_lps_rc'][index]); shape = tuple(cache['t1_ms'][index].shape)
                native_world = native_world_grid(affine, shape)
                roi = {name: project_sax_mask_volume(volume, mask_affine, affine, shape) for name, volume in volumes.items()}
                roi['peri_myocardium_ring'] = binary_dilation(roi['myocardium_full'], structure=np.ones((3, 3), bool), iterations=3) & ~roi['myocardium_full']
                world = {}
                for name in METHODS:
                    if role == 'observed':
                        global_group = offsets[stack][0] + mapping[stack][group]
                        train_world = direct_native_world_grid(models[name], spaces[name], group_idx=global_group, shape=shape, spacing_rc_mm=cache['pixel_spacing_rc_mm'][index], torch_module=torch)
                        world[name] = spaces[name].train_ras_to_physical(train_world).detach().cpu().numpy()
                    else: world[name] = native_world
                sigma = {name: sigma_maps(models[name], spaces[name], world[name]) for name in METHODS[1:]}
                coverage = {name: coverage_maps(world[name], supports, stack) for name in METHODS[1:]}
                source = np.asarray(cache['source_support'][index], bool)
                arrays = {'reference_world_ras_mm': native_world}
                for name in METHODS[1:]:
                    for key, val in sigma[name].items(): arrays[f'{name}_{key}'] = val
                    for key, val in coverage[name].items():
                        if val is not None: arrays[f'{name}_{key}'] = val
                for p in ('t1', 't2'):
                    ref = np.asarray(cache[p + '_ms'][index], float); predictions, valid, native_predictions, native_valid = {}, {}, {}, {}
                    for name in METHODS:
                        predictions[name], valid[name] = sample_volume(images[name][p], world[name])
                        native_predictions[name], native_valid[name] = sample_volume(images[name][p], native_world)
                        if np.any(source & valid[name] & ~np.isfinite(predictions[name])) or np.any(source & native_valid[name] & ~np.isfinite(native_predictions[name])):
                            raise ValueError(f'{name}: nonfinite in-volume predictions on source support')
                        arrays[f'{name}_{p}_learned_pose'] = predictions[name]; arrays[f'{name}_{p}_native_pose'] = native_predictions[name]
                    for name in METHODS:
                        mask0 = source & np.isfinite(ref) & valid[name] & native_valid[name] & np.isfinite(predictions[name]) & np.isfinite(native_predictions[name])
                        for roi_name in ('myocardium_core_1px', 'blood_pool', 'lv_outer_filled'):
                            selected = mask0 & roi[roi_name]
                            for pose_label, prediction in (('learned', predictions[name]), ('native', native_predictions[name])):
                                metrics = _metrics(ref, prediction, selected, psnr_data_range=PSNR_DATA_RANGES_MS[p]) if selected.any() else None
                                pose_metrics.append({'subject': subject, 'stack': stack, 'group': group, 'role': role, 'parameter': p, 'method': name, 'roi': roi_name, 'pose': pose_label, 'support_voxels': int(selected.sum()), 'rmse': metrics['rmse'] if metrics else None, 'mae': metrics['mae'] if metrics else None, 'bias': metrics['bias'] if metrics else None})
                    common = source & np.isfinite(ref)
                    for name in METHODS[1:]: common &= valid[name] & np.isfinite(predictions[name])
                    change = np.full(shape, np.nan); change[common] = error_change(ref[common], predictions[METHODS[1]][common], predictions[METHODS[2]][common]); arrays[f'{p}_detach_minus_h10_absolute_error'] = change
                    for name in METHODS[1:]:
                        sig = sigma[name]['sigma_mean_across_weights']; cov = coverage[name]; absolute = np.abs(predictions[name] - ref)
                        for roi_name in ('myocardium_core_1px', 'myocardium_full', 'peri_myocardium_ring', 'blood_pool'):
                            mask = common & roi[roi_name] & np.isfinite(sig) & np.isfinite(cov['nearest_other_view_distance_mm'])
                            base = {'subject': subject, 'stack': stack, 'group': group, 'role': role, 'parameter': p, 'method': name, 'roi': roi_name, 'n': int(mask.sum())}
                            if not mask.any(): continue
                            for key, values in sigma[name].items(): sigma_rows.append({**base, 'metric': key, **_summary(values[mask])})
                            for key, values in cov.items():
                                if values is not None: coverage_rows.append({**base, 'metric': key, **_summary(values[mask])})
                            pairs = {'absolute_residual_vs_sigma': (absolute, sig), 'absolute_residual_vs_nearest_view_distance': (absolute, cov['nearest_other_view_distance_mm']), 'sigma_vs_nearest_view_distance': (sig, cov['nearest_other_view_distance_mm'])}
                            for threshold in THRESHOLDS: pairs[f'absolute_residual_vs_other_view_count_{threshold}mm'] = (absolute, cov[f'other_view_count_within_{threshold}mm'])
                            for metric, (x, y) in pairs.items(): corr_rows.append({**base, 'relationship': metric, **correlations(x[mask], y[mask])})
                            if name == METHODS[1]:
                                pose_record = next((r for r in pose_rows if r['stack'] == stack and r['group'] == group and r['comparison'] == 'h10_vs_detach'), None)
                                for metric, values in (('h10_sigma', sig), ('nearest_other_view_distance_mm', cov['nearest_other_view_distance_mm']), ('other_view_count_within_3mm', cov['other_view_count_within_3mm'])):
                                    change_rows.append({**base, 'pose_translation_mm': pose_record['translation_mm'] if pose_record else None, 'pose_rotation_deg': pose_record['rotation_deg'] if pose_record else None, 'mean_delta_abs_error': float(change[mask].mean()), 'relationship': metric, **correlations(change[mask], values[mask])})
                    if common.any():
                        target = stage / 'figures' / stack / f'group{group:03d}_{p}.png'
                        _plot(target, ref, predictions[METHODS[1]], predictions[METHODS[2]], sigma[METHODS[1]]['sigma_mean_across_weights'], coverage[METHODS[1]]['nearest_other_view_distance_mm'], change, common, p, f'{subject} {case} {stack} {group} {role} {p.upper()}')
                        figures.append(str(target.relative_to(stage)))
                target = stage / 'maps' / stack / f'group{group:03d}.npz'; target.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(target, **arrays); map_records.append(_record(target))
        _write_csv(stage / 'pose_per_group.csv', pose_rows, ['subject', 'stack', 'group', 'comparison', 'rotation_deg', 'translation_mm', 'center_displacement_mm'])
        summary_fields = ['subject', 'stack', 'comparison', 'metric', 'n', 'mean', 'median', 'p90', 'p95', 'max']
        _write_csv(stage / 'pose_summary.csv', pose_summary, summary_fields)
        _write_csv(stage / 'learned_vs_native_pose_metrics.csv', pose_metrics, ['subject', 'stack', 'group', 'role', 'parameter', 'method', 'roi', 'pose', 'support_voxels', 'rmse', 'mae', 'bias'])
        base_fields = ['subject', 'stack', 'group', 'role', 'parameter', 'method', 'roi', 'n']
        for file, rows in (('spatial_sigma_summary.csv', sigma_rows), ('coverage_summary.csv', coverage_rows)):
            _write_csv(stage / file, rows, base_fields + ['metric', 'mean', 'median', 'p90', 'p95', 'max'])
        _write_csv(stage / 'residual_sigma_coverage_correlations.csv', corr_rows, base_fields + ['relationship', 'pearson', 'spearman', 'status'])
        _write_csv(stage / 'h10_vs_detach_change_summary.csv', change_rows, base_fields + ['pose_translation_mm', 'pose_rotation_deg', 'mean_delta_abs_error', 'relationship', 'pearson', 'spearman', 'status'])
        pose_change_rows = []
        keys = sorted({(r['stack'], r['role'], r['parameter'], r['roi']) for r in change_rows if r['relationship'] == 'h10_sigma' and r['pose_translation_mm'] is not None})
        for stack, role, parameter, region in keys:
            selected = [r for r in change_rows if (r['stack'], r['role'], r['parameter'], r['roi'], r['relationship']) == (stack, role, parameter, region, 'h10_sigma') and r['pose_translation_mm'] is not None]
            for metric in ('pose_translation_mm', 'pose_rotation_deg'):
                pose_change_rows.append({'subject': subject, 'stack': stack, 'role': role, 'parameter': parameter, 'roi': region, 'relationship': metric, **correlations([r['mean_delta_abs_error'] for r in selected], [r[metric] for r in selected])})
        _write_csv(stage / 'error_change_pose_correlations.csv', pose_change_rows, ['subject', 'stack', 'role', 'parameter', 'roi', 'relationship', 'n', 'pearson', 'spearman', 'status'])
        for item in map_records: item['path'] = str(Path(item['path']).relative_to(stage))
        manifest = {'schema': 'v2_3_spatial_uncertainty_geometry_audit/v1', 'subject': subject, 'case_id': case, 'profile_id': sparse['profile_id'], 'comparability': comparison, 'verified_stack_policies': verified_policies, 'h10_detach_comparability': detach_comparison, 'sparse_manifest': _record(sparse_path), 'prepared_inputs': [_record(p) for p in prepared_paths], 'final_pose_files': {name: _record(root / 'final_rigid_poses.json') for name, root in roots.items()}, 'models': authority, 'roi_archive': _record(archive), 'roi_provenance': roi_provenance, 'pose_policy': 'same final central quantitative NIfTI sampled at final learned grid and native original plane; heldout uses original pose', 'pose_translation_semantics': 'relative rigid transform translation; center displacement also reported', 'sigma_policy': 'pointwise trained bounded variance head, square root, no PSF aggregation; checkpoint rebuilt with vendored Torch backend', 'coverage_policy': 'nearest retained joint anchor per view in original DICOM physical RAS geometry; supporting-view counts exclude target view', 'correlations': 'within subject/parameter/stack/group/role/method/ROI; diagnostic association, not causal proof', 'change_sign': 'abs(detach-reference) - abs(h10-reference); negative improves', 'edge_abnormality_question': 'localized myocardial boundary phenomenon', 'sax_fidelity_question': 'agreement between observed SAX and native 2D SAX reference', 'figures': figures, 'maps': map_records}
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n'); stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True); raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mse-root', 'h10-root', 'detach-root', 'prepared-root', 'r0-native-maps-root', 'sparse-manifest', 'roi-bundle', 'output'): parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cpu')
    print(run(**vars(parser.parse_args())))


if __name__ == '__main__': main()
