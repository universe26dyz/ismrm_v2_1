"""Zero-training audit of historical scalar-count stack weights and joint support."""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

from ismrm_v2.fit_first.contracts import sha256
from ismrm_v2.proposed.quantitative_point_dataset import QuantPointDataset
from ismrm_v2.proposed.q002.joint_dataset import JointAnchorDataset
from ismrm_v2.proposed.q002.contracts import profiled_observation_paths
from ismrm_v2.proposed.q002.stack_policy import resolve_stack_policy
from ismrm_v2.sampling.sparse_profile_manifest import validate_sparse_profile_manifest


def contribution_rows(scalar, joint, sparse, stack_indices):
    if len(set(stack_indices.values())) != len(stack_indices):
        raise ValueError("stack indices must be unique")
    policies = {}
    for name, config in (("off", {"enabled": False}), ("equal", {"enabled": True}),
                         ("slice", {"enabled": True, "policy": "slice_proportional_total"})):
        policies[name] = resolve_stack_policy(scalar, joint, config)
    scalar_counts = {idx: int((scalar.stack_idx == idx).sum().item()) for idx in stack_indices.values()}
    joint_counts = {idx: int((joint.stack_idx == idx).sum().item()) for idx in stack_indices.values()}
    if set(scalar_counts) != set(policies["equal"][0]):
        raise ValueError("manifest stack identities disagree with dataset")
    scalar_total, joint_total = sum(scalar_counts.values()), sum(joint_counts.values())
    rows = []
    for name in ("sax", "2ch", "4ch"):
        info = sparse["stacks"][name]
        retained = len(info["retained_original_group_ids"])
        idx = stack_indices.get(name)
        if (idx is None) != (retained == 0):
            raise ValueError(f"{name}: active stack and retained groups disagree")
        groups = [g for g in joint.qc["groups"].values() if g["stack_idx"] == idx] if idx is not None else []
        if len(groups) != retained:
            raise ValueError(f"{name}: retained group count disagrees with joint dataset")
        n_scalar, n_joint = scalar_counts.get(idx, 0), joint_counts.get(idx, 0)
        union = sum(g["union_valid_count"] for g in groups)
        row = {"stack_name": name, "stack_idx": idx, "retained_slice_count": retained,
               "heldout_slice_count": len(info["heldout_original_group_ids"]),
               "scalar_sample_count": n_scalar, "joint_anchor_count": n_joint,
               "scalar_fraction": n_scalar / scalar_total, "joint_anchor_fraction": n_joint / joint_total,
               "joint_to_scalar_ratio": n_joint / n_scalar if n_scalar else None,
               "joint_support_fraction_of_union": n_joint / union if union else None}
        for policy, prefix in (("equal", "historical_effective"), ("off", "off_policy"), ("slice", "slice_policy")):
            weights, qc = policies[policy]
            weight = weights[idx] if idx is not None else None
            row[("historical_equal_stack_weight" if policy == "equal" else prefix + "_weight")] = weight
            row[prefix + "_weighted_mass"] = n_joint * weight if weight is not None else 0.0
            row[prefix + "_weighted_fraction"] = qc["expected_weighted_fraction_by_stack"].get(idx, 0.0)
        # Keep the names emitted by the pre-existing audit for existing consumers.
        row["current_stack_weight"] = row["historical_equal_stack_weight"]
        row["current_expected_weighted_mass"] = row["historical_effective_weighted_mass"]
        row["current_expected_weighted_fraction"] = row["historical_effective_weighted_fraction"]
        rows.append(row)
    return rows


def run(prepared_root, sparse_manifest, output):
    import numpy as np
    sparse_path = Path(sparse_manifest)
    sparse = validate_sparse_profile_manifest(sparse_path, strict=False)
    paths = profiled_observation_paths(prepared_root, sparse['subject'], sparse_path)
    stack_indices = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            unique = np.unique(data['stack_idx'])
        if len(unique) != 1: raise ValueError(f'{path}: expected one stack index')
        name = path.parent.name
        with np.load(sparse['stacks'][name]['source_observation']['path'], allow_pickle=False) as source:
            source_indices = np.unique(source['stack_idx'])
        if len(source_indices) != 1 or int(source_indices[0]) != int(unique[0]):
            raise ValueError(f'{name}: source and prepared stack indices disagree')
        stack_indices[name] = int(unique[0])
    if len(set(stack_indices.values())) != len(stack_indices): raise ValueError('stack indices are not unique')
    scalar = QuantPointDataset(paths, device='cpu')
    joint = JointAnchorDataset(paths, scalar)
    rows = contribution_rows(scalar, joint, sparse, stack_indices)
    policy_qc = {name: resolve_stack_policy(scalar, joint, cfg)[1] for name, cfg in (("off", {"enabled": False}), ("equal", {"enabled": True}), ("slice", {"enabled": True, "policy": "slice_proportional_total"}))}
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))
    try:
        (stage / 'stack_contribution_audit.json').write_text(json.dumps({'stacks': rows, 'joint_dataset_qc': joint.qc}, indent=2) + '\n')
        with (stage / 'stack_contribution_audit.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        manifest = {'schema': 'v2_3_stack_balance_audit/v1', 'subject': sparse['subject'], 'case_id': sparse['case_id'], 'profile_id': sparse['profile_id'], 'sparse_manifest': {'path': str(sparse_path.resolve()), 'sha256': sha256(sparse_path)}, 'prepared_inputs': [{'path': str(p.resolve()), 'sha256': sha256(p)} for p in paths], 'stack_policy': 'current_inverse_scalar_frequency', 'weight_source': 'scalar_dataset.validate_balanced_samples', 'formula': 'weight_s = total_scalar_samples / (number_of_stacks * scalar_samples_s); expected mass = joint_anchor_count_s * weight_s', 'no_training': True, 'stack_policy_implementation': 'ismrm_v2.proposed.q002.stack_policy.resolve_stack_policy', 'policy_version': 'stack_policy_audit_v1', 'policy_provenance': policy_qc, 'retained_slice_count_by_stack': {r['stack_name']: r['retained_slice_count'] for r in rows}, 'scalar_count_by_stack': {r['stack_name']: r['scalar_sample_count'] for r in rows}, 'joint_anchor_count_by_stack': {r['stack_name']: r['joint_anchor_count'] for r in rows}}
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared-root', 'sparse-manifest', 'output'): parser.add_argument('--' + name, required=True)
    print(run(**vars(parser.parse_args())))

if __name__ == '__main__': main()
