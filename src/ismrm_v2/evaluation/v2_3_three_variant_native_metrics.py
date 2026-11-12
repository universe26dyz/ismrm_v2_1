"""Three-way downstream native-plane comparison. See --help for explicit inputs.

Observed planes use each training root's final-pose mapping; held-out planes use each
central volume on the original native plane. Results are agreement with 2D
reference, not ground-truth accuracy.
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path
import numpy as np
from ismrm_v2.fit_first.contracts import sha256
from ismrm_v2.sampling.sparse_profile_manifest import validate_sparse_profile_manifest
from ismrm_v2.evaluation.evaluation import _metrics, PSNR_DATA_RANGES_MS, reproject_volume_to_native_plane
from ismrm_v2.evaluation.current_native_central_component_mask import load_validated_current_native_mask_plane
from ismrm_v2.evaluation.final_cyj_reporting import _load_all_verified_native_cache_groups
from ismrm_v2.evaluation.final_native_grid_metrics import native_ras_affine
from ismrm_v2.evaluation.aha.final5_adapter import resolve_subject_roi, require_exact_native_roi_mapping
from ismrm_v2.evaluation.aha.final5_myocardial_roi_metrics import PRIMARY_ROI, mapped_roi_plane
from ismrm_v2.evaluation.v2_3_boundary_mechanism_metrics import mechanism_statistics
from ismrm_v2.v2_3.hgnll import HGNLLConfig

METHODS = ("torch_mse", "torch_hgnll10", "torch_hgnll1")
STACKS = ("sax", "2ch", "4ch")
PARAMETERS = ("t1", "t2")
SUPPORT = "native mask AND finite(reference) AND all three method valid AND finite(all three predictions)"


def common_support(reference, predictions, valid, roi):
    if set(predictions) != set(METHODS) or set(valid) != set(METHODS):
        raise ValueError("exactly three named methods are required")
    shape = np.asarray(reference).shape
    arrays = [roi] + list(predictions.values()) + list(valid.values())
    if any(np.asarray(a).shape != shape for a in arrays):
        raise ValueError("all native arrays must share shape")
    support = np.asarray(roi, bool).copy() & np.isfinite(reference)
    for name in METHODS:
        support &= np.asarray(valid[name], bool) & np.isfinite(predictions[name])
    return support


def statistics(reference, predictions, support, parameter):
    count = int(np.count_nonzero(support))
    return {"support_voxels": count, "methods": {name: _metrics(reference, predictions[name], support, psnr_data_range=PSNR_DATA_RANGES_MS[parameter]) if count else None for name in METHODS}}


def _record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _clean(value):
    if isinstance(value, dict): return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    return value


EXPECTED_ROUTES = {"torch_mse": "B1_R0", "torch_hgnll10": "B1_R0_HGNLL", "torch_hgnll1": "B1_R0_HGNLL_ANCHORVAR1"}


def _resolve_run_log(root):
    local, parent = root / "run.log", root.parent / "run.log"
    if local.is_file() and parent.is_file():
        if sha256(local) != sha256(parent):
            raise ValueError(f"ambiguous deterministic backend logs: {local} and {parent}")
        return local
    return local if local.is_file() else parent


def _backend_provenance(path, required):
    if not path.is_file():
        if required: raise ValueError(f"deterministic Torch backend log missing: {path}")
        return {"status": "NOT_VERIFIABLE_FROM_ROOT"}
    content = path.read_text(errors="replace")
    checks = {"USE_TORCH": "True", "deterministic_algorithms": "True", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    findings = {}
    for key, value in checks.items():
        matches = re.findall(r"(?m)\b" + re.escape(key) + r"\s*=\s*([^\s,;]+)", content)
        findings[key] = bool(matches) and all(match == value for match in matches)
    hashes = re.findall(r"(?m)\bPYTHONHASHSEED\s*=\s*([^\s,;]+)", content)
    if hashes and any(x != "20260911" for x in hashes): findings["PYTHONHASHSEED"] = False
    elif hashes: findings["PYTHONHASHSEED"] = True
    passed = all(findings.values())
    if required and not passed: raise ValueError(f"deterministic Torch provenance incomplete or contradictory: {path}")
    return {"status": "PASS" if passed else "NOT_VERIFIABLE_FROM_ROOT", "backend": "vendored_torch_hashgrid" if passed else None, "checks": findings, "run_log": _record(path)}


def _derived_groups(sparse):
    result = {}
    for stack in STACKS:
        item = sparse.get("stacks", {}).get(stack)
        if not isinstance(item, dict): raise ValueError(f"{stack}: missing sparse stack")
        retained, heldout = item.get("retained_original_group_ids"), item.get("heldout_original_group_ids")
        if not isinstance(retained, list) or not isinstance(heldout, list) or any(type(x) is not int for x in retained + heldout) or len(set(retained + heldout)) != len(retained + heldout):
            raise ValueError(f"{stack}: invalid sparse group partition")
        if "training_stack_present" in item and item["training_stack_present"] is not bool(retained): raise ValueError(f"{stack}: active stack mismatch")
        records = item.get("derived_to_original_group_idx")
        if retained:
            if records is None and sparse.get("schema_version") == "phase4r_sparse_v1":
                pairs = list(enumerate(retained))
            elif isinstance(records, list):
                pairs = sorted((r["derived_group_idx"], r["original_group_idx"]) for r in records)
            else: raise ValueError(f"{stack}: missing derived group mapping")
            if [x for x, _ in pairs] != list(range(len(retained))) or set(y for _, y in pairs) != set(retained): raise ValueError(f"{stack}: derived mapping mismatch")
            result[stack] = {original: derived for derived, original in pairs}
        else: result[stack] = {}
    return result


def _training_authority(name, root, sparse, sparse_path, group_mapping, strict):
    paths = [root / "q002_route_manifest.json", root / "experiment_manifest.json"]
    first, second = (json.loads(p.read_text()) for p in paths)
    if first != second: raise ValueError(f"{name}: training manifests disagree")
    m = first; route = m.get("route", {}); controls = m.get("controls", {})
    if m.get("experiment_id") != EXPECTED_ROUTES[name] or route.get("experiment_id") != EXPECTED_ROUTES[name]: raise ValueError(f"{name}: experiment identity mismatch")
    if (m.get("subject_id"), m.get("case_id"), m.get("profile_id")) != (sparse.get("subject"), sparse.get("case_id"), sparse.get("profile_id")) or not sparse.get("profile_id"):
        raise ValueError(f"{name}: subject or sparse profile mismatch")
    if m.get("sparse_manifest", {}).get("sha256") != sha256(sparse_path): raise ValueError(f"{name}: sparse manifest mismatch")
    expected = {"stage_a_iterations": 2000, "stage_b_iterations": 4000, "anchor_batch_size": 640, "weights_per_anchor": 10, "training_psf_samples": 8, "seed": 20260911}
    if any(k in m and m[k] != v for k, v in expected.items()): raise ValueError(f"{name}: contradictory training controls")
    if any(controls.get(k) != v for k, v in expected.items()): raise ValueError(f"{name}: training controls mismatch")
    if route.get("sampling", {}).get("anchor_batch_size") != 640 or route.get("sampling", {}).get("weights_per_anchor") != 10 or route.get("sampling", {}).get("shared_training_psf_samples") != 8: raise ValueError(f"{name}: sampling contract mismatch")
    if ("stack_balance_enabled" in m and m["stack_balance_enabled"] is not True) or ("cross_view_reliability_enabled" in m and m["cross_view_reliability_enabled"] is not False): raise ValueError(f"{name}: contradictory weighting contract")
    if route.get("stack_balance", {}).get("enabled") is not True or route.get("cross_view_reliability", {}).get("enabled") is not False: raise ValueError(f"{name}: weighting contract mismatch")
    if name == "torch_mse":
        if route.get("heteroscedastic", {}).get("enabled", False) is not False or m.get("loss", "MSE").upper() not in ("MSE", "MSE_ONLY"): raise ValueError("MSE route must not enable HGNLL")
        identity = {"experiment_id": EXPECTED_ROUTES[name], "loss": "mse"}
    else:
        h = route.get("heteroscedastic", {})
        expected_mode = ("per_weight", 10, False) if name == "torch_hgnll10" else ("anchor_scalar_shared_across_weights", 1, True)
        mode = (h.get("variance_parameterization", "per_weight"), h.get("variance_output_dim", 10), h.get("variance_shared_across_weights", False))
        if h.get("enabled") is not True or m.get("loss") != "HGNLL" or mode != expected_mode or m.get("hgnll") != h: raise ValueError(f"{name}: HGNLL variance identity mismatch")
        HGNLLConfig.from_mapping(h, required=True)
        identity = {"experiment_id": EXPECTED_ROUTES[name], "loss": "hgnll", "variance_parameterization": mode[0], "variance_output_dim": mode[1], "variance_shared_across_weights": mode[2]}
    checkpoint = root / "model.pt"
    if m.get("q002_checkpoint_sha256") != sha256(checkpoint): raise ValueError(f"{name}: checkpoint SHA mismatch")
    prepared = m.get("prepared_inputs_sha256")
    if not isinstance(prepared, dict) or not prepared: raise ValueError(f"{name}: missing prepared hashes")
    prepared_by_stack = {}
    for path, digest in prepared.items():
        stack = Path(path).parent.name
        if stack not in STACKS or stack in prepared_by_stack or not isinstance(digest, str): raise ValueError(f"{name}: ambiguous prepared inputs")
        prepared_by_stack[stack] = digest
    active = {s for s in STACKS if group_mapping[s]}
    if set(prepared_by_stack) != active: raise ValueError(f"{name}: prepared active stacks mismatch")
    for stack in active:
        declared = sparse["stacks"][stack].get("derived_observation", sparse["stacks"][stack].get("source_observation", {}))
        if declared.get("sha256") != prepared_by_stack[stack]: raise ValueError(f"{name}: prepared input SHA mismatch for {stack}")
    shared = {"subject_id": m["subject_id"], "case_id": m["case_id"], "profile_id": m["profile_id"], "sparse_manifest_sha256": sha256(sparse_path), "prepared_inputs_sha256": prepared_by_stack, "b6_model_sha256": m.get("b6_model_sha256"), "decoder_sha256": m.get("approved_decoder_sha256"), "protocol_sha256": m.get("protocol_sha256"), **expected, "stack_balance_enabled": True, "cross_view_reliability_enabled": False}
    if any(not shared[k] for k in ("b6_model_sha256", "decoder_sha256", "protocol_sha256")): raise ValueError(f"{name}: missing scientific input hashes")
    dep = m.get("dependency_sha256", {})
    if any(dep.get(key) != shared[field] for key, field in (("B6_model", "b6_model_sha256"), ("frozen_decoder", "decoder_sha256"), ("protocol", "protocol_sha256"))): raise ValueError(f"{name}: inconsistent scientific input hashes")
    mappings = {stack: root / "evaluation" / "mapping_central_no_psf" / f"t1_t2_native_plane_{stack}.npz" for stack in active}
    for path in mappings.values():
        if not path.is_file(): raise FileNotFoundError(path)
    provenance = {"training_root": str(root.resolve()), "q002_route_manifest": _record(paths[0]), "experiment_manifest": _record(paths[1]), "checkpoint": _record(checkpoint), "t1_volume": _record(root / "T1_3D.nii.gz"), "t2_volume": _record(root / "T2_3D.nii.gz"), "observed_native_mapping": {k: _record(v) for k, v in mappings.items()}, "experiment_identity": identity, "backend_provenance": _backend_provenance(_resolve_run_log(root), strict)}
    return provenance, shared, mappings


def _observed_plane(path, group, shape):
    with np.load(path, allow_pickle=False) as data:
        required = ("group_idx", "weight_idx", "t1_ms", "t2_ms", "masks", "timing9_ms", "tr_ms", "vps")
        if any(k not in data for k in required): raise ValueError(f"incomplete native mapping: {path}")
        groups, weights = np.asarray(data["group_idx"]), np.asarray(data["weight_idx"])
        if groups.ndim != 1 or weights.shape != groups.shape: raise ValueError("invalid mapping group/weight shape")
        if np.asarray(data["timing9_ms"]).shape != (len(groups), 9) or any(np.asarray(data[key]).shape != (len(groups),) for key in ("tr_ms", "vps")): raise ValueError("invalid mapping acquisition metadata shape")
        rows = np.flatnonzero(groups == group)
        if len(rows) != 10 or sorted(weights[rows].tolist()) != list(range(10)): raise ValueError(f"{path}: group {group} must have weights 0..9")
        masks = np.asarray(data["masks"])[rows].astype(bool)
        if masks.shape != (10, *shape): raise ValueError("mapping plane shape mismatch")
        canonical = int(np.flatnonzero(weights[rows] == 0)[0]); result = {}
        for parameter in PARAMETERS:
            values = np.asarray(data[parameter + "_ms"])[rows]
            if values.shape != masks.shape: raise ValueError("mapping parameter shape mismatch")
            if np.any(masks & ~np.isfinite(values)): raise ValueError("mapping contains nonfinite values inside mask")
            common = masks.all(axis=0)
            if not common.any(): raise ValueError("mapping has no common 10-weight support")
            for row in range(10):
                for other in range(row):
                    overlap = masks[row] & masks[other]
                    if overlap.any() and not np.allclose(values[row][overlap], values[other][overlap], rtol=1e-5, atol=1e-3):
                        raise ValueError(f"{path}: inconsistent {parameter} maps across weights")
            result[parameter] = (values[canonical].copy(), common.copy())
        return result


def _audit_shared_contracts(shared_contracts):
    if any(shared_contracts[name] != shared_contracts[METHODS[0]] for name in METHODS[1:]):
        raise ValueError("three-method scientific comparability contract mismatch")
    return {"status": "PASS", "shared_contract": shared_contracts[METHODS[0]], "allowed_differences": ["experiment_id", "loss", "variance_parameterization", "checkpoint", "final_pose"]}


def run(*, torch_mse_root, torch_hgnll10_root, torch_hgnll1_root, native_mask_root, r0_native_maps_root, sparse_manifest, roi_root, roi_mapping, output, standardization_manifest=None, cardiac_crop_root=None, require_deterministic_torch_provenance=False):
    import nibabel as nib
    from ismrm_v2.evaluation.v2_3_three_variant_visualization import render_plane, render_sax_reformats
    roots = dict(zip(METHODS, map(Path, (torch_mse_root, torch_hgnll10_root, torch_hgnll1_root))))
    sparse_path = Path(sparse_manifest); sparse = validate_sparse_profile_manifest(sparse_path, strict=False); subject, case = sparse.get("subject"), sparse.get("case_id")
    if sparse.get("status") != "ELIGIBLE" or not subject or not case: raise ValueError("eligible subject/case sparse manifest required")
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    inputs = [*roots.values(), Path(native_mask_root), Path(r0_native_maps_root), Path(roi_root)]
    if any(output.is_relative_to(p.resolve()) for p in inputs): raise ValueError("output must be outside input roots")
    roi = resolve_subject_roi(roi_root, subject)
    if roi is None: raise ValueError("audited subject-specific ROI bundle required")
    mapping = require_exact_native_roi_mapping(roi_mapping, subject=subject, case=case, roi_root=roi, standardization_manifest_path=standardization_manifest)
    outer_path, inner_path = roi / "Segmentation_1-out-label.nii.gz", roi / "Segmentation_2-in-label.nii.gz"
    outer, inner = nib.load(str(outer_path)), nib.load(str(inner_path))
    if outer.shape != inner.shape or not np.allclose(outer.affine, inner.affine, atol=1e-6, rtol=0): raise ValueError("ROI geometries differ")
    entries = {int(e["native_group_idx"]): e for e in mapping["planes"]}
    if len(entries) != len(mapping["planes"]): raise ValueError("duplicate ROI group mapping")
    group_mapping = _derived_groups(sparse)
    images, provenance, mappings, shared_contracts = {}, {}, {}, {}
    for name in METHODS:
        provenance[name], shared_contracts[name], mappings[name] = _training_authority(name, roots[name], sparse, sparse_path, group_mapping, require_deterministic_torch_provenance)
        images[name] = {p: nib.load(str(roots[name] / f"{p.upper()}_3D.nii.gz")) for p in PARAMETERS}
    comparability = _audit_shared_contracts(shared_contracts)
    variance_configs = []
    for name in ("torch_hgnll10", "torch_hgnll1"):
        h = json.loads((roots[name] / "q002_route_manifest.json").read_text())["route"]["heteroscedastic"]
        normalized = HGNLLConfig.from_mapping(h, required=True).as_dict()
        for key in ("output", "variance_parameterization", "variance_output_dim", "variance_shared_across_weights"): normalized.pop(key)
        variance_configs.append(normalized)
    if variance_configs[0] != variance_configs[1]: raise ValueError("HGNLL variance policies differ beyond output parameterization")
    caches = {s: _load_all_verified_native_cache_groups(r0_native_maps_root, sparse_path, s) for s in STACKS}
    roles = {}
    for stack in STACKS:
        retained = set(group_mapping[stack]); heldout = set(sparse["stacks"][stack]["heldout_original_group_ids"])
        groups = set(map(int, caches[stack]["group_idx"]))
        if retained | heldout != groups: raise ValueError(f"{stack}: sparse roles do not partition native groups")
        roles.update({(stack, group): "observed" if group in retained else "heldout" for group in groups})
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        rows, mechanism, figures, roi_by_index, crop_sources = [], [], [], {}, {}
        pool = {p: {g: {domain: {n: [] for n in ("reference", *METHODS)} for domain in ("common_native", PRIMARY_ROI)} for g in ("all", "observed", "heldout")} for p in PARAMETERS}
        for stack in STACKS:
            cache = caches[stack]
            for i, group in enumerate(cache["group_idx"]):
                group = int(group); role = roles[(stack, group)]; affine = native_ras_affine(cache["affine_lps_rc"][i])
                entry = entries.get(group) if stack == "sax" else None
                observed = {name: _observed_plane(mappings[name][stack], group_mapping[stack][group], tuple(cache["t1_ms"][i].shape)) for name in METHODS} if role == "observed" else {}
                for p in PARAMETERS:
                    ref = np.asarray(cache[p + "_ms"][i], float); source = np.asarray(cache["source_support"][i], bool)
                    native, _ = load_validated_current_native_mask_plane(Path(native_mask_root), parameter=p, stack=stack, original_group_idx=group, shape_rc=ref.shape, source_support=source)
                    pred, valid = {}, {}
                    for name in METHODS:
                        if role == "observed":
                            pred[name], valid[name] = observed[name][p]
                        else:
                            image = images[name][p]
                            pred[name], valid[name] = reproject_volume_to_native_plane(image.get_fdata(), image.affine, affine, ref.shape)
                    common = common_support(ref, pred, valid, native["primary_native_mask"])
                    rois = mapped_roi_plane(outer, inner, entry=entry, expected_shape=ref.shape, spacing_rc_mm=tuple(map(float, cache["pixel_spacing_rc_mm"][i][:2])))["rois"] if entry else None
                    if rois is not None: roi_by_index[i] = rois["myocardium_full"]
                    masks = {"common_native": common}
                    if rois is not None: masks[PRIMARY_ROI] = common & rois[PRIMARY_ROI]
                    row = {"stack": stack, "group": group, "role": role, "parameter": p, "domains": {d: statistics(ref, pred, m, p) for d, m in masks.items()}}
                    rows.append(row)
                    for domain, mask in masks.items():
                        if mask.any():
                            for part in ("all", role):
                                pool[p][part][domain]["reference"].append(ref[mask])
                                for n in METHODS: pool[p][part][domain][n].append(pred[n][mask])
                    if rois is not None:
                        mechanism.append({"stack": stack, "group": group, "role": role, "parameter": p, **mechanism_statistics(ref, pred, common, rois[PRIMARY_ROI], rois["myocardium_full"])})
                    crop = rois["myocardium_full"] if rois is not None else None
                    if crop is None and cardiac_crop_root is not None:
                        path = Path(cardiac_crop_root) / stack / f"group{group:03d}.npy"
                        if path.is_file():
                            crop = np.load(path, allow_pickle=False)
                            crop_sources[f"{stack}/{group}"] = _record(path)
                    if crop is not None and np.any(crop) and common.any():
                        fig = render_plane(ref, pred, common, crop, p, stage / "figures" / stack / f"group{group:03d}_{p}.png", title=f"{subject} {case} {stack} {group} {p.upper()} ({role})")
                        fig["path"] = str(Path(fig["path"]).relative_to(stage)); fig.update({"stack": stack, "group": group, "parameter": p}); figures.append(fig)
        for figure in render_sax_reformats(caches["sax"], images, roi_by_index, stage / "figures" / "reformats"):
            figure["path"] = str(Path(figure["path"]).relative_to(stage)); figures.append(figure)
        aggregates = {}
        for p in PARAMETERS:
            aggregates[p] = {}
            for part, domains in pool[p].items():
                aggregates[p][part] = {}
                for domain, values in domains.items():
                    if values["reference"]:
                        ref = np.concatenate(values["reference"]); predictions = {n: np.concatenate(values[n]) for n in METHODS}
                        aggregates[p][part][domain] = statistics(ref, predictions, np.ones(ref.shape, bool), p)
                    else: aggregates[p][part][domain] = {"support_voxels": 0, "methods": {n: None for n in METHODS}}
        standard = Path(standardization_manifest) if standardization_manifest else Path(mapping["standardization_manifest"]["path"])
        manifest = {"schema": "v2_3_three_variant_native/v2", "authority_mode": "q002_training_roots", "comparability_audit": comparability, "subject": subject, "case": case, "primary_roi": PRIMARY_ROI, "support_equation": SUPPORT, "methods": provenance, "sparse_manifest": _record(sparse_path), "native_mask_manifest": _record(Path(native_mask_root) / "manifest.json"), "native_reference": {s: caches[s]["provenance"] for s in STACKS}, "roi_mapping": _record(roi_mapping), "standardization_manifest": _record(standard), "roi_outer": _record(outer_path), "roi_inner": _record(inner_path), "observed_mapping_policy": "per-group ten weight maps audited on overlapping mask pixels; valid support is intersection of ten masks; canonical weight 0", "mapping_provenance_limit": "training-root mapping files have no independent checkpoint digest; files are identified by path and SHA256 but their creation from the checkpoint cannot be independently verified", "geometry_policy": {"observed": "per-method final learned pose mapping_central_no_psf", "heldout": "central endpoint reprojected to original native plane"}, "crop_policy": "legacy _crop_bounds on audited myocardium full + 3 pixels; optional explicit native cardiac masks for 2ch/4ch", "cardiac_crop_root": str(Path(cardiac_crop_root).resolve()) if cardiac_crop_root else None, "cardiac_crop_inputs": crop_sources, "colorbar_policy": "legacy Lipari/Navia map LUTs and ranges; signed coolwarm with symmetric legacy residual magnitude ranges", "figures": figures, "mechanism_ring": "three 3x3 dilations outside full myocardium intersected with three-way common native support", "roi_note": "primary ROI applies only to explicitly mapped SAX planes; held-out LAX myocardium unavailable from current ROI authority; empty partitions remain empty",  "long_axis_note": "SAX centre-plane dense reformats use shared physical geometry; at least two SAX slices required. 2ch/4ch panels use matched native planes and optional explicit cardiac crop masks"}
        for filename, data in (("per_plane_metrics.json", {"planes": rows}), ("aggregate_metrics.json", {"primary_roi": PRIMARY_ROI, "partitions": aggregates}), ("mechanism_metrics.json", {"planes": mechanism}), ("manifest.json", manifest)):
            (stage / filename).write_text(json.dumps(_clean(data), indent=2, allow_nan=False) + "\n")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("torch-mse-root", "torch-hgnll10-root", "torch-hgnll1-root", "native-mask-root", "r0-native-maps-root", "sparse-manifest", "roi-root", "roi-mapping", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--standardization-manifest")
    parser.add_argument("--cardiac-crop-root", help="optional native cardiac masks: STACK/groupNNN.npy for long-axis planes")
    parser.add_argument("--require-deterministic-torch-provenance", action="store_true")
    print(run(**vars(parser.parse_args())))

if __name__ == "__main__": main()
