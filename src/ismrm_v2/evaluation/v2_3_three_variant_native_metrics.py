"""Three-way downstream native-plane comparison. See --help for explicit inputs.

Observed planes use each method's final-pose bundle; held-out planes use each
central volume on the original native plane. Results are agreement with 2D
reference, not ground-truth accuracy.
"""
from __future__ import annotations
import argparse
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
from ismrm_v2.fit_first.contracts import sha256
from ismrm_v2.evaluation.evaluation import _metrics, PSNR_DATA_RANGES_MS, reproject_volume_to_native_plane
from ismrm_v2.evaluation.current_native_central_component_mask import load_validated_current_native_mask_plane
from ismrm_v2.evaluation.final5_partitioned_native_metrics import _bundle_plane
from ismrm_v2.evaluation.final5_publication_visualization import publication_group_roles
from ismrm_v2.evaluation.final_cyj_reporting import _load_all_verified_native_cache_groups
from ismrm_v2.evaluation.final_native_grid_metrics import native_ras_affine
from ismrm_v2.evaluation.aha.final5_adapter import resolve_subject_roi, require_exact_native_roi_mapping
from ismrm_v2.evaluation.aha.final5_myocardial_roi_metrics import PRIMARY_ROI, mapped_roi_plane
from ismrm_v2.evaluation.v2_3_boundary_mechanism_metrics import mechanism_statistics

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


def run(*, torch_mse_root, torch_hgnll10_root, torch_hgnll1_root, torch_mse_bundle, torch_hgnll10_bundle, torch_hgnll1_bundle, native_mask_root, r0_native_maps_root, sparse_manifest, roi_root, roi_mapping, output, standardization_manifest=None, cardiac_crop_root=None):
    import nibabel as nib
    from ismrm_v2.evaluation.v2_3_three_variant_visualization import render_plane, render_sax_reformats
    roots = dict(zip(METHODS, map(Path, (torch_mse_root, torch_hgnll10_root, torch_hgnll1_root))))
    bundles = dict(zip(METHODS, map(Path, (torch_mse_bundle, torch_hgnll10_bundle, torch_hgnll1_bundle))))
    sparse_path = Path(sparse_manifest); sparse = json.loads(sparse_path.read_text()); subject, case = sparse.get("subject"), sparse.get("case_id")
    if sparse.get("status") != "ELIGIBLE" or not subject or not case: raise ValueError("eligible subject/case sparse manifest required")
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    inputs = [*roots.values(), *bundles.values(), Path(native_mask_root), Path(r0_native_maps_root), Path(roi_root)]
    if any(output.is_relative_to(p.resolve()) for p in inputs): raise ValueError("output must be outside input roots")
    roi = resolve_subject_roi(roi_root, subject)
    if roi is None: raise ValueError("audited subject-specific ROI bundle required")
    mapping = require_exact_native_roi_mapping(roi_mapping, subject=subject, case=case, roi_root=roi, standardization_manifest_path=standardization_manifest)
    outer_path, inner_path = roi / "Segmentation_1-out-label.nii.gz", roi / "Segmentation_2-in-label.nii.gz"
    outer, inner = nib.load(str(outer_path)), nib.load(str(inner_path))
    if outer.shape != inner.shape or not np.allclose(outer.affine, inner.affine, atol=1e-6, rtol=0): raise ValueError("ROI geometries differ")
    entries = {int(e["native_group_idx"]): e for e in mapping["planes"]}
    if len(entries) != len(mapping["planes"]): raise ValueError("duplicate ROI group mapping")
    images, manifests, bundle_manifests, provenance = {}, {}, {}, {}
    for name in METHODS:
        mp, bp = roots[name] / "manifest.json", bundles[name] / "manifest.json"
        manifest, bundle = json.loads(mp.read_text()), json.loads(bp.read_text())
        if manifest.get("schema") != "phase4r_formal_proposed_export/v1" or (manifest.get("subject"), manifest.get("condition"), manifest.get("endpoint")) != (subject, case, "central"):
            raise ValueError(f"{name}: endpoint identity mismatch")
        if bundle.get("schema") not in ("direct_sparse_native_prediction_bundle/v1", "cyj11_r3_direct_final_pose_native_bundle/v1") or (bundle.get("subject"), bundle.get("case_id"), bundle.get("endpoint")) != (subject, case, "central"):
            raise ValueError(f"{name}: native bundle identity mismatch")
        if bundle.get("sparse_manifest", {}).get("sha256") != sha256(sparse_path): raise ValueError(f"{name}: bundle sparse manifest mismatch")
        if not bundle.get("checkpoint", {}).get("sha256") or bundle["checkpoint"]["sha256"] != manifest.get("source_checkpoint", {}).get("sha256"):
            raise ValueError(f"{name}: endpoint and native bundle checkpoint mismatch")
        for p in PARAMETERS:
            if sha256(roots[name] / f"{p.upper()}_3D.nii.gz") != manifest.get("source_volumes", {}).get(p, {}).get("sha256"):
                raise ValueError(f"{name}: endpoint {p} volume SHA256 mismatch")
        images[name] = {p: nib.load(str(roots[name] / f"{p.upper()}_3D.nii.gz")) for p in PARAMETERS}
        manifests[name], bundle_manifests[name] = manifest, bundle
        provenance[name] = {"endpoint_manifest": _record(mp), "native_bundle_manifest": _record(bp), "volumes": {p: _record(roots[name] / f"{p.upper()}_3D.nii.gz") for p in PARAMETERS}}
    caches = {s: _load_all_verified_native_cache_groups(r0_native_maps_root, sparse_path, s) for s in STACKS}
    roles = publication_group_roles(sparse, {s: [int(x) for x in caches[s]["group_idx"]] for s in STACKS})
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
                for p in PARAMETERS:
                    ref = np.asarray(cache[p + "_ms"][i], float); source = np.asarray(cache["source_support"][i], bool)
                    native, _ = load_validated_current_native_mask_plane(Path(native_mask_root), parameter=p, stack=stack, original_group_idx=group, shape_rc=ref.shape, source_support=source)
                    pred, valid = {}, {}
                    for name in METHODS:
                        if role == "observed":
                            record = next((x for x in bundle_manifests[name]["planes"] if x.get("stack") == stack and x.get("original_group_idx") == group), None)
                            if record is None or sha256(bundles[name] / record["path"]) != record.get("sha256"): raise ValueError(f"{name}: missing or altered native plane")
                            pred[name], valid[name] = _bundle_plane(bundle_manifests[name], bundles[name], parameter=p, stack=stack, group=group, method="proposed")
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
        manifest = {"schema": "v2_3_three_variant_native/v1", "subject": subject, "case": case, "primary_roi": PRIMARY_ROI, "support_equation": SUPPORT, "methods": provenance, "sparse_manifest": _record(sparse_path), "native_mask_manifest": _record(Path(native_mask_root) / "manifest.json"), "native_reference": {s: caches[s]["provenance"] for s in STACKS}, "roi_mapping": _record(roi_mapping), "standardization_manifest": _record(standard), "roi_outer": _record(outer_path), "roi_inner": _record(inner_path), "geometry_policy": {"observed": "per-method final learned pose native bundle", "heldout": "central endpoint reprojected to original native plane"}, "crop_policy": "legacy _crop_bounds on audited myocardium full + 3 pixels; optional explicit native cardiac masks for 2ch/4ch", "cardiac_crop_root": str(Path(cardiac_crop_root).resolve()) if cardiac_crop_root else None, "cardiac_crop_inputs": crop_sources, "colorbar_policy": "legacy Lipari/Navia map LUTs and ranges; signed coolwarm with symmetric legacy residual magnitude ranges", "figures": figures, "mechanism_ring": "three 3x3 dilations outside full myocardium intersected with three-way common native support", "roi_note": "primary ROI applies only to explicitly mapped SAX planes; empty partitions remain empty", "method_identity_note": "method labels are caller-declared; inspect endpoint training/backend provenance to verify deterministic Torch and variance configuration", "long_axis_note": "SAX centre-plane dense reformats use shared physical geometry; at least two SAX slices required. 2ch/4ch panels use matched native planes and optional explicit cardiac crop masks"}
        for filename, data in (("per_plane_metrics.json", {"planes": rows}), ("aggregate_metrics.json", {"primary_roi": PRIMARY_ROI, "partitions": aggregates}), ("mechanism_metrics.json", {"planes": mechanism}), ("manifest.json", manifest)):
            (stage / filename).write_text(json.dumps(_clean(data), indent=2, allow_nan=False) + "\n")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("torch-mse-root", "torch-hgnll10-root", "torch-hgnll1-root", "torch-mse-bundle", "torch-hgnll10-bundle", "torch-hgnll1-bundle", "native-mask-root", "r0-native-maps-root", "sparse-manifest", "roi-root", "roi-mapping", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--standardization-manifest")
    parser.add_argument("--cardiac-crop-root", help="optional native cardiac masks: STACK/groupNNN.npy for long-axis planes")
    print(run(**vars(parser.parse_args())))

if __name__ == "__main__": main()
