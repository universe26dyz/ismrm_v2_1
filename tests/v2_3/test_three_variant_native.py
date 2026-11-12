import numpy as np
import pytest
from ismrm_v2.evaluation.v2_3_three_variant_native_metrics import METHODS, PRIMARY_ROI, common_support, statistics
from ismrm_v2.evaluation.v2_3_boundary_mechanism_metrics import mechanism_statistics
from ismrm_v2.evaluation.v2_3_three_variant_visualization import render_plane


def test_three_way_support_metrics_and_roi():
    ref = np.arange(25., dtype=float).reshape(5, 5)
    pred = {name: ref + i for i, name in enumerate(METHODS)}
    valid = {name: np.ones(ref.shape, bool) for name in METHODS}
    valid[METHODS[0]][0, 0] = False
    valid[METHODS[1]][0, 1] = False
    pred[METHODS[2]][0, 2] = np.nan
    mask = common_support(ref, pred, valid, np.ones(ref.shape, bool))
    assert mask.sum() == 22 and not mask[0, :3].any()
    report = statistics(ref, pred, mask, "t1")
    assert PRIMARY_ROI == "myocardium_core_1px" and set(report["methods"]) == set(METHODS)
    for i, m in enumerate(METHODS):
        assert report["methods"][m]["support_voxels"] == 22
        assert report["methods"][m]["rmse"] == pytest.approx(i)
        assert {"mae", "nrmse", "ncc", "bias", "psnr"} <= report["methods"][m].keys()


def test_mechanism_and_cardiac_crop(tmp_path):
    ref = np.full((32, 32), 50.); core = np.zeros((32, 32), bool); core[15:17, 15:17] = True
    full = np.zeros_like(core); full[14:18, 14:18] = True
    predictions = {name: np.where(full, 100., 60.) for name in METHODS}
    support = np.ones_like(core)
    report = mechanism_statistics(ref, predictions, support, core, full)
    assert report["methods"][METHODS[0]]["contrast_abs_ms"] == 40
    assert report["methods"][METHODS[0]]["residual_mad"] == 0
    figure = render_plane(ref, predictions, support, full, "t1", tmp_path / "plot.png")
    assert figure["crop_bounds_rc"] == [11, 21, 11, 21]
    assert figure["residual_range_ms"] == [-300., 300.]
    assert (tmp_path / "plot.png").is_file()


def test_old_entrypoints_import():
    from ismrm_v2.evaluation.final5_partitioned_native_metrics import build_final5_partitioned_native_metrics
    from ismrm_v2.evaluation.final_sparse_native_metrics import build_final_sparse_native_metrics
    from ismrm_v2.evaluation.aha.final5_myocardial_roi_metrics import build_final5_myocardial_roi_metrics
    from ismrm_v2.evaluation.final5_publication_visualization import render_final5_publication_visualization
    assert all(callable(x) for x in (build_final5_partitioned_native_metrics, build_final_sparse_native_metrics, build_final5_myocardial_roi_metrics, render_final5_publication_visualization))


def test_long_axis_reformats_synthetic(tmp_path):
    import nibabel as nib
    from ismrm_v2.evaluation.v2_3_three_variant_visualization import render_sax_reformats
    affine_lps = np.diag([-1., -1., 1., 1.])
    affines = np.repeat(affine_lps[None], 4, axis=0)
    affines[:, 2, 3] = np.arange(4)
    data = np.broadcast_to(np.arange(20.)[:, None], (20, 20)) + 500
    cache = {"group_idx": np.arange(4), "affine_lps_rc": affines, "slice_thickness_mm": np.ones(4), "source_support": np.ones((4, 20, 20), bool), "t1_ms": np.repeat(data[None], 4, axis=0), "t2_ms": np.repeat((data / 10)[None], 4, axis=0)}
    roi = np.zeros((20, 20), bool); roi[7:13, 7:13] = True
    images = {n: {"t1": nib.Nifti1Image(np.repeat(data[:, :, None], 4, axis=2), np.eye(4)), "t2": nib.Nifti1Image(np.repeat((data / 10)[:, :, None], 4, axis=2), np.eye(4))} for n in METHODS}
    records = render_sax_reformats(cache, images, {0: roi, 3: roi}, tmp_path)
    assert len(records) == 4
    assert all((tmp_path / __import__("pathlib").Path(r["path"]).name).is_file() for r in records)


def test_pipeline_synthetic_inputs(tmp_path, monkeypatch):
    import json
    import nibabel as nib
    import ismrm_v2.evaluation.v2_3_three_variant_native_metrics as module
    from ismrm_v2.fit_first.contracts import sha256
    sparse = tmp_path / "sparse.json"
    sparse.write_text(json.dumps({"subject": "S", "case_id": "L50_fold0", "status": "ELIGIBLE", "stacks": {"2ch": {"retained_original_group_ids": [], "heldout_original_group_ids": [0]}, "4ch": {"retained_original_group_ids": [0], "heldout_original_group_ids": []}}}))
    roots, bundles = [], []
    for method in METHODS:
        root, bundle = tmp_path / (method + "_root"), tmp_path / (method + "_bundle")
        root.mkdir(); bundle.mkdir(); roots.append(root); bundles.append(bundle)
        volpaths = {}
        for p in ("t1", "t2"):
            path = root / f"{p.upper()}_3D.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((12, 12, 4)) * (100 if p == "t1" else 50), np.eye(4)), path)
            volpaths[p] = {"sha256": sha256(path)}
        (root / "manifest.json").write_text(json.dumps({"schema": "phase4r_formal_proposed_export/v1", "subject": "S", "condition": "L50_fold0", "endpoint": "central", "source_checkpoint": {"sha256": method}, "source_volumes": volpaths}))
        planes = []
        for stack in ("sax", "4ch"):
            path = bundle / (stack + ".npz"); np.savez(path, t1_ms=np.full((12, 12), 100.), t2_ms=np.full((12, 12), 50.))
            planes.append({"stack": stack, "original_group_idx": 0, "path": path.name, "sha256": sha256(path)})
        (bundle / "manifest.json").write_text(json.dumps({"schema": "direct_sparse_native_prediction_bundle/v1", "subject": "S", "case_id": "L50_fold0", "endpoint": "central", "sparse_manifest": {"sha256": sha256(sparse)}, "checkpoint": {"sha256": method}, "planes": planes}))
    roiroot = tmp_path / "roi"; roiroot.mkdir()
    for name in ("Segmentation_1-out-label.nii.gz", "Segmentation_2-in-label.nii.gz"):
        nib.save(nib.Nifti1Image(np.ones((12, 12, 1)), np.eye(4)), roiroot / name)
    mapping = tmp_path / "map.json"; mapping.write_text("{}")
    standard = tmp_path / "standard.json"; standard.write_text("{}")
    native_mask = tmp_path / "mask"; native_mask.mkdir(); (native_mask / "manifest.json").write_text("{}")
    native_root = tmp_path / "native"; native_root.mkdir()
    cache = {"group_idx": np.array([0]), "t1_ms": np.full((1, 12, 12), 100.), "t2_ms": np.full((1, 12, 12), 50.), "affine_lps_rc": np.diag([-1., -1., 1., 1.])[None], "pixel_spacing_rc_mm": np.ones((1, 2)), "slice_thickness_mm": np.ones(1), "source_support": np.ones((1, 12, 12), bool), "provenance": {}}
    roi = np.zeros((12, 12), bool); roi[3:9, 3:9] = True
    monkeypatch.setattr(module, "resolve_subject_roi", lambda *_: roiroot)
    monkeypatch.setattr(module, "require_exact_native_roi_mapping", lambda *a, **kw: {"planes": [{"native_group_idx": 0}], "standardization_manifest": {"path": str(standard)}})
    monkeypatch.setattr(module, "mapped_roi_plane", lambda *a, **kw: {"rois": {"myocardium_full": roi, "myocardium_core_1px": roi}})
    monkeypatch.setattr(module, "_load_all_verified_native_cache_groups", lambda *a: cache)
    monkeypatch.setattr(module, "load_validated_current_native_mask_plane", lambda *a, **kw: ({"primary_native_mask": np.ones((12, 12), bool)}, {}))
    out = module.run(torch_mse_root=roots[0], torch_hgnll10_root=roots[1], torch_hgnll1_root=roots[2], torch_mse_bundle=bundles[0], torch_hgnll10_bundle=bundles[1], torch_hgnll1_bundle=bundles[2], native_mask_root=native_mask, r0_native_maps_root=native_root, sparse_manifest=sparse, roi_root=roiroot, roi_mapping=mapping, output=tmp_path / "result", standardization_manifest=standard)
    data = json.loads((out / "aggregate_metrics.json").read_text())
    assert data["partitions"]["t1"]["all"]["common_native"]["support_voxels"] == 432
    assert data["partitions"]["t1"]["heldout"][PRIMARY_ROI]["support_voxels"] == 0
    assert len(json.loads((out / "mechanism_metrics.json").read_text())["planes"]) == 2
