import numpy as np
import json
import pytest
import torch
from ismrm_v2.fit_first.contracts import sha256
from ismrm_v2.proposed.q002.contracts import build_q002_route_config
from ismrm_v2.evaluation.v2_3_h10_detach_native_metrics import audit_h10_detach


def test_historical_and_detach_authority(tmp_path):
    roots = []
    for detach in (False, True):
        root = tmp_path / str(detach); root.mkdir(); roots.append(root)
        name = "B1_R0_HGNLL_DETACH" if detach else "B1_R0_HGNLL"
        route = build_q002_route_config({"training": {}, "decoder": {}}, cropped_prepared_root="cropped", experiment_id=name, anchor_batch_size=640)["route"]
        h = route["heteroscedastic"]
        if not detach:
            for field in ("variance_parameterization", "variance_output_dim", "variance_shared_across_weights", "variance_detach_shared_latent"):
                h.pop(field)
        torch.save({"hgnll_config": h, "model_state": {"joint_variance_head.network.0.weight": torch.zeros(32, 32), "joint_variance_head.network.0.bias": torch.zeros(32), "joint_variance_head.network.2.weight": torch.zeros(10, 32), "joint_variance_head.network.2.bias": torch.zeros(10)}}, root / "model.pt")
        manifest = {"experiment_id": name, "route": route, "loss": "HGNLL", "hgnll": h, "q002_checkpoint_sha256": sha256(root / "model.pt"), "subject_id": "S", "case_id": "L50_fold0", "profile_id": "p", "dependency_sha256": {"x": "a"}, "prepared_inputs_sha256": {"x": "b"}, "controls": {"seed": 20260911}, "sparse_manifest": {"sha256": "c", "generator_git_sha": "d"}}
        for file in ("q002_route_manifest.json", "experiment_manifest.json"): (root / file).write_text(json.dumps(manifest))
    assert audit_h10_detach(*roots)["status"] == "PASS"
    import copy
    for index, key, value in ((0, "output", "bad"), (0, "weights_per_anchor", 1), (1, "variance_detach_shared_latent", False), (1, "variance_learning_rate_key", "other"), (1, "sigma_max", .5)):
        root = roots[index]; original = json.loads((root / "experiment_manifest.json").read_text()); altered = copy.deepcopy(original)
        altered["hgnll"][key] = value; altered["route"]["heteroscedastic"][key] = value
        for file in ("q002_route_manifest.json", "experiment_manifest.json"): (root / file).write_text(json.dumps(altered))
        with pytest.raises(ValueError): audit_h10_detach(*roots)
        for file in ("q002_route_manifest.json", "experiment_manifest.json"): (root / file).write_text(json.dumps(original))
    for key in ("dependency_sha256", "prepared_inputs_sha256", "sparse_manifest"):
        root = roots[1]; original = json.loads((root / "experiment_manifest.json").read_text()); altered = copy.deepcopy(original)
        altered[key] = {"changed": "value"}
        for file in ("q002_route_manifest.json", "experiment_manifest.json"): (root / file).write_text(json.dumps(altered))
        with pytest.raises(ValueError): audit_h10_detach(*roots)
        for file in ("q002_route_manifest.json", "experiment_manifest.json"): (root / file).write_text(json.dumps(original))
    path = roots[1] / "model.pt"
    checkpoint = torch.load(path, weights_only=False); checkpoint["model_state"]["joint_variance_head.network.2.weight"] = torch.zeros(1, 32); torch.save(checkpoint, path)
    manifest = json.loads((roots[1] / "experiment_manifest.json").read_text()); manifest["q002_checkpoint_sha256"] = sha256(path)
    for file in ("q002_route_manifest.json", "experiment_manifest.json"): (roots[1] / file).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="shape"):
        audit_h10_detach(*roots)


def test_summary_improvement_and_missing_values(tmp_path):
    import csv
    from ismrm_v2.evaluation.v2_3_h10_detach_summary import summarize, METHODS
    for subject, vals in (("A", (10., 8., 6.)), ("B", (20., 25., None))):
        root = tmp_path / subject; root.mkdir()
        (root / "manifest.json").write_text(json.dumps({"schema": "v2_3_h10_detach_native/v1", "subject": subject}))
        methods = {name: ({"rmse": v, "mae": v, "nrmse": v} if v is not None else None) for name, v in zip(METHODS, vals)}
        (root / "aggregate_metrics.json").write_text(json.dumps({"partitions": {"t1": {"heldout_2ch": {"myocardium_core_1px": {"support_voxels": 3, "methods": methods}}}}}))
    output = summarize(tmp_path, tmp_path / "summary")
    with (output / "paired_method_deltas.csv").open() as f: deltas = list(csv.DictReader(f))
    assert float(next(x for x in deltas if x["subject"] == "A" and x["comparison"] == "detach_vs_h10" and x["metric"] == "rmse")["improvement_percent"]) == 25.
    with (output / "cohort_mean_metrics.csv").open() as f: means = list(csv.DictReader(f))
    row = next(x for x in means if x["method"] == METHODS[2] and x["metric"] == "rmse")
    assert float(row["mean"]) == 6 and int(row["n_subjects"]) == 1


def test_detach_visualization_and_common_support(tmp_path):
    import numpy as np
    from ismrm_v2.evaluation.v2_3_h10_detach_native_metrics import METHODS, common_support
    from ismrm_v2.evaluation.v2_3_h10_detach_visualization import render_plane
    ref = np.ones((20, 20)); roi = np.zeros((20, 20), bool); roi[7:13, 7:13] = True
    predictions = {name: ref.copy() for name in METHODS}; valid = {name: np.ones_like(ref, bool) for name in METHODS}
    valid[METHODS[-1]][8, 8] = False
    support = common_support(ref, predictions, valid, np.ones_like(ref, bool))
    assert not support[8, 8] and support.sum() == 399
    info = render_plane(ref, predictions, support, roi, "t2", tmp_path / "plot.png")
    assert (tmp_path / "plot.png").exists() and info["map_range_ms"] == [0., 200.] and info["residual_range_ms"] == [-30., 30.]

from test_three_variant_native import training_fixture


def test_detach_full_pipeline(tmp_path, monkeypatch, training_fixture):
    import json
    import ismrm_v2.evaluation.v2_3_h10_detach_native_metrics as module
    import ismrm_v2.evaluation.v2_3_h10_detach_visualization as visual
    from ismrm_v2.fit_first.contracts import sha256
    from scipy.ndimage import binary_erosion
    from ismrm_v2.evaluation.v2_3_h10_detach_native_metrics import METHODS, PRIMARY_ROI
    sparse, payload, roots = training_fixture
    # The second SAX and the 2CH plane are held out; each target intersects the SAX volume.
    payload["stacks"]["sax"]["heldout_original_group_ids"] = [1]
    payload["stacks"]["4ch"]["heldout_original_group_ids"] = [1]
    from ismrm_v2.proposed.q002.contracts import build_q002_route_config
    for index in (1, 2):
        root = roots[index]; name = "B1_R0_HGNLL" if index == 1 else "B1_R0_HGNLL_DETACH"
        manifest = json.loads((root / "q002_route_manifest.json").read_text())
        route = build_q002_route_config({"training": {}, "decoder": {}}, cropped_prepared_root="cropped", experiment_id=name, anchor_batch_size=640)["route"]
        h = route["heteroscedastic"]
        checkpoint = {"hgnll_config": h, "model_state": {"joint_variance_head.network.0.weight": torch.zeros(32, 32), "joint_variance_head.network.0.bias": torch.zeros(32), "joint_variance_head.network.2.weight": torch.zeros(10, 32), "joint_variance_head.network.2.bias": torch.zeros(10)}}
        torch.save(checkpoint, root / "model.pt")
        manifest.update({"experiment_id": name, "route": route, "hgnll": h, "q002_checkpoint_sha256": sha256(root / "model.pt")})
        manifest["sparse_manifest"]["generator_git_sha"] = "gen"
        for filename in ("experiment_manifest.json", "q002_route_manifest.json"): (root / filename).write_text(json.dumps(manifest))
    roiroot = tmp_path / "roi"; roiroot.mkdir()
    outer = np.zeros((2, 12, 12), bool); outer[:, 2:10, 2:10] = True
    inner = np.zeros_like(outer); inner[:, 5:7, 5:7] = True
    full = outer & ~inner
    core = np.stack([binary_erosion(x, structure=np.ones((3, 3), bool)) for x in full])
    archive = roiroot / "sax_myocardium_masks.npz"
    np.savez(archive, group_idx=np.arange(2), outer_filled=outer, inner_filled=inner, blood_pool=inner, myocardium_full=full, myocardium_core_legacy=core, myocardium_core_1px=core, landmark_labels=np.zeros_like(outer))
    (roiroot / "manifest.json").write_text(json.dumps({"schema": "exact_native_myocardium_bundle/v2", "status": "PASS", "subject_id": "S", "outputs": {"sax_myocardium_masks.npz": sha256(archive)}}))
    native_root = tmp_path / "native"; native_root.mkdir()
    def make_cache(n, affines):
        return {"group_idx": np.arange(n), "t1_ms": np.full((n, 12, 12), 100.), "t2_ms": np.full((n, 12, 12), 50.), "affine_lps_rc": affines, "pixel_spacing_rc_mm": np.ones((n, 2)), "slice_thickness_mm": np.ones(n), "source_support": np.ones((n, 12, 12), bool), "provenance": {}}
    lps = np.diag([-1., -1., 1., 1.]); sax_affines = np.repeat(lps[None], 2, axis=0); sax_affines[1, 2, 3] = 1
    target = np.array([[1., 0., 0., 0.], [0., 0., 1., 3.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
    caches = {"sax": make_cache(2, sax_affines), "2ch": make_cache(1, (lps @ target)[None]), "4ch": make_cache(2, np.repeat((lps @ target)[None], 2, axis=0))}
    monkeypatch.setattr(module, "validate_sparse_profile_manifest", lambda path, **kwargs: payload)
    monkeypatch.setattr(module, "_load_all_verified_native_cache_groups", lambda root, path, stack: caches[stack])
    captured = []
    def capture(ref, pred, support, crop, parameter, output, **kwargs):
        captured.append((str(output), parameter, pred[METHODS[0]].copy(), crop.copy()))
        return {"path": str(output)}
    monkeypatch.setattr(visual, "render_plane", capture)
    monkeypatch.setattr(visual, "render_sax_reformats", lambda *args: [])
    out = module.run(torch_mse_root=roots[0], torch_hgnll10_root=roots[1], torch_hgnll10_detach_root=roots[2], r0_native_maps_root=native_root, sparse_manifest=sparse, roi_bundle=roiroot, output=tmp_path / "result", require_deterministic_torch_provenance=True)
    data = json.loads((out / "aggregate_metrics.json").read_text())["partitions"]["t1"]
    assert data["observed"]["common_native"]["methods"][METHODS[0]]["bias"] == 10
    assert data["heldout"]["common_native"]["methods"][METHODS[0]]["bias"] == 20
    assert data["heldout_2ch"][PRIMARY_ROI]["support_voxels"] > 0
    assert data["heldout_4ch"][PRIMARY_ROI]["support_voxels"] > 0
    assert set(data["all"]) == {"common_native", "lv_outer_filled", "blood_pool", "myocardium_full", "myocardium_core_legacy", "myocardium_core_1px"}
    assert np.all(captured[0][2] == 110)
    from ismrm_v2.evaluation.aha.common_native_all_stack_roi_metrics import stack_sax_masks_as_volume, project_sax_mask_volume
    from ismrm_v2.evaluation.final_native_grid_metrics import native_ras_affine
    volumes, affine = stack_sax_masks_as_volume(caches["sax"], {"outer": outer})
    for stack in ("sax", "2ch", "4ch"):
        crop = next(mask for path, _, _, mask in captured if f"/{stack}/" in path)
        expected = project_sax_mask_volume(volumes["outer"], affine, native_ras_affine(caches[stack]["affine_lps_rc"][0]), (12, 12))
        assert crop.any() and np.array_equal(crop, expected)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["primary_roi"] == PRIMARY_ROI and manifest["roi_authority"]["archive"]["sha256"] == sha256(archive)
    assert manifest["roi_authority"]["archive_provenance"]["layout"] == "current_writer"
    assert manifest["authority_mode"] == "q002_training_roots" and manifest["comparability_audit"]["status"] == "PASS"
    assert all(manifest["methods"][n]["backend_provenance"]["status"] == "PASS" for n in METHODS)
    archive.write_bytes(b"corrupted archive")
    with pytest.raises(ValueError, match="archive SHA mismatch"):
        module.run(torch_mse_root=roots[0], torch_hgnll10_root=roots[1], torch_hgnll10_detach_root=roots[2], r0_native_maps_root=native_root, sparse_manifest=sparse, roi_bundle=roiroot, output=tmp_path / "bad-result", require_deterministic_torch_provenance=True)

