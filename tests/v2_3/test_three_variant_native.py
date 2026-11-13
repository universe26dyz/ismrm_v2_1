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




@pytest.fixture
def training_fixture(tmp_path):
    import json
    import nibabel as nib
    from ismrm_v2.fit_first.contracts import sha256
    from ismrm_v2.proposed.q002.contracts import build_q002_route_config
    from ismrm_v2.proposed.q002.training import q002_training_controls
    training = {"stage_a_iterations": 2000, "stage_b_iterations": 4000, "batch_size": 640, "psf_samples": 8, "seed": 20260911}
    sparse = tmp_path / "sparse.json"
    stacks = {}
    for stack in ("sax", "2ch", "4ch"):
        active = stack != "2ch"
        stacks[stack] = {"retained_original_group_ids": [0] if active else [], "heldout_original_group_ids": [] if active else [0], "training_stack_present": active}
        if active:
            stacks[stack].update({"derived_to_original_group_idx": [{"derived_group_idx": 0, "original_group_idx": 0}], "derived_observation": {"sha256": "input_" + stack}})
    payload = {"subject": "S", "case_id": "L50_fold0", "profile_id": "L50", "status": "ELIGIBLE", "schema_version": "phase4r_sparse_profile_v2", "stacks": stacks}
    sparse.write_text(json.dumps(payload))
    roots = []
    for name, route_id in zip(METHODS, ("B1_R0", "B1_R0_HGNLL", "B1_R0_HGNLL_ANCHORVAR1")):
        root = tmp_path / name; root.mkdir(); roots.append(root)
        (root / "model.pt").write_bytes(name.encode())
        for p in ("t1", "t2"):
            nib.save(nib.Nifti1Image(np.full((12, 12, 4), 120. if p == "t1" else 60.), np.eye(4)), root / f"{p.upper()}_3D.nii.gz")
        mapping = root / "evaluation" / "mapping_central_no_psf"; mapping.mkdir(parents=True)
        for stack in ("sax", "4ch"):
            np.savez(mapping / f"t1_t2_native_plane_{stack}.npz", group_idx=np.zeros(10, int), weight_idx=np.arange(10), t1_ms=np.full((10, 12, 12), 110.), t2_ms=np.full((10, 12, 12), 55.), masks=np.ones((10, 12, 12), bool), timing9_ms=np.zeros((10, 9)), tr_ms=np.ones(10), vps=np.ones(10))
        route = build_q002_route_config({"training": training, "decoder": {}}, cropped_prepared_root="cropped", experiment_id=route_id, anchor_batch_size=640)["route"]
        dep = {"B6_model": "b6", "frozen_decoder": "decoder", "protocol": "protocol"}
        manifest = {"experiment_id": route_id, "subject_id": "S", "case_id": "L50_fold0", "profile_id": "L50", "sparse_manifest": {"sha256": sha256(sparse)}, "route": route, "controls": q002_training_controls(training, anchor_batch_size=640), "q002_checkpoint_sha256": sha256(root / "model.pt"), "prepared_inputs_sha256": {"/data/sax/observations.npz": "input_sax", "/data/4ch/observations.npz": "input_4ch"}, "b6_model_sha256": "b6", "approved_decoder_sha256": "decoder", "protocol_sha256": "protocol", "dependency_sha256": dep}
        if name != "torch_mse": manifest.update({"loss": "HGNLL", "hgnll": route["heteroscedastic"]})
        for file in ("q002_route_manifest.json", "experiment_manifest.json"): (root / file).write_text(json.dumps(manifest))
        (root / "run.log").write_text("USE_TORCH = True\ndeterministic_algorithms = True\nCUBLAS_WORKSPACE_CONFIG = :4096:8\nPYTHONHASHSEED = 20260911\n")
    return sparse, payload, roots


def test_pipeline_training_roots_observed_heldout_and_visualization(tmp_path, monkeypatch, training_fixture):
    import json
    import ismrm_v2.evaluation.v2_3_three_variant_native_metrics as module
    import ismrm_v2.evaluation.v2_3_three_variant_visualization as visual
    from ismrm_v2.fit_first.contracts import sha256
    from scipy.ndimage import binary_erosion
    sparse, payload, roots = training_fixture
    # The second SAX and the 2CH plane are held out; each target intersects the SAX volume.
    payload["stacks"]["sax"]["heldout_original_group_ids"] = [1]
    payload["stacks"]["4ch"]["heldout_original_group_ids"] = [1]
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
    out = module.run(torch_mse_root=roots[0], torch_hgnll10_root=roots[1], torch_hgnll1_root=roots[2], r0_native_maps_root=native_root, sparse_manifest=sparse, roi_bundle=roiroot, output=tmp_path / "result", require_deterministic_torch_provenance=True)
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
    assert manifest["authority_mode"] == "q002_training_roots" and manifest["comparability_audit"]["status"] == "PASS"
    assert all(manifest["methods"][n]["backend_provenance"]["status"] == "PASS" for n in METHODS)
    archive.write_bytes(b"corrupted archive")
    with pytest.raises(ValueError, match="archive SHA mismatch"):
        module.run(torch_mse_root=roots[0], torch_hgnll10_root=roots[1], torch_hgnll1_root=roots[2], r0_native_maps_root=native_root, sparse_manifest=sparse, roi_bundle=roiroot, output=tmp_path / "bad-result", require_deterministic_torch_provenance=True)


def test_clean_roi_projection_and_three_way_support():
    from ismrm_v2.evaluation.aha.common_native_all_stack_roi_metrics import stack_sax_masks_as_volume, project_sax_mask_volume
    from ismrm_v2.evaluation.final_native_grid_metrics import native_ras_affine
    lps = np.diag([-1., -1., 1., 1.]); affines = np.repeat(lps[None], 3, axis=0); affines[:, 2, 3] = np.arange(3)
    native = {"affine_lps_rc": affines}
    core = np.zeros((3, 12, 12), bool); core[:, 3:9, 3:9] = True
    volumes, affine = stack_sax_masks_as_volume(native, {"core": core})
    for i in range(3):
        assert np.array_equal(project_sax_mask_volume(volumes["core"], affine, native_ras_affine(affines[i]), (12, 12)), core[i])
    target = np.array([[1., 0., 0., 0.], [0., 0., 1., 4.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
    projected = project_sax_mask_volume(volumes["core"], affine, target, (12, 12))
    assert projected.any()
    ref = np.ones((12, 12)); predictions = {n: ref.copy() for n in METHODS}; valid = {n: np.ones_like(projected) for n in METHODS}
    index = tuple(np.argwhere(projected)[0]); valid[METHODS[1]][index] = False
    support = common_support(ref, predictions, valid, np.ones_like(projected)) & projected
    assert support.sum() == projected.sum() - 1
    assert all(x["support_voxels"] == support.sum() for x in statistics(ref, predictions, support, "t1")["methods"].values())


def test_authority_identity_and_backend_fail_closed(training_fixture):
    import json
    from ismrm_v2.evaluation.v2_3_three_variant_native_metrics import _training_authority, _derived_groups, _backend_provenance
    sparse, payload, roots = training_fixture
    groups = _derived_groups(payload)
    for name, root in zip(METHODS, roots): _training_authority(name, root, payload, sparse, groups, True)
    with pytest.raises(ValueError, match="identity"): _training_authority(METHODS[2], roots[0], payload, sparse, groups, True)
    with pytest.raises(ValueError, match="identity"): _training_authority(METHODS[0], roots[1], payload, sparse, groups, True)
    root = roots[2]; path = root / "q002_route_manifest.json"; manifest = json.loads(path.read_text())
    manifest["route"]["heteroscedastic"]["variance_output_dim"] = 10
    manifest["hgnll"] = manifest["route"]["heteroscedastic"]
    for file in ("q002_route_manifest.json", "experiment_manifest.json"): (root / file).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="variance identity"): _training_authority(METHODS[2], root, payload, sparse, groups, True)
    path = roots[0] / "run.log"; path.unlink()
    assert _backend_provenance(path, False)["status"] == "NOT_VERIFIABLE_FROM_ROOT"
    with pytest.raises(ValueError): _backend_provenance(path, True)
    path.write_text("USE_TORCH = False\ndeterministic_algorithms = True\nCUBLAS_WORKSPACE_CONFIG = :4096:8\n")
    with pytest.raises(ValueError): _backend_provenance(path, True)


def test_mapping_weight_consistency_and_shape(training_fixture):
    from ismrm_v2.evaluation.v2_3_three_variant_native_metrics import _observed_plane
    path = training_fixture[2][0] / "evaluation/mapping_central_no_psf/t1_t2_native_plane_sax.npz"
    assert np.all(_observed_plane(path, 0, (12, 12))["t1"][0] == 110)
    with np.load(path) as data: arrays = {key: data[key] for key in data.files}
    arrays["t1_ms"][5, 3, 4] += 2
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="inconsistent t1"): _observed_plane(path, 0, (12, 12))
    with pytest.raises(ValueError, match="weights"): _observed_plane(path, 1, (12, 12))


@pytest.mark.parametrize("field", ["protocol_sha256", "approved_decoder_sha256", "b6_model_sha256", "seed", "sparse_sha"])
def test_scientific_provenance_mismatch_fails(training_fixture, field):
    import json
    from ismrm_v2.evaluation.v2_3_three_variant_native_metrics import _training_authority, _derived_groups
    sparse, payload, roots = training_fixture
    root = roots[0]; manifest = json.loads((root / "q002_route_manifest.json").read_text())
    if field == "seed": manifest["controls"]["seed"] = 1
    elif field == "sparse_sha": manifest["sparse_manifest"]["sha256"] = "wrong"
    else: manifest[field] = "wrong"
    for file in ("q002_route_manifest.json", "experiment_manifest.json"): (root / file).write_text(json.dumps(manifest))
    with pytest.raises(ValueError): _training_authority(METHODS[0], root, payload, sparse, _derived_groups(payload), True)


def test_cross_method_scientific_comparability(training_fixture):
    import json
    from ismrm_v2.evaluation.v2_3_three_variant_native_metrics import _training_authority, _derived_groups, _audit_shared_contracts
    sparse, payload, roots = training_fixture
    root = roots[1]; manifest = json.loads((root / "q002_route_manifest.json").read_text())
    manifest["protocol_sha256"] = "different_protocol"
    manifest["dependency_sha256"]["protocol"] = "different_protocol"
    for file in ("q002_route_manifest.json", "experiment_manifest.json"): (root / file).write_text(json.dumps(manifest))
    contracts = {name: _training_authority(name, root, payload, sparse, _derived_groups(payload), True)[1] for name, root in zip(METHODS, roots)}
    with pytest.raises(ValueError, match="comparability"): _audit_shared_contracts(contracts)


def test_parent_and_ambiguous_backend_logs(training_fixture):
    from ismrm_v2.evaluation.v2_3_three_variant_native_metrics import _training_authority, _derived_groups
    sparse, payload, roots = training_fixture
    root = roots[0]
    local = root / "run.log"; parent = root.parent / "run.log"
    parent.write_bytes(local.read_bytes()); local.unlink()
    record = _training_authority(METHODS[0], root, payload, sparse, _derived_groups(payload), True)[0]["backend_provenance"]
    from ismrm_v2.fit_first.contracts import sha256
    assert record["status"] == "PASS" and record["run_log"] == {"path": str(parent.resolve()), "sha256": sha256(parent)}
    local.write_bytes(parent.read_bytes())
    record = _training_authority(METHODS[0], root, payload, sparse, _derived_groups(payload), True)[0]["backend_provenance"]
    assert record["run_log"]["path"] == str(local.resolve())
    local.write_text("different")
    with pytest.raises(ValueError, match="ambiguous"):
        _training_authority(METHODS[0], root, payload, sparse, _derived_groups(payload), False)


@pytest.mark.parametrize("schema", ["legacy", "v2"])
def test_real_sparse_profile_identity_normalization(tmp_path, training_fixture, schema):
    import json
    from pathlib import Path
    from ismrm_v2.fit_first.contracts import sha256
    from ismrm_v2.sampling.sparse import build_sparse_condition, build_sparse_profile_condition
    from ismrm_v2.sampling.sparse_profile_manifest import validate_sparse_profile_manifest
    from ismrm_v2.evaluation.v2_3_three_variant_native_metrics import _training_authority, _derived_groups
    source = tmp_path / "source"
    groups = np.repeat(np.arange(4), 10); weights = np.tile(np.arange(10), 4); rows = len(groups)
    affine = np.repeat(np.eye(4)[None], rows, axis=0); affine[:, 2, 2] = 5.; affine[:, 2, 3] = -groups
    for stack in ("sax", "2ch", "4ch"):
        path = source / "S" / stack / "observations.npz"; path.parent.mkdir(parents=True)
        np.savez_compressed(path, images=np.ones((rows, 2, 2)), masks=np.ones((rows, 2, 2), bool), group_idx=groups, weight_idx=weights, stack_idx=np.zeros(rows, int), acquisition_time_ms=np.zeros(rows), timing9_ms=np.tile(np.arange(9), (rows, 1)), affine_lps_rc=affine, pixel_spacing_rc_mm=np.ones((rows, 2)), slice_thickness_mm=np.full(rows, 5.), tr_ms=np.full(rows, 2.61), vps=np.full(rows, 87))
    def audit(path):
        with np.load(path) as data:
            ids = np.unique(data["group_idx"])
            positions = [float(data["affine_lps_rc"][np.flatnonzero(data["group_idx"] == group)[0], 2, 3]) for group in ids]
        return {"regular_gap": True, "gap_mm": abs(positions[1] - positions[0]) if len(ids) > 1 else 0., "group_count": len(ids)}
    if schema == "legacy":
        result = build_sparse_condition(source, "S", "L50", 0, tmp_path / "legacy", git_sha="a" * 40, audit_stack=audit)
    else:
        profile = Path(__file__).resolve().parents[2] / "configs/sparse_profiles/S2CH50_4CH100.json"
        result = build_sparse_profile_condition(source, "S", profile, 0, tmp_path / "profile", git_sha="a" * 40, audit_stack=audit)
    sparse_path = Path(result["manifest_path"]); raw = json.loads(sparse_path.read_text())
    normalized = validate_sparse_profile_manifest(sparse_path, strict=False)
    if schema == "legacy":
        assert "case_id" not in raw and "profile_id" not in raw
        assert (normalized["profile_id"], normalized["case_id"]) == ("legacy_group_id_modulo_L50", "L50_fold0")
    else:
        assert normalized["profile_id"] == raw["profile_id"] and normalized["case_id"] == raw["case_id"]
    root = training_fixture[2][0]; manifest = json.loads((root / "q002_route_manifest.json").read_text())
    manifest.update({"profile_id": normalized["profile_id"], "case_id": normalized["case_id"], "sparse_manifest": {"sha256": sha256(sparse_path)}, "prepared_inputs_sha256": {f"/data/{stack}/observations.npz": item["derived_observation"]["sha256"] for stack, item in normalized["stacks"].items() if item.get("derived_observation")}})
    for file in ("q002_route_manifest.json", "experiment_manifest.json"): (root / file).write_text(json.dumps(manifest))
    mapping_dir = root / "evaluation/mapping_central_no_psf"
    for stack in ("sax", "2ch", "4ch"):
        target = mapping_dir / f"t1_t2_native_plane_{stack}.npz"
        if normalized["stacks"][stack].get("derived_observation") and not target.exists(): target.write_bytes((mapping_dir / "t1_t2_native_plane_sax.npz").read_bytes())
    record, shared, _ = _training_authority(METHODS[0], root, normalized, sparse_path, _derived_groups(normalized), True)
    assert shared["profile_id"] == normalized["profile_id"] and record["experiment_identity"]["experiment_id"] == "B1_R0"


@pytest.mark.parametrize("field,value", [("schema", "wrong"), ("status", "FAIL"), ("subject_id", "other")])
def test_roi_bundle_rejects_invalid_identity(tmp_path, field, value):
    import json
    from ismrm_v2.fit_first.contracts import sha256
    from ismrm_v2.evaluation.v2_3_three_variant_native_metrics import _validate_roi_bundle_manifest
    archive = tmp_path / "sax_myocardium_masks.npz"; archive.write_bytes(b"archive")
    manifest = {"schema": "exact_native_myocardium_bundle/v2", "status": "PASS", "subject_id": "S", "outputs": {archive.name: sha256(archive)}}
    assert _write_roi_manifest_and_validate(tmp_path, manifest, "S") == archive
    manifest[field] = value
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema, status, or subject"):
        _validate_roi_bundle_manifest(tmp_path, "S")


def _write_roi_manifest_and_validate(root, manifest, subject):
    import json
    from ismrm_v2.evaluation.v2_3_three_variant_native_metrics import _validate_roi_bundle_manifest
    (root / "manifest.json").write_text(json.dumps(manifest))
    return _validate_roi_bundle_manifest(root, subject)


def test_roi_bundle_requires_authoritative_hash_and_archive(tmp_path):
    from ismrm_v2.fit_first.contracts import sha256
    from ismrm_v2.evaluation.v2_3_three_variant_native_metrics import _validate_roi_bundle_manifest
    archive = tmp_path / "sax_myocardium_masks.npz"; archive.write_bytes(b"archive")
    manifest = {"schema": "exact_native_myocardium_bundle/v2", "status": "PASS", "subject_id": "S", "sax_myocardium_masks": {"sha256": sha256(archive)}}
    with pytest.raises(ValueError, match="archive SHA mismatch"):
        _write_roi_manifest_and_validate(tmp_path, manifest, "S")
    manifest["outputs"] = {archive.name: sha256(archive)}
    _write_roi_manifest_and_validate(tmp_path, manifest, "S")
    archive.unlink()
    with pytest.raises(FileNotFoundError):
        _validate_roi_bundle_manifest(tmp_path, "S")
