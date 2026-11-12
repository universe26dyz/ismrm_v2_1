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
    import nibabel as nib
    import ismrm_v2.evaluation.v2_3_three_variant_native_metrics as module
    import ismrm_v2.evaluation.v2_3_three_variant_visualization as visual
    sparse, _, roots = training_fixture
    roiroot = tmp_path / "roi"; roiroot.mkdir()
    for name in ("Segmentation_1-out-label.nii.gz", "Segmentation_2-in-label.nii.gz"):
        nib.save(nib.Nifti1Image(np.ones((12, 12, 1)), np.eye(4)), roiroot / name)
    mapping = tmp_path / "map.json"; mapping.write_text("{}")
    standard = tmp_path / "standard.json"; standard.write_text("{}")
    native_mask = tmp_path / "mask"; native_mask.mkdir(); (native_mask / "manifest.json").write_text("{}")
    native_root = tmp_path / "native"; native_root.mkdir()
    cache = {"group_idx": np.array([0]), "t1_ms": np.full((1, 12, 12), 100.), "t2_ms": np.full((1, 12, 12), 50.), "affine_lps_rc": np.diag([-1., -1., 1., 1.])[None], "pixel_spacing_rc_mm": np.ones((1, 2)), "slice_thickness_mm": np.ones(1), "source_support": np.ones((1, 12, 12), bool), "provenance": {}}
    roi = np.zeros((12, 12), bool); roi[3:9, 3:9] = True
    monkeypatch.setattr(module, "validate_sparse_profile_manifest", lambda path, **kwargs: training_fixture[1])
    monkeypatch.setattr(module, "resolve_subject_roi", lambda *_: roiroot)
    monkeypatch.setattr(module, "require_exact_native_roi_mapping", lambda *a, **kw: {"planes": [{"native_group_idx": 0}], "standardization_manifest": {"path": str(standard)}})
    monkeypatch.setattr(module, "mapped_roi_plane", lambda *a, **kw: {"rois": {"myocardium_full": roi, "myocardium_core_1px": roi}})
    monkeypatch.setattr(module, "_load_all_verified_native_cache_groups", lambda *a: cache)
    monkeypatch.setattr(module, "load_validated_current_native_mask_plane", lambda *a, **kw: ({"primary_native_mask": np.ones((12, 12), bool)}, {}))
    captured = []
    def capture(ref, pred, support, crop, parameter, output, **kwargs):
        captured.append((parameter, pred[METHODS[0]].copy()))
        return {"path": str(output)}
    monkeypatch.setattr(visual, "render_plane", capture)
    out = module.run(torch_mse_root=roots[0], torch_hgnll10_root=roots[1], torch_hgnll1_root=roots[2], native_mask_root=native_mask, r0_native_maps_root=native_root, sparse_manifest=sparse, roi_root=roiroot, roi_mapping=mapping, output=tmp_path / "result", standardization_manifest=standard, require_deterministic_torch_provenance=True)
    data = json.loads((out / "aggregate_metrics.json").read_text())
    assert data["partitions"]["t1"]["all"]["common_native"]["support_voxels"] == 432
    assert data["partitions"]["t1"]["observed"]["common_native"]["methods"][METHODS[0]]["bias"] == 10
    assert data["partitions"]["t1"]["heldout"]["common_native"]["methods"][METHODS[0]]["bias"] == 20
    assert data["partitions"]["t1"]["heldout"][PRIMARY_ROI]["support_voxels"] == 0
    assert np.all(captured[0][1] == 110)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["authority_mode"] == "q002_training_roots" and manifest["comparability_audit"]["status"] == "PASS"
    assert all(manifest["methods"][n]["backend_provenance"]["status"] == "PASS" for n in METHODS)


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
