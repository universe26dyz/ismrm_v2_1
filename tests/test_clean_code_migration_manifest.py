from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"ACTIVE_ENTRYPOINT", "ACTIVE_EXECUTED", "ACTIVE_TRANSITIVE", "CONDITIONAL_ACTIVE", "CONFIG_RUNTIME_ASSET", "THIRD_PARTY_ACTIVE", "COMPATIBILITY_IMPORT_ONLY"}


def test_manifest_covers_inventory_without_premature_deletion() -> None:
    inventory = json.loads((ROOT / "docs/active_dependency_inventory.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "docs/clean_code_migration_manifest.json").read_text(encoding="utf-8"))
    entries = {entry["current_path"]: entry for entry in manifest["entries"]}
    assert manifest["schema_version"] == "clean-code-migration/v1"
    assert manifest["entry_count"] == len(inventory["nodes"])
    assert entries.keys() == {node["path"] for node in inventory["nodes"]}
    for node in inventory["nodes"]:
        entry = entries[node["path"]]
        assert entry["source_sha256"] == node["source_sha256"]
        assert entry["classification"] == node["classification"]
        assert entry["stage_roles"] == node["stage_roles"]
        assert entry["safe_to_delete_now"] is False
    assert REQUIRED <= {entry["classification"] for entry in entries.values()}


def test_compatibility_and_standardization_are_deferred_not_reactivated() -> None:
    manifest = json.loads((ROOT / "docs/clean_code_migration_manifest.json").read_text(encoding="utf-8"))
    compatibility = [entry for entry in manifest["entries"] if entry["classification"] == "COMPATIBILITY_IMPORT_ONLY"]
    assert compatibility
    assert {entry["future_action"] for entry in compatibility} == {"REMOVE_AFTER_IMPORT_DECOUPLING"}
    assert not any("standard" in str(entry["future_target_path"]).lower() and entry["classification"] in REQUIRED - {"COMPATIBILITY_IMPORT_ONLY"} for entry in manifest["entries"])


def test_duplicate_implementations_are_marked_for_later_merge() -> None:
    manifest = json.loads((ROOT / "docs/clean_code_migration_manifest.json").read_text(encoding="utf-8"))
    duplicates = [item for item in manifest["duplicate_implementation_plan"] if item["relationship"] == "DUPLICATE_IMPLEMENTATION"]
    assert len(duplicates) == 2
    assert all(item["phase1_action"] == "MERGE_LATER" and item["future_action"] == "MERGE_LATER" for item in duplicates)


def test_phase2a_subject_admission_moves_are_recorded_without_old_inventory_nodes() -> None:
    manifest = json.loads((ROOT / "docs/clean_code_migration_manifest.json").read_text(encoding="utf-8"))
    entries = {entry["current_path"]: entry for entry in manifest["entries"]}
    moved = {
        "scripts/validate_subject_id.py": ("src/ismrm_v2/io/validate_subject_id.py", "MOVE"),
        "scripts/build_subject_deployment_manifest.py": ("src/ismrm_v2/io/build_subject_deployment_manifest.py", "MOVE_WITH_IMPORT_ADAPTATION"),
    }
    for old_path, (new_path, kind) in moved.items():
        assert old_path not in entries
        assert entries[new_path]["migration_status"] == "MIGRATED_PHASE_2A"
        assert entries[new_path]["future_target_path"] == new_path
        record = next(item for item in manifest["completed_migrations"] if item["from_path"] == old_path)
        assert record["to_path"] == new_path
        assert record["kind"] == kind
        assert record["pre_move_source_sha256"]
        assert record["post_move_source_sha256"]


def test_phase2b_deployment_manifest_move_is_recorded_without_old_inventory_node() -> None:
    manifest = json.loads((ROOT / "docs/clean_code_migration_manifest.json").read_text(encoding="utf-8"))
    entries = {entry["current_path"]: entry for entry in manifest["entries"]}
    old_path = "common/scripts/deployment_manifest.py"
    new_path = "src/ismrm_v2/io/deployment_manifest.py"
    assert old_path not in entries
    assert entries[new_path]["migration_status"] == "MIGRATED_PHASE_2B"
    assert entries[new_path]["future_target_path"] == new_path
    record = next(item for item in manifest["completed_migrations"] if item["from_path"] == old_path)
    assert record["to_path"] == new_path
    assert record["kind"] == "MOVE"
    assert record["pre_move_source_sha256"] == record["post_move_source_sha256"]


def test_phase2c_foundation_moves_and_import_only_adaptations_are_recorded() -> None:
    manifest = json.loads((ROOT / "docs/clean_code_migration_manifest.json").read_text(encoding="utf-8"))
    entries = {entry["current_path"]: entry for entry in manifest["entries"]}
    moved = {
        "common/trad/scripts/validate_transferred_preprocessed.py": "src/ismrm_v2/preprocessing/validation.py",
        "common/trad/scripts/prepare_all_observations.py": "src/ismrm_v2/observations/prepare_all.py",
        "common/trad/modules/module_02_data_bridge/build_manifest.py": "src/ismrm_v2/observations/build_manifest.py",
        "common/trad/modules/module_02_data_bridge/mat_v73.py": "src/ismrm_v2/observations/mat_v73.py",
        "common/trad/modules/module_02_data_bridge/timing.py": "src/ismrm_v2/observations/timing.py",
        "common/trad/modules/module_02_data_bridge/prepare_observations.py": "src/ismrm_v2/observations/prepare_observations.py",
        "common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse.py": "src/ismrm_v2/sampling/sparse.py",
        "common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse_profile_manifest.py": "src/ismrm_v2/sampling/sparse_profile_manifest.py",
    }
    for old_path, new_path in moved.items():
        assert old_path not in entries
        assert entries[new_path]["migration_status"] == "MIGRATED_PHASE_2C"
        record = next(item for item in manifest["completed_migrations"] if item["from_path"] == old_path)
        assert record["phase"] == "2C"
        assert record["to_path"] == new_path
        assert record["pre_move_source_sha256"]
        assert record["post_move_source_sha256"]
    adapted = {item["path"] for item in manifest["import_path_adaptations"]}
    assert "common/quality_experiments/Q002_B6_roi_same_anchor_mse/contracts.py" in adapted
    assert "common/quality_experiments/Q002_B6_roi_same_anchor_mse/run_q002_reconstruction.py" in adapted
    assert all(item["classification"] == "IMPORT_PATH_ADAPTATION_ONLY" for item in manifest["import_path_adaptations"])
