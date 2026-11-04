#!/usr/bin/env python3
"""Build a deterministic, data-free semantic trace of the 01--09 pipeline.

The declared edges below are narrow runtime-boundary overrides.  Everything
else is discovered from Python imports; a declared repository path is never
silently downgraded to an unresolved dependency.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


STAGE_ENTRYPOINTS = {f"{number:02d}": f"pipeline/{name}/run_stage.sh" for number, name in enumerate((
    "01_RAW_DICOM", "02_PREPROCESS_MATLAB", "03_PREPARE_OBSERVATIONS", "04_SPARSE_PROFILES",
    "05_SCMR_V5_FIT_FIRST", "06_PROPOSED", "07_FINAL_POSE", "08_EVALUATION_AND_VISUALIZATION",
    "09_INTEGRITY_AND_REPORTING"), 1)}
STAGE_IO = {
    "01": (['RAW_ROOT/SUBJECT/{sax,2ch,4ch}/*.dcm'], ['subjects/SUBJECT/manifest.json']),
    "02": (['deployment manifest', 'DICOM'], ['RUN_ROOT/preprocessed/SUBJECT/{stack}/preprocessed.mat']),
    "03": (['preprocessed MAT', 'deployment manifest', 'protocol YAML'], ['observations.npz', 'manifest.json', 'timing.npy', 'qc_summary.json']),
    "04": (['prepared observations', 'sparse profile'], ['sparse_inputs/SUBJECT/CASE/sparse_manifest.json']),
    "05": (['preprocessed MAT', 'prepared observations', 'sparse inputs'], ['native cache', 'fit_first reconstruction']),
    "06": (['prepared or sparse observations', 'decoder/B6 assets', 'protocol'], ['model.pt', 'T1_3D.nii.gz', 'T2_3D.nii.gz', 'B1_3D.nii.gz', 'amplitude_3D.nii.gz', 'final_rigid_poses.json']),
    "07": (['completed Proposed/Fit-first outputs', 'ROI bundle'], ['evaluation/final_pose/SUBJECT']),
    "08": (['completed outputs', 'ROI bundle', 'final-pose/native cache'], ['evaluation/common_native/SUBJECT']),
    "09": (['repository contracts'], ['integrity and freeze reports']),
}

# (stage, source, destination, edge type, condition, evidence, classification)
DECLARED_EDGES = [
    ("01", STAGE_ENTRYPOINTS["01"], "src/ismrm_v2/io/validate_subject_id.py", "SHELL_PYTHON_MODULE", "always", "run_stage invokes validator module", "ACTIVE_EXECUTED"),
    ("01", STAGE_ENTRYPOINTS["01"], "src/ismrm_v2/io/build_subject_deployment_manifest.py", "SHELL_PYTHON_MODULE", "always", "run_stage builds manifest module", "ACTIVE_EXECUTED"),
    ("02", STAGE_ENTRYPOINTS["02"], "src/ismrm_v2/io/validate_subject_id.py", "SHELL_PYTHON_MODULE", "always", "run_stage invokes validator module", "ACTIVE_EXECUTED"),
    ("03", STAGE_ENTRYPOINTS["03"], "src/ismrm_v2/io/validate_subject_id.py", "SHELL_PYTHON_MODULE", "always", "run_stage invokes validator module", "ACTIVE_EXECUTED"),
    ("04", STAGE_ENTRYPOINTS["04"], "src/ismrm_v2/io/validate_subject_id.py", "SHELL_PYTHON_MODULE", "always", "run_stage invokes validator module", "ACTIVE_EXECUTED"),
    ("05", STAGE_ENTRYPOINTS["05"], "src/ismrm_v2/io/validate_subject_id.py", "SHELL_PYTHON_MODULE", "always", "run_stage invokes validator module", "ACTIVE_EXECUTED"),
    ("06", STAGE_ENTRYPOINTS["06"], "src/ismrm_v2/io/validate_subject_id.py", "SHELL_PYTHON_MODULE", "always", "run_stage invokes validator module", "ACTIVE_EXECUTED"),
    ("07", STAGE_ENTRYPOINTS["07"], "src/ismrm_v2/io/validate_subject_id.py", "SHELL_PYTHON_MODULE", "always", "run_stage invokes validator module", "ACTIVE_EXECUTED"),
    ("08", STAGE_ENTRYPOINTS["08"], "src/ismrm_v2/io/validate_subject_id.py", "SHELL_PYTHON_MODULE", "always", "run_stage invokes validator module", "ACTIVE_EXECUTED"),
    ("02", STAGE_ENTRYPOINTS["02"], "matlab/preprocessing/local_preprocess_all.m", "MATLAB_FUNCTION", "always", "MATLAB -batch local_preprocess_all", "ACTIVE_EXECUTED"),
    ("02", "matlab/preprocessing/local_preprocess_all.m", "matlab/preprocessing/local_preprocess_one.m", "MATLAB_FUNCTION", "per stack", "local_preprocess_all loop", "ACTIVE_EXECUTED"),
    ("02", "matlab/preprocessing/local_preprocess_one.m", "matlab/preprocessing/preprocessing_v1/preprocess_stack_v1.m", "MATLAB_FUNCTION", "per stack", "explicit preprocess call", "ACTIVE_EXECUTED"),
    ("02", "matlab/preprocessing/local_preprocess_one.m", "matlab/preprocessing/preprocessing_v1/preprocess_options_v1.m", "MATLAB_FUNCTION", "per stack", "explicit options construction", "ACTIVE_TRANSITIVE"),
    ("02", STAGE_ENTRYPOINTS["02"], "src/ismrm_v2/preprocessing/validation.py", "SHELL_PYTHON_MODULE", "always", "post-transfer validator", "ACTIVE_EXECUTED"),
    ("03", STAGE_ENTRYPOINTS["03"], "src/ismrm_v2/observations/prepare_all.py", "SHELL_PYTHON_MODULE", "always", "explicit --protocol", "ACTIVE_EXECUTED"),
    ("03", "src/ismrm_v2/observations/prepare_all.py", "configs/protocol_scmr_v5_10_100.yaml", "CONFIG_READ", "always", "Stage 03 --protocol", "CONFIG_RUNTIME_ASSET"),
    ("04", STAGE_ENTRYPOINTS["04"], "pipeline/04_SPARSE_PROFILES/internal/build_sparse_profiles.sh", "SHELL_BASH", "always", "stage wrapper", "ACTIVE_EXECUTED"),
    ("04", "pipeline/04_SPARSE_PROFILES/internal/build_sparse_profiles.sh", "src/ismrm_v2/sampling/sparse.py", "SHELL_PYTHON_MODULE", "per selected sparse profile", "profile builder", "CONDITIONAL_ACTIVE"),
    ("05", STAGE_ENTRYPOINTS["05"], "pipeline/05_SCMR_V5_FIT_FIRST/internal/run_scmr_v5_fit_first.sh", "SHELL_BASH", "always", "stage wrapper", "ACTIVE_EXECUTED"),
    ("05", "pipeline/05_SCMR_V5_FIT_FIRST/internal/run_scmr_v5_fit_first.sh", "src/ismrm_v2/fit_first/scmr_v5_native_cache.py", "SHELL_PYTHON_MODULE", "always", "SCMR-v5 native cache", "ACTIVE_EXECUTED"),
    ("05", "src/ismrm_v2/fit_first/scmr_v5_native_cache.py", "matlab/scmr_v5/build_scmr_v5_native_maps.m", "MATLAB_FUNCTION", "per stack", "MATLAB mapper subprocess", "ACTIVE_EXECUTED"),
    ("05", "matlab/scmr_v5/build_scmr_v5_native_maps.m", "matlab/scmr_v5/scmr_v5_matcher/function_T1T2_10HB_bssfp_qc.m", "MATLAB_ADDPATH", "per stack", "mapper verifies and addpaths matcher", "ACTIVE_TRANSITIVE"),
    ("05", "matlab/scmr_v5/build_scmr_v5_native_maps.m", "matlab/scmr_v5/scmr_v5_matcher/matlab/dictionary/sim_T1T2_10HB_bssfp.m", "MATLAB_ADDPATH", "per stack", "mapper verifies and addpaths simulator", "ACTIVE_TRANSITIVE"),
    ("05", "matlab/scmr_v5/build_scmr_v5_native_maps.m", "matlab/scmr_v5/scmr_v5_matcher/matlab/utils/d2r.m", "MATLAB_ADDPATH", "per stack", "mapper verifies and addpaths utility", "ACTIVE_TRANSITIVE"),
    ("05", "pipeline/05_SCMR_V5_FIT_FIRST/internal/run_scmr_v5_fit_first.sh", "src/ismrm_v2/fit_first/baseline.py", "SHELL_PYTHON_MODULE", "L100 and sparse cases", "baseline CLI", "CONDITIONAL_ACTIVE"),
    ("06", STAGE_ENTRYPOINTS["06"], "src/ismrm_v2/proposed/proposed.py", "SHELL_PYTHON_MODULE", "always", "python -m proposed", "ACTIVE_EXECUTED"),
    ("06", "src/ismrm_v2/proposed/proposed.py", "src/ismrm_v2/proposed/q002/run_q002_reconstruction.py", "PYTHON_CALL", "formal Q002S640", "Q002 formal reconstruction call", "ACTIVE_EXECUTED"),
    ("06", "src/ismrm_v2/proposed/q002/run_q002_reconstruction.py", "src/ismrm_v2/proposed/q002/training.py", "PYTHON_CALL", "formal Q002S640", "Q002 training call", "ACTIVE_EXECUTED"),
    ("06", "src/ismrm_v2/proposed/q002/training.py", "src/ismrm_v2/proposed/reconstruction/orchestration.py", "PYTHON_CALL", "formal Q002S640", "shared reconstruction engine", "ACTIVE_EXECUTED"),
    ("06", STAGE_ENTRYPOINTS["06"], "configs/protocol_scmr_v5_10_100.yaml", "CONFIG_READ", "always", "--protocol", "CONFIG_RUNTIME_ASSET"),
    ("06", STAGE_ENTRYPOINTS["06"], "configs/models_manifest.json", "CONFIG_READ", "always", "model provenance", "CONFIG_RUNTIME_ASSET"),
    ("06", STAGE_ENTRYPOINTS["06"], "runtime/runtime_contract.json", "CONFIG_READ", "always", "runtime contract", "CONFIG_RUNTIME_ASSET"),
    ("07", STAGE_ENTRYPOINTS["07"], "pipeline/07_FINAL_POSE/internal/hhz_paired_final_pose.sh", "SHELL_BASH", "always", "stage wrapper", "ACTIVE_EXECUTED"),
    ("07", "pipeline/07_FINAL_POSE/internal/hhz_paired_final_pose.sh", "src/ismrm_v2/evaluation/aha/cyj11_r3_direct_final_pose_bundle.py", "SHELL_PYTHON_MODULE", "per case", "Proposed final-pose bundle", "CONDITIONAL_ACTIVE"),
    ("07", "pipeline/07_FINAL_POSE/internal/hhz_paired_final_pose.sh", "src/ismrm_v2/evaluation/aha/current_fit_first_final_pose_bundle.py", "SHELL_PYTHON_MODULE", "per case", "Fit-first final-pose bundle", "CONDITIONAL_ACTIVE"),
    ("07", "pipeline/07_FINAL_POSE/internal/hhz_paired_final_pose.sh", "src/ismrm_v2/evaluation/aha/derive_final_pose_mapping.py", "SHELL_PYTHON_MODULE", "per case", "ROI mapping", "CONDITIONAL_ACTIVE"),
    ("07", "pipeline/07_FINAL_POSE/internal/hhz_paired_final_pose.sh", "src/ismrm_v2/evaluation/aha/paired_final_pose_postprocess.py", "SHELL_PYTHON_MODULE", "per case", "paired AHA metrics", "CONDITIONAL_ACTIVE"),
    ("07", "pipeline/07_FINAL_POSE/internal/hhz_paired_final_pose.sh", "src/ismrm_v2/evaluation/aha/paired_aha_sd.py", "SHELL_PYTHON_MODULE", "twice per case", "AHA SD CLI for 1px and legacy cores", "CONDITIONAL_ACTIVE"),
    ("07", "pipeline/07_FINAL_POSE/internal/hhz_paired_final_pose.sh", "src/ismrm_v2/evaluation/aha/paired_final_pose_through_plane.py", "SHELL_PYTHON_MODULE", "per case", "through-plane QC CLI", "CONDITIONAL_ACTIVE"),
    ("08", STAGE_ENTRYPOINTS["08"], "pipeline/08_EVALUATION_AND_VISUALIZATION/internal/hhz_common_native_metrics_qc.sh", "SHELL_BASH", "always", "first evaluation wrapper", "ACTIVE_EXECUTED"),
    ("08", STAGE_ENTRYPOINTS["08"], "pipeline/08_EVALUATION_AND_VISUALIZATION/internal/hhz_common_native_all_stack_roi_metrics.sh", "SHELL_BASH", "always", "second evaluation wrapper", "ACTIVE_EXECUTED"),
    ("08", "pipeline/08_EVALUATION_AND_VISUALIZATION/internal/hhz_common_native_metrics_qc.sh", "src/ismrm_v2/evaluation/aha/common_native_metrics_qc.py", "SHELL_PYTHON_MODULE", "per case", "common-native metrics CLI", "CONDITIONAL_ACTIVE"),
    ("08", "pipeline/08_EVALUATION_AND_VISUALIZATION/internal/hhz_common_native_all_stack_roi_metrics.sh", "src/ismrm_v2/evaluation/aha/common_native_all_stack_roi_metrics.py", "SHELL_PYTHON_MODULE", "per case", "all-stack ROI CLI", "CONDITIONAL_ACTIVE"),
    ("09", STAGE_ENTRYPOINTS["09"], "scripts/verify_project_integrity.py", "SHELL_PYTHON_SCRIPT", "always", "integrity checker", "ACTIVE_EXECUTED"),
    ("09", STAGE_ENTRYPOINTS["09"], "scripts/verify_no_legacy_code_path.py", "SHELL_PYTHON_SCRIPT", "always", "legacy checker", "ACTIVE_EXECUTED"),
    ("09", STAGE_ENTRYPOINTS["09"], "scripts/verify_runtime_contract.py", "SHELL_PYTHON_SCRIPT", "always", "runtime checker", "ACTIVE_EXECUTED"),
    ("09", STAGE_ENTRYPOINTS["09"], "scripts/verify_scientific_freeze.py", "SHELL_PYTHON_SCRIPT", "always", "freeze checker", "ACTIVE_EXECUTED"),
]
KNOWN_RUNTIME_ASSETS = [
    ("04", "pipeline/04_SPARSE_PROFILES/internal/build_sparse_profiles.sh", f"configs/sparse_profiles/{name}.json", "always within Stage 04 batch", "PROFILE loop and --profile path")
    for name in ("SAX100_2CH0_4CH0", "SAX100_2CH0_4CH100", "SAX100_2CH100_4CH0", "S2CH100_4CH50", "S2CH50_4CH100")
]
ROLE_PRIORITY = {"ACTIVE_ENTRYPOINT": 9, "ACTIVE_EXECUTED": 8, "CONDITIONAL_ACTIVE": 7, "ACTIVE_TRANSITIVE": 6, "COMPATIBILITY_IMPORT_ONLY": 5, "CONFIG_RUNTIME_ASSET": 4, "THIRD_PARTY_ACTIVE": 3, "TEST_ONLY": 2, "LEGACY_UNUSED_BY_PIPELINE": 1}
COMPATIBILITY = {
    "src/ismrm_v2/fit_first/fit_first_intensity_preserving.py": "baseline imports it, but Stage 05 supplies SCMR-v5 prebuilt/reuse flags, never intensity-preserving flags",
    "src/ismrm_v2/fit_first/reuse_standardized_native_dictionary_maps.py": "baseline imports it, but Stage 05 never supplies standardized reuse flags",
    "src/ismrm_v2/fit_first/standardize_subject_slices.py": "standardization route is not selected by Stage 05",
}
AUDITED_UNUSED = {
    "src/ismrm_v2/proposed/q002/run_q004_s640_k1_reconstruction.py": "Q004/K1 experiment is not reached by formal Q002S640 command",
}
EXTERNAL = [
    {"id": "matlab", "kind": "executable", "required_by_stages": ["02", "05"], "resolution": "MATLAB argument or PATH", "locally_available": False},
    {"id": "nesvor", "kind": "executable", "required_by_stages": ["05"], "resolution": "NESVOR environment variable or PATH", "locally_available": False},
    {"id": "cuda", "kind": "runtime_capability", "required_by_stages": ["06", "07"], "resolution": "cuda:0 requested by Stage commands", "locally_available": False},
    {"id": "decoder_scmr_v5_10_100.pth", "kind": "model_asset", "required_by_stages": ["06"], "expected_sha256": "338f2c0b8db90851f733ed31caa2c99914605fed3f7215ec2f69f19eba59583b", "repository_embedded": False},
    {"id": "b6_model.pt", "kind": "model_asset", "required_by_stages": ["06"], "expected_sha256": "15981fbe93d2c70e14755ea756a02856c9383481e7d92d239619268a628a29b5", "repository_embedded": False},
]
PROXIES = [
    {"wrapper": "src/ismrm_v2/proposed/mlp_trainer.py", "implementation": "src/ismrm_v2/proposed/reconstruction/orchestration.py", "relationship": "FACADE", "evidence": "imports and re-exports the sole reconstruction engine", "active_caller": "MLP compatibility route", "cleanup_candidate": True},
    {"wrapper": "src/ismrm_v2/proposed/signal_decoder/frozen_decoder.py", "implementation": "src/ismrm_v2/proposed/trad_signal_decoder/decoder_factory.py", "relationship": "INDEPENDENT_IMPLEMENTATION", "evidence": "frozen MLP decoder class is not a forwarding proxy", "active_caller": "Q002 frozen decoder", "cleanup_candidate": False},
]


def _language(path: str) -> str:
    return {".py": "python", ".sh": "shell", ".m": "matlab", ".json": "json", ".yaml": "yaml"}.get(Path(path).suffix, "other")


def _resolve_import(root: Path, module: str, source: Path, level: int) -> Path | None:
    base = source.parent
    for _ in range(max(level - 1, 0)):
        base = base.parent
    stems = [base.joinpath(*module.split(".")), root / "src" / Path(*module.split(".")), root / "third_party/nesvor" / Path(*module.split(".")), root / "scripts" / Path(*module.split("."))]
    for stem in stems:
        for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
            if candidate.is_file():
                return candidate
    return None


def _imports(root: Path, path: Path) -> list[tuple[Path, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    found = []
    for item in ast.walk(tree):
        if isinstance(item, ast.Import):
            for name in item.names:
                target = _resolve_import(root, name.name, path, 0)
                if target: found.append((target, f"import {name.name}"))
        elif isinstance(item, ast.ImportFrom):
            target = _resolve_import(root, item.module or "", path, item.level)
            if target: found.append((target, f"from {'.' * item.level}{item.module or ''} import"))
    return found


def _matlab_local_functions(path: Path) -> set[str]:
    matches = re.findall(r"(?m)^\s*function(?:\s+\[[^\]]+\]|\s+\w+)?\s*=\s*(\w+)\s*\(|^\s*function\s+(\w+)\s*\(", path.read_text(encoding="utf-8", errors="ignore"))
    return {name for pair in matches for name in pair if name}


def _matlab_calls(root: Path, path: Path) -> list[tuple[Path, str]]:
    """Resolve calls only when their function name is repository-local.

    MATLAB builtins/toolbox calls are deliberately ignored; an apparent call to
    a function with a repository-local implementation must resolve or fail.
    """
    text = re.sub(r"%.*$", "", path.read_text(encoding="utf-8", errors="ignore"), flags=re.M)
    local = _matlab_local_functions(path)
    canonical = {
        "deformableReg2Dmind_asym_nodisplay": "matlab/preprocessing/hhz_original/MIND_registration/deformableReg2Dmind_asym_nodisplay.m",
        "MIND_descriptor2D": "matlab/preprocessing/hhz_original/MIND_registration/MIND_descriptor2D.m",
        "HHMMSS2Sec": "matlab/preprocessing/hhz_original/Utility/HHMMSS2Sec.m",
        "mppca_denoise_multimap_stack": "matlab/preprocessing/preprocessing_v1/mppca_denoise_multimap_stack.m",
        "sim_T1T2_10HB_bssfp": "matlab/scmr_v5/scmr_v5_matcher/matlab/dictionary/sim_T1T2_10HB_bssfp.m",
        "d2r": "matlab/scmr_v5/scmr_v5_matcher/matlab/utils/d2r.m",
    }
    index = {candidate.stem: candidate for candidate in sorted((root / "matlab").glob("**/*.m"))}
    for name, relative in canonical.items():
        candidate = root / relative
        if candidate.is_file():
            index[name] = candidate
    names = set(re.findall(r"(?<![\w.])([A-Za-z]\w*)\s*\(", text)) - local
    calls: list[tuple[Path, str]] = []
    for name in sorted(names):
        target = index.get(name)
        if target is not None and target != path:
            calls.append((target, f"MATLAB call {name}(...)"))
    return calls


def _shell_modules(root: Path, path: Path) -> list[tuple[Path, str]]:
    """Resolve repository Python modules executed by a shell script.

    This deliberately targets the stable ``python -m package.module`` form.
    Variable-expanded filesystem commands remain represented by the narrow
    declared boundary edges above.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    modules = re.findall(r"(?:\"\$PY(?:THON)?\"|\$PY(?:THON)?|python(?:3)?)\s+-m\s+([A-Za-z_]\w*(?:\.\w+)*)", text)
    resolved: list[tuple[Path, str]] = []
    for module in sorted(set(modules)):
        for package_root in (root / "src", root / "third_party/nesvor"):
            target = (package_root / Path(*module.split("."))).with_suffix(".py")
            if target.is_file():
                resolved.append((target, f"shell python -m {module}"))
                break
    return resolved


def _historical(path: str) -> list[str]:
    return [token for token in ("cyj11", "final5", "hhz", "legacy") if token in path.lower()]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generation_provenance(root: Path, nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Fingerprint source content without embedding a self-referential HEAD."""
    fingerprint = hashlib.sha256()
    fingerprint.update(b"active-dependency-inventory/v3\0")
    generator = Path(__file__).resolve()
    generator_sha256 = _sha256_file(generator)
    fingerprint.update(b"scripts/build_active_dependency_inventory.py\0")
    fingerprint.update(generator_sha256.encode("ascii"))
    for path in sorted(nodes):
        source = root / path
        if not source.is_file():
            continue
        fingerprint.update(path.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(nodes[path]["source_sha256"].encode("ascii"))
    return {
        "inventory_schema": "active-dependency-inventory/v3",
        "generator_sha256": generator_sha256,
        "source_runtime_fingerprint_sha256": fingerprint.hexdigest(),
        "source_file_count": sum((root / path).is_file() for path in nodes),
    }


def scan(root: Path) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}; edges: list[dict[str, Any]] = []; direct: dict[tuple[str, str], str] = {}
    stage_paths: dict[str, set[str]] = defaultdict(set); queue: deque[tuple[str, str, str, str]] = deque()
    def internal(path: str) -> bool: return path.startswith(("pipeline/", "scripts/", "configs/", "runtime/", "src/", "matlab/", "third_party/"))
    def add_node(path: str, stage: str, classification: str, reason: str, discovery: str) -> None:
        if internal(path) and not (root / path).exists():
            raise FileNotFoundError(f"declared active repository dependency does not exist: {path}")
        record = nodes.get(path)
        if record is None:
            tokens = _historical(path)
            nodes[path] = {"path": path, "language": _language(path), "classification": classification, "stage_roles": {stage: classification}, "stages": [stage], "functional_area": f"stage_{stage}", "reason": reason, "evidence": [reason], "inputs": [], "outputs": [], "historical_name": bool(tokens), "historical_name_tokens": tokens, "generic_runtime_usage": bool(tokens) and classification not in {"LEGACY_UNUSED_BY_PIPELINE", "TEST_ONLY"}, "discovery_methods": [discovery], "source_sha256": _sha256_file(root / path)}
        else:
            if stage not in record["stages"]: record["stages"].append(stage)
            record["evidence"].append(reason)
            record["discovery_methods"].append(discovery)
            old = record["stage_roles"].get(stage)
            if old is None or ROLE_PRIORITY[classification] > ROLE_PRIORITY[old]:
                record["stage_roles"][stage] = classification
        stage_paths[stage].add(path)
    for stage, entry in STAGE_ENTRYPOINTS.items(): add_node(entry, stage, "ACTIVE_ENTRYPOINT", "Stage entrypoint", "ENTRYPOINT")
    for stage, src, dst, kind, condition, evidence, classification in DECLARED_EDGES:
        add_node(dst, stage, classification, evidence, "MANUAL_DYNAMIC_OVERRIDE")
        direct[(stage, dst)] = classification
        edges.append({"from": src, "to": dst, "edge_type": kind, "stage": stage, "condition": condition, "evidence": evidence})
    for stage, src, dst, condition, evidence in KNOWN_RUNTIME_ASSETS:
        add_node(dst, stage, "CONFIG_RUNTIME_ASSET", evidence, "CONFIG_REFERENCE")
        direct[(stage, dst)] = "CONFIG_RUNTIME_ASSET"
        edges.append({"from": src, "to": dst, "edge_type": "CONFIG_READ", "stage": stage, "condition": condition, "evidence": evidence})
    for path, reason in AUDITED_UNUSED.items(): add_node(path, "06", "TEST_ONLY" if "/tests/" in path else "LEGACY_UNUSED_BY_PIPELINE", reason, "AUDIT_EXCLUSION")
    queue.extend((path, stage, "DECLARED_RUNTIME", "declared runtime execution") for stage, _, path, _, _, _, _ in DECLARED_EDGES if Path(path).suffix in {".py", ".m", ".sh"})
    seen: set[tuple[str, str]] = set()
    while queue:
        raw, stage, _, reason = queue.popleft(); key = (raw, stage)
        if key in seen: continue
        seen.add(key); path = root / raw
        if not path.is_file(): continue
        if path.suffix == ".py":
            for target, evidence in _imports(root, path):
                dest = target.relative_to(root).as_posix()
                if dest in COMPATIBILITY:
                    classification = "COMPATIBILITY_IMPORT_ONLY"
                elif (stage, dest) in direct:
                    classification = direct[(stage, dest)]
                elif dest.startswith("third_party/") or "/third_party/" in dest:
                    classification = "THIRD_PARTY_ACTIVE"
                elif dest.startswith(("configs/", "runtime/")) or "/configs/" in dest:
                    classification = "CONFIG_RUNTIME_ASSET"
                else:
                    classification = "ACTIVE_TRANSITIVE"
                add_node(dest, stage, classification, COMPATIBILITY.get(dest, evidence), "PYTHON_IMPORT")
                edges.append({"from": raw, "to": dest, "edge_type": "PYTHON_IMPORT", "stage": stage, "condition": "import-time", "evidence": evidence})
                queue.append((dest, stage, "PYTHON_IMPORT", evidence))
        if path.suffix == ".m":
            for target, evidence in _matlab_calls(root, path):
                dest = target.relative_to(root).as_posix()
                add_node(dest, stage, "ACTIVE_TRANSITIVE", evidence, "MATLAB_CALL")
                edges.append({"from": raw, "to": dest, "edge_type": "MATLAB_LOCAL_FUNCTION", "stage": stage, "condition": "runtime function lookup", "evidence": evidence})
                queue.append((dest, stage, "MATLAB_FUNCTION", evidence))
        if path.suffix == ".sh":
            for target, evidence in _shell_modules(root, path):
                dest = target.relative_to(root).as_posix()
                classification = direct.get((stage, dest), "CONDITIONAL_ACTIVE")
                add_node(dest, stage, classification, evidence, "SHELL_MODULE_SCAN")
                edges.append({"from": raw, "to": dest, "edge_type": "SHELL_PYTHON_MODULE", "stage": stage, "condition": "shell command", "evidence": evidence})
                queue.append((dest, stage, "SHELL_PYTHON_MODULE", evidence))
    for path, reason in COMPATIBILITY.items():
        if (root / path).is_file(): add_node(path, "05", "COMPATIBILITY_IMPORT_ONLY", reason, "AUDIT_EXCLUSION")
    for item in EXTERNAL:
        for stage in item["required_by_stages"]:
            source = STAGE_ENTRYPOINTS[stage]
            edges.append({"from": source, "to": f"external:{item['id']}", "edge_type": "EXTERNAL_RUNTIME", "stage": stage, "condition": "runtime", "evidence": item.get("resolution", "runtime-provided model asset")})
    for stage, values in STAGE_IO.items():
        stage_paths[stage].update([STAGE_ENTRYPOINTS[stage]])
        for path in stage_paths[stage]:
            if path in nodes:
                nodes[path]["inputs"], nodes[path]["outputs"] = values if path == STAGE_ENTRYPOINTS[stage] else ([], [])
    unique = {(e["from"], e["to"], e["edge_type"], e["stage"], e["condition"], e["evidence"]): e for e in edges}
    for node in nodes.values():
        node["stages"].sort(); node["evidence"] = sorted(set(node["evidence"])); node["discovery_methods"] = sorted(set(node["discovery_methods"]))
        node["classification"] = max(node["stage_roles"].values(), key=lambda role: ROLE_PRIORITY[role])
    summary = dict(sorted(Counter(node["classification"] for node in nodes.values()).items()))
    return {"schema_version": "active-dependency-inventory/v3", "generation_provenance": _generation_provenance(root, nodes), "entrypoints": list(STAGE_ENTRYPOINTS.values()), "nodes": sorted(nodes.values(), key=lambda n: n["path"]), "edges": sorted(unique.values(), key=lambda e: (e["from"], e["to"], e["edge_type"], e["stage"])), "stages": {stage: {"entrypoint": STAGE_ENTRYPOINTS[stage], "paths": sorted(paths), "inputs": STAGE_IO[stage][0], "outputs": STAGE_IO[stage][1]} for stage, paths in sorted(stage_paths.items())}, "external_runtime_boundaries": EXTERNAL, "duplicate_or_proxy_modules": PROXIES, "classification_summary": summary, "audit_notes": {"formal_proposed_route": "Q002S640_B6_roi_same_anchor_mse; K8, anchor batch 640, 10 weights, 6400 residuals, A2000/B4000", "fit_first": "MATLAB SCMR-v5 native cache and L100/sparse reuse are active; Python fit_native and standardized routes are compatibility-only", "historical_names": "Historical filename tokens are metadata, never a legacy classification heuristic."}}


def markdown(inventory: dict[str, Any]) -> str:
    provenance = inventory["generation_provenance"]
    lines = ["# Active pipeline dependency trace", "", f"Generated deterministically from runtime-source fingerprint `{provenance['source_runtime_fingerprint_sha256']}` (generator `{provenance['generator_sha256']}`). Static only: no DICOM, MATLAB, NeSVoR, GPU, or model binary is executed.", ""]
    nodes = {node["path"]: node for node in inventory["nodes"]}
    for stage, details in inventory["stages"].items():
        paths = details["paths"]; by_class = defaultdict(list)
        for path in paths:
            if path in nodes and stage in nodes[path]["stage_roles"]:
                by_class[nodes[path]["stage_roles"][stage]].append(path)
        lines += [f"## Stage {stage}", "", f"Entrypoint: `{details['entrypoint']}`", "", "Inputs:"] + [f"- `{item}`" for item in details["inputs"]] + ["", "Outputs:"] + [f"- `{item}`" for item in details["outputs"]]
        for label, category in (("Direct execution chain", "ACTIVE_EXECUTED"), ("Conditional direct execution", "CONDITIONAL_ACTIVE"), ("Transitive implementation dependencies", "ACTIVE_TRANSITIVE"), ("Compatibility import closure", "COMPATIBILITY_IMPORT_ONLY"), ("Runtime configs/assets", "CONFIG_RUNTIME_ASSET"), ("Third-party runtime closure", "THIRD_PARTY_ACTIVE")):
            lines += ["", f"{label}:"] + [f"- `{path}`" for path in sorted(by_class.get(category, []))]
        boundaries = [item["id"] for item in inventory["external_runtime_boundaries"] if stage in item["required_by_stages"]]
        historical = [path for path in paths if path in nodes and nodes[path]["historical_name"]]
        lines += ["", "External runtime boundaries:"] + [f"- `{item}`" for item in boundaries] + ["", "Historical-name cleanup candidates:"] + [f"- `{path}`" for path in sorted(historical)] + [""]
    lines += ["## Historical-name cleanup candidates", "", "Historical `cyj11`, `final5`, `hhz`, and `legacy` names are recorded as metadata; reachability, not spelling, decides classification.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); parser.add_argument("--json", type=Path, required=True); parser.add_argument("--markdown", type=Path, required=True); parser.add_argument("--edges-csv", type=Path)
    args = parser.parse_args(); inventory = scan(args.root.resolve()); args.json.parent.mkdir(parents=True, exist_ok=True); args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"); args.markdown.write_text(markdown(inventory), encoding="utf-8")
    if args.edges_csv:
        with args.edges_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["from", "to", "edge_type", "stage", "condition", "evidence"]); writer.writeheader(); writer.writerows(inventory["edges"])
    print(f"ACTIVE DEPENDENCY INVENTORY: PASS ({len(inventory['nodes'])} nodes, {len(inventory['edges'])} edges)")


if __name__ == "__main__": main()
