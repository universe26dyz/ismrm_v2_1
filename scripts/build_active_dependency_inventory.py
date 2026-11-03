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
import json
import subprocess
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
    ("01", STAGE_ENTRYPOINTS["01"], "scripts/validate_subject_id.py", "SHELL_PYTHON_SCRIPT", "always", "run_stage invokes validator", "ACTIVE_EXECUTED"),
    ("01", STAGE_ENTRYPOINTS["01"], "scripts/build_subject_deployment_manifest.py", "SHELL_PYTHON_SCRIPT", "always", "run_stage builds manifest", "ACTIVE_EXECUTED"),
    ("02", STAGE_ENTRYPOINTS["02"], "common/trad/scripts/local_preprocess_all.m", "MATLAB_FUNCTION", "always", "MATLAB -batch local_preprocess_all", "ACTIVE_EXECUTED"),
    ("02", "common/trad/scripts/local_preprocess_all.m", "common/trad/scripts/local_preprocess_one.m", "MATLAB_FUNCTION", "per stack", "local_preprocess_all loop", "ACTIVE_EXECUTED"),
    ("02", "common/trad/scripts/local_preprocess_one.m", "common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/preprocess_stack_v1.m", "MATLAB_FUNCTION", "per stack", "explicit preprocess call", "ACTIVE_EXECUTED"),
    ("02", STAGE_ENTRYPOINTS["02"], "common/trad/scripts/validate_transferred_preprocessed.py", "SHELL_PYTHON_SCRIPT", "always", "post-transfer validator", "ACTIVE_EXECUTED"),
    ("03", STAGE_ENTRYPOINTS["03"], "common/trad/scripts/prepare_all_observations.py", "SHELL_PYTHON_SCRIPT", "always", "explicit --protocol", "ACTIVE_EXECUTED"),
    ("03", "common/trad/scripts/prepare_all_observations.py", "configs/protocol_scmr_v5_10_100.yaml", "CONFIG_READ", "always", "Stage 03 --protocol", "CONFIG_RUNTIME_ASSET"),
    ("04", STAGE_ENTRYPOINTS["04"], "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/build_sparse_profiles.sh", "SHELL_BASH", "always", "stage wrapper", "ACTIVE_EXECUTED"),
    ("04", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/build_sparse_profiles.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse.py", "SHELL_PYTHON_MODULE", "per selected sparse profile", "profile builder", "CONDITIONAL_ACTIVE"),
    ("05", STAGE_ENTRYPOINTS["05"], "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh", "SHELL_BASH", "always", "stage wrapper", "ACTIVE_EXECUTED"),
    ("05", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/scmr_v5_native_cache.py", "SHELL_PYTHON_MODULE", "always", "SCMR-v5 native cache", "ACTIVE_EXECUTED"),
    ("05", "common/paper_experiments/joint_vs_2dfit/phase2_clean/scmr_v5_native_cache.py", "common/trad/evaluation/scmr/matlab/build_scmr_v5_native_maps.m", "MATLAB_FUNCTION", "per stack", "MATLAB mapper subprocess", "ACTIVE_EXECUTED"),
    ("05", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/baseline.py", "SHELL_PYTHON_MODULE", "L100 and sparse cases", "baseline CLI", "CONDITIONAL_ACTIVE"),
    ("06", STAGE_ENTRYPOINTS["06"], "common/paper_experiments/joint_vs_2dfit/phase2_clean/proposed.py", "SHELL_PYTHON_MODULE", "always", "python -m proposed", "ACTIVE_EXECUTED"),
    ("06", "common/paper_experiments/joint_vs_2dfit/phase2_clean/proposed.py", "common/quality_experiments/Q002_B6_roi_same_anchor_mse/run_q002_reconstruction.py", "PYTHON_CALL", "formal Q002S640", "Q002 formal reconstruction call", "ACTIVE_EXECUTED"),
    ("06", "common/quality_experiments/Q002_B6_roi_same_anchor_mse/run_q002_reconstruction.py", "common/quality_experiments/Q002_B6_roi_same_anchor_mse/training.py", "PYTHON_CALL", "formal Q002S640", "Q002 training call", "ACTIVE_EXECUTED"),
    ("06", "common/quality_experiments/Q002_B6_roi_same_anchor_mse/training.py", "common/reconstruction_core/orchestration.py", "PYTHON_CALL", "formal Q002S640", "shared reconstruction engine", "ACTIVE_EXECUTED"),
    ("06", STAGE_ENTRYPOINTS["06"], "configs/protocol_scmr_v5_10_100.yaml", "CONFIG_READ", "always", "--protocol", "CONFIG_RUNTIME_ASSET"),
    ("06", STAGE_ENTRYPOINTS["06"], "configs/models_manifest.json", "CONFIG_READ", "always", "model provenance", "CONFIG_RUNTIME_ASSET"),
    ("06", STAGE_ENTRYPOINTS["06"], "runtime/runtime_contract.json", "CONFIG_READ", "always", "runtime contract", "CONFIG_RUNTIME_ASSET"),
    ("07", STAGE_ENTRYPOINTS["07"], "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh", "SHELL_BASH", "always", "stage wrapper", "ACTIVE_EXECUTED"),
    ("07", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh", "common/paper_experiments/joint_vs_2dfit/aha/cyj11_r3_direct_final_pose_bundle.py", "SHELL_PYTHON_MODULE", "per case", "Proposed final-pose bundle", "CONDITIONAL_ACTIVE"),
    ("07", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh", "common/paper_experiments/joint_vs_2dfit/aha/current_fit_first_final_pose_bundle.py", "SHELL_PYTHON_MODULE", "per case", "Fit-first final-pose bundle", "CONDITIONAL_ACTIVE"),
    ("07", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh", "common/paper_experiments/joint_vs_2dfit/aha/derive_final_pose_mapping.py", "SHELL_PYTHON_MODULE", "per case", "ROI mapping", "CONDITIONAL_ACTIVE"),
    ("07", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh", "common/paper_experiments/joint_vs_2dfit/aha/paired_final_pose_postprocess.py", "SHELL_PYTHON_MODULE", "per case", "paired AHA metrics", "CONDITIONAL_ACTIVE"),
    ("08", STAGE_ENTRYPOINTS["08"], "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_metrics_qc.sh", "SHELL_BASH", "always", "first evaluation wrapper", "ACTIVE_EXECUTED"),
    ("08", STAGE_ENTRYPOINTS["08"], "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_all_stack_roi_metrics.sh", "SHELL_BASH", "always", "second evaluation wrapper", "ACTIVE_EXECUTED"),
    ("08", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_metrics_qc.sh", "common/paper_experiments/joint_vs_2dfit/aha/common_native_metrics_qc.py", "SHELL_PYTHON_MODULE", "per case", "common-native metrics CLI", "CONDITIONAL_ACTIVE"),
    ("08", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_all_stack_roi_metrics.sh", "common/paper_experiments/joint_vs_2dfit/aha/common_native_all_stack_roi_metrics.py", "SHELL_PYTHON_MODULE", "per case", "all-stack ROI CLI", "CONDITIONAL_ACTIVE"),
    ("09", STAGE_ENTRYPOINTS["09"], "scripts/verify_project_integrity.py", "SHELL_PYTHON_SCRIPT", "always", "integrity checker", "ACTIVE_EXECUTED"),
    ("09", STAGE_ENTRYPOINTS["09"], "scripts/verify_no_legacy_code_path.py", "SHELL_PYTHON_SCRIPT", "always", "legacy checker", "ACTIVE_EXECUTED"),
    ("09", STAGE_ENTRYPOINTS["09"], "scripts/verify_runtime_contract.py", "SHELL_PYTHON_SCRIPT", "always", "runtime checker", "ACTIVE_EXECUTED"),
    ("09", STAGE_ENTRYPOINTS["09"], "scripts/verify_scientific_freeze.py", "SHELL_PYTHON_SCRIPT", "always", "freeze checker", "ACTIVE_EXECUTED"),
]
COMPATIBILITY = {
    "common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_intensity_preserving.py": "baseline imports it, but Stage 05 supplies SCMR-v5 prebuilt/reuse flags, never intensity-preserving flags",
    "common/paper_experiments/joint_vs_2dfit/phase2_clean/reuse_standardized_native_dictionary_maps.py": "baseline imports it, but Stage 05 never supplies standardized reuse flags",
    "common/paper_experiments/joint_vs_2dfit/phase2_clean/standardize_subject_slices.py": "standardization route is not selected by Stage 05",
}
AUDITED_UNUSED = {
    "common/quality_experiments/Q002_B6_roi_same_anchor_mse/run_q003_reconstruction.py": "Q003 experiment is not reached by formal Q002S640 command",
    "common/quality_experiments/Q002_B6_roi_same_anchor_mse/run_q004_s640_k1_reconstruction.py": "Q004/K1 experiment is not reached by formal Q002S640 command",
    "common/quality_experiments/Q002_B6_roi_same_anchor_mse/evaluate_q003_s640_10k.py": "10k evaluation variant is not reached by formal Q002S640 command",
    "common/quality_experiments/Q002_B6_roi_same_anchor_mse/tests/test_q005s640_k4.py": "Q005/K4 is test-only, not a Stage 06 runtime dependency",
    "common/quality_experiments/Q002_B6_roi_same_anchor_mse/tests/test_q002s640_k256.py": "K256 is test-only, not a Stage 06 runtime dependency",
}
EXTERNAL = [
    {"id": "matlab", "kind": "executable", "required_by_stages": ["02", "05"], "resolution": "MATLAB argument or PATH", "locally_available": False},
    {"id": "nesvor", "kind": "executable", "required_by_stages": ["05"], "resolution": "NESVOR environment variable or PATH", "locally_available": False},
    {"id": "cuda", "kind": "runtime_capability", "required_by_stages": ["06", "07"], "resolution": "cuda:0 requested by Stage commands", "locally_available": False},
    {"id": "decoder_scmr_v5_10_100.pth", "kind": "model_asset", "required_by_stages": ["06"], "expected_sha256": "338f2c0b8db90851f733ed31caa2c99914605fed3f7215ec2f69f19eba59583b", "repository_embedded": False},
    {"id": "b6_model.pt", "kind": "model_asset", "required_by_stages": ["06"], "expected_sha256": "15981fbe93d2c70e14755ea756a02856c9383481e7d92d239619268a628a29b5", "repository_embedded": False},
]
PROXIES = [
    {"wrapper": "common/mlp/modules/module_07_objective_training/mlp_trainer.py", "implementation": "common/reconstruction_core/orchestration.py", "relationship": "FACADE", "evidence": "imports and re-exports the sole reconstruction engine", "active_caller": "MLP compatibility route", "cleanup_candidate": True},
    {"wrapper": "common/mlp/modules/module_04_quantitative_inr/quantitative_inr.py", "implementation": "common/trad/modules/module_04_quantitative_inr/quantitative_inr.py", "relationship": "DUPLICATE_IMPLEMENTATION", "evidence": "parallel source implementations with package-local imports", "active_caller": "Q002 uses MLP copy", "cleanup_candidate": True},
    {"wrapper": "common/mlp/modules/module_05_signal_decoder/frozen_decoder.py", "implementation": "common/trad/modules/module_05_signal_decoder/decoder_factory.py", "relationship": "INDEPENDENT_IMPLEMENTATION", "evidence": "frozen MLP decoder class is not a forwarding proxy", "active_caller": "Q002 frozen decoder", "cleanup_candidate": False},
    {"wrapper": "common/mlp/modules/module_06_rigid_psf/rigid_psf_forward.py", "implementation": "common/trad/modules/module_06_rigid_psf/rigid_psf_forward.py", "relationship": "DUPLICATE_IMPLEMENTATION", "evidence": "parallel source implementations with package-local imports", "active_caller": "MLP/TRAD compatibility packages", "cleanup_candidate": True},
]


def _language(path: str) -> str:
    return {".py": "python", ".sh": "shell", ".m": "matlab", ".json": "json", ".yaml": "yaml"}.get(Path(path).suffix, "other")


def _resolve_import(root: Path, module: str, source: Path, level: int) -> Path | None:
    base = source.parent
    for _ in range(max(level - 1, 0)):
        base = base.parent
    stems = [base.joinpath(*module.split(".")), root / "common" / Path(*module.split(".")), root / "scripts" / Path(*module.split("."))]
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


def _historical(path: str) -> list[str]:
    return [token for token in ("cyj11", "final5", "hhz", "legacy") if token in path.lower()]


def scan(root: Path) -> dict[str, Any]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    nodes: dict[str, dict[str, Any]] = {}; edges: list[dict[str, Any]] = []; direct: dict[str, str] = {}
    stage_paths: dict[str, set[str]] = defaultdict(set); queue: deque[tuple[str, str, str, str]] = deque()
    def internal(path: str) -> bool: return path.startswith(("common/", "pipeline/", "scripts/", "configs/", "runtime/"))
    def add_node(path: str, stage: str, classification: str, reason: str, discovery: str) -> None:
        if internal(path) and not (root / path).exists():
            raise FileNotFoundError(f"declared active repository dependency does not exist: {path}")
        record = nodes.get(path)
        if record is None:
            tokens = _historical(path)
            nodes[path] = {"path": path, "language": _language(path), "classification": classification, "stages": [stage], "functional_area": f"stage_{stage}", "reason": reason, "evidence": [reason], "inputs": [], "outputs": [], "historical_name": bool(tokens), "historical_name_tokens": tokens, "generic_runtime_usage": bool(tokens) and classification not in {"LEGACY_UNUSED_BY_PIPELINE", "TEST_ONLY"}, "discovery_method": discovery, "git_sha": sha}
        else:
            if stage not in record["stages"]: record["stages"].append(stage)
            record["evidence"].append(reason)
        stage_paths[stage].add(path)
    for stage, entry in STAGE_ENTRYPOINTS.items(): add_node(entry, stage, "ACTIVE_ENTRYPOINT", "Stage entrypoint", "ENTRYPOINT")
    for stage, src, dst, kind, condition, evidence, classification in DECLARED_EDGES:
        add_node(dst, stage, classification, evidence, "MANUAL_DYNAMIC_OVERRIDE")
        direct[dst] = classification
        edges.append({"from": src, "to": dst, "edge_type": kind, "stage": stage, "condition": condition, "evidence": evidence})
    for path, reason in AUDITED_UNUSED.items(): add_node(path, "06", "TEST_ONLY" if "/tests/" in path else "LEGACY_UNUSED_BY_PIPELINE", reason, "AUDIT_EXCLUSION")
    queue.extend((path, stage, "PYTHON_IMPORT", "declared Python execution") for stage, _, path, kind, _, _, _ in DECLARED_EDGES if kind.startswith("SHELL_PYTHON") or kind == "PYTHON_CALL")
    seen: set[tuple[str, str]] = set()
    while queue:
        raw, stage, _, reason = queue.popleft(); key = (raw, stage)
        if key in seen: continue
        seen.add(key); path = root / raw
        if not path.is_file() or path.suffix != ".py": continue
        for target, evidence in _imports(root, path):
            dest = target.relative_to(root).as_posix()
            if dest in COMPATIBILITY:
                classification = "COMPATIBILITY_IMPORT_ONLY"
            elif dest in direct:
                classification = direct[dest]
            elif "/third_party/" in dest:
                classification = "THIRD_PARTY_ACTIVE"
            elif dest.startswith(("configs/", "runtime/")) or "/configs/" in dest:
                classification = "CONFIG_RUNTIME_ASSET"
            else:
                classification = "ACTIVE_TRANSITIVE"
            add_node(dest, stage, classification, COMPATIBILITY.get(dest, evidence), "PYTHON_IMPORT")
            edges.append({"from": raw, "to": dest, "edge_type": "PYTHON_IMPORT", "stage": stage, "condition": "import-time", "evidence": evidence})
            queue.append((dest, stage, "PYTHON_IMPORT", evidence))
    for path, reason in COMPATIBILITY.items():
        if (root / path).is_file(): add_node(path, "05", "COMPATIBILITY_IMPORT_ONLY", reason, "AUDIT_EXCLUSION")
    for item in EXTERNAL:
        stage = item["required_by_stages"][0]; source = STAGE_ENTRYPOINTS[stage]
        edges.append({"from": source, "to": f"external:{item['id']}", "edge_type": "EXTERNAL_RUNTIME", "stage": stage, "condition": "runtime", "evidence": item.get("resolution", "runtime-provided model asset")})
    for stage, values in STAGE_IO.items():
        stage_paths[stage].update([STAGE_ENTRYPOINTS[stage]])
        for path in stage_paths[stage]:
            if path in nodes:
                nodes[path]["inputs"], nodes[path]["outputs"] = values if path == STAGE_ENTRYPOINTS[stage] else ([], [])
    unique = {(e["from"], e["to"], e["edge_type"], e["stage"], e["condition"], e["evidence"]): e for e in edges}
    for node in nodes.values(): node["stages"].sort(); node["evidence"] = sorted(set(node["evidence"]))
    summary = dict(sorted(Counter(node["classification"] for node in nodes.values()).items()))
    return {"schema_version": "active-dependency-inventory/v2", "generated_from_git_sha": sha, "entrypoints": list(STAGE_ENTRYPOINTS.values()), "nodes": sorted(nodes.values(), key=lambda n: n["path"]), "edges": sorted(unique.values(), key=lambda e: (e["from"], e["to"], e["edge_type"], e["stage"])), "stages": {stage: {"entrypoint": STAGE_ENTRYPOINTS[stage], "paths": sorted(paths), "inputs": STAGE_IO[stage][0], "outputs": STAGE_IO[stage][1]} for stage, paths in sorted(stage_paths.items())}, "external_runtime_boundaries": EXTERNAL, "duplicate_or_proxy_modules": PROXIES, "classification_summary": summary, "audit_notes": {"formal_proposed_route": "Q002S640_B6_roi_same_anchor_mse; K8, anchor batch 640, 10 weights, 6400 residuals, A2000/B4000", "fit_first": "MATLAB SCMR-v5 native cache and L100/sparse reuse are active; Python fit_native and standardized routes are compatibility-only", "historical_names": "Historical filename tokens are metadata, never a legacy classification heuristic."}}


def markdown(inventory: dict[str, Any]) -> str:
    lines = ["# Active pipeline dependency trace", "", f"Generated deterministically from `{inventory['generated_from_git_sha']}`. Static only: no DICOM, MATLAB, NeSVoR, GPU, or model binary is executed.", ""]
    nodes = {node["path"]: node for node in inventory["nodes"]}
    for stage, details in inventory["stages"].items():
        paths = details["paths"]; by_class = defaultdict(list)
        for path in paths:
            if path in nodes: by_class[nodes[path]["classification"]].append(path)
        lines += [f"## Stage {stage}", "", f"Entrypoint: `{details['entrypoint']}`", "", "Inputs:"] + [f"- `{item}`" for item in details["inputs"]] + ["", "Outputs:"] + [f"- `{item}`" for item in details["outputs"]]
        for label, category in (("Direct execution chain", "ACTIVE_EXECUTED"), ("Transitive implementation dependencies", "ACTIVE_TRANSITIVE"), ("Conditional branches", "CONDITIONAL_ACTIVE"), ("Compatibility-only dependencies", "COMPATIBILITY_IMPORT_ONLY"), ("Configs/assets", "CONFIG_RUNTIME_ASSET")):
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
