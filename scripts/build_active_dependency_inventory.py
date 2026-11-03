#!/usr/bin/env python3
"""Build a deterministic, data-free trace of the 01--09 active pipeline."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


STAGES = {
    "01": ["pipeline/01_RAW_DICOM/run_stage.sh", "scripts/validate_active_subject.py", "common/scripts/extract_subject_deployment_manifest.py", "configs/subject_stack_manifest.example.json"],
    "02": ["pipeline/02_PREPROCESS_MATLAB/run_stage.sh", "common/trad/scripts/local_preprocess_all.m", "common/trad/scripts/local_preprocess_one.m", "common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/preprocess_stack_v1.m", "common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/preprocess_options_v1.m", "common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/mppca_denoise_multimap_stack.m", "common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/deformableReg2Dmind_adaptive.m", "common/trad/scripts/validate_transferred_preprocessed.py"],
    "03": ["pipeline/03_PREPARE_OBSERVATIONS/run_stage.sh", "common/trad/scripts/prepare_all_observations.py", "common/scripts/deployment_manifest.py", "common/trad/modules/module_02_data_bridge/prepare_observations.py", "common/trad/modules/module_02_data_bridge/mat_v73.py", "common/trad/modules/module_02_data_bridge/build_manifest.py", "common/trad/modules/module_02_data_bridge/geometry.py", "common/trad/modules/module_02_data_bridge/timing.py", "configs/protocol_scmr_v5_10_100.yaml"],
    "04": ["pipeline/04_SPARSE_PROFILES/run_stage.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/build_sparse_profiles.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse.py", "common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse_profile_manifest.py", "common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles"],
    "05": ["pipeline/05_SCMR_V5_FIT_FIRST/run_stage.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/scmr_v5_native_cache.py", "common/trad/evaluation/scmr/matlab/build_scmr_v5_native_maps.m", "common/paper_experiments/joint_vs_2dfit/phase2_clean/baseline.py", "common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first.py", "common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_denormalize.py"],
    "06": ["pipeline/06_PROPOSED/run_stage.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/proposed.py", "common/quality_experiments/Q002_B6_roi_same_anchor_mse/run_q002_reconstruction.py", "common/quality_experiments/Q002_B6_roi_same_anchor_mse/joint_dataset.py", "common/quality_experiments/Q002_B6_roi_same_anchor_mse/training.py", "common/reconstruction_core/orchestration.py", "common/mlp/modules/module_04_quantitative_inr/quantitative_inr.py", "common/mlp/modules/module_05_signal_decoder/frozen_decoder.py", "common/trad/modules/module_06_rigid_psf/rigid_psf_forward.py", "runtime/runtime_contract.json", "configs/models_manifest.json", "runtime/model_assets"],
    "07": ["pipeline/07_FINAL_POSE/run_stage.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh", "common/paper_experiments/joint_vs_2dfit/aha/cyj11_r3_direct_final_pose_bundle.py", "common/paper_experiments/joint_vs_2dfit/aha/current_fit_first_final_pose_bundle.py", "common/paper_experiments/joint_vs_2dfit/aha/derive_final_pose_mapping.py", "common/paper_experiments/joint_vs_2dfit/aha/paired_final_pose_postprocess.py", "common/paper_experiments/joint_vs_2dfit/aha/paired_aha_sd.py", "common/paper_experiments/joint_vs_2dfit/aha/paired_final_pose_through_plane.py"],
    "08": ["pipeline/08_EVALUATION_AND_VISUALIZATION/run_stage.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_metrics_qc.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_all_stack_roi_metrics.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/common_native_metrics_qc.py", "common/paper_experiments/joint_vs_2dfit/phase2_clean/common_native_all_stack_roi_metrics.py", "common/paper_experiments/joint_vs_2dfit/phase2_clean/common_native.py"],
    "09": ["pipeline/09_INTEGRITY_AND_REPORTING/run_stage.sh", "scripts/verify_project_integrity.py", "scripts/verify_no_legacy_code_path.py", "scripts/verify_runtime_contract.py", "scripts/verify_scientific_freeze.py", "configs/scientific_freeze_contract.json"],
}

PROXIES = [
    {"wrapper": "common/mlp/modules/module_04_quantitative_inr/quantitative_inr.py", "implementation": "common/trad/modules/module_04_quantitative_inr/quantitative_inr.py", "active_caller": "Q002 training", "future_cleanup_recommendation": "retain facade until package consolidation"},
    {"wrapper": "common/mlp/modules/module_05_signal_decoder/frozen_decoder.py", "implementation": "common/trad/modules/module_05_signal_decoder", "active_caller": "Q002 training", "future_cleanup_recommendation": "record dynamic/proxy boundary before renaming"},
]


def language(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {".py": "python", ".sh": "shell", ".m": "matlab", ".json": "json", ".yaml": "yaml", ".yml": "yaml"}.get(suffix, "other")


def classify(path: str, stage: str, entry: bool = False) -> str:
    if entry:
        return "ACTIVE_ENTRYPOINT"
    if path.startswith("runtime/") or path.startswith("configs/") or "/configs/" in path:
        return "CONFIG_RUNTIME_ASSET"
    if "/third_party/" in path:
        return "THIRD_PARTY_ACTIVE"
    if any(token in path for token in ("standardize_", "reuse_standardized", "fit_first_intensity_preserving")):
        return "COMPATIBILITY_IMPORT_ONLY"
    if any(token in path.lower() for token in ("legacy", "q003", "q004", "q005", "cyjj11", "final5")):
        return "LEGACY_UNUSED_BY_PIPELINE"
    return "ACTIVE_EXECUTED" if stage in {"01", "02", "03", "04", "05", "06", "07", "08", "09"} else "ACTIVE_TRANSITIVE"


def resolve_import(root: Path, module: str, source: Path, level: int = 0) -> Path | None:
    base = source.parent
    if level:
        for _ in range(level - 1):
            base = base.parent
    candidate = base.joinpath(*module.split(".")) if module else base
    for path in (candidate.with_suffix(".py"), candidate / "__init__.py", root / "common" / Path(*module.split("."))).__iter__():
        if path.is_file():
            return path
    direct = root / "common" / Path(*module.split("."))
    for path in (direct.with_suffix(".py"), direct / "__init__.py"):
        if path.is_file():
            return path
    return None


def python_imports(root: Path, path: Path) -> list[tuple[Path, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    found: list[tuple[Path, str]] = []
    for item in ast.walk(tree):
        module = ""
        level = 0
        if isinstance(item, ast.Import):
            for name in item.names:
                target = resolve_import(root, name.name, path)
                if target:
                    found.append((target, f"import {name.name}"))
        elif isinstance(item, ast.ImportFrom):
            module, level = item.module or "", item.level
            target = resolve_import(root, module, path, level)
            if target:
                found.append((target, f"from {'.' * level}{module} import"))
    return found


def scan(root: Path) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    stage_paths: dict[str, list[str]] = defaultdict(list)
    queue: deque[tuple[str, str, bool, str]] = deque()
    for stage, paths in STAGES.items():
        for raw in paths:
            if (root / raw).is_dir():
                for child in sorted((root / raw).rglob("*")):
                    if child.is_file() and child.suffix in {".json", ".py", ".sh"}:
                        queue.append((child.relative_to(root).as_posix(), stage, False, "declared runtime directory"))
            else:
                queue.append((raw, stage, raw.endswith("run_stage.sh"), "declared active stage dependency"))
    while queue:
        raw, stage, is_entry, reason = queue.popleft()
        path = root / raw
        if raw not in nodes:
            node_classification = "UNRESOLVED_DYNAMIC" if not path.exists() else classify(raw, stage, is_entry)
            nodes[raw] = {"path": raw, "language": language(raw), "classification": node_classification, "stages": [stage], "functional_area": f"stage_{stage}", "reason": reason, "evidence": [reason], "inputs": [], "outputs": []}
        elif stage not in nodes[raw]["stages"]:
            nodes[raw]["stages"].append(stage)
        stage_paths[stage].append(raw)
        if not path.is_file() or path.suffix != ".py":
            continue
        for target, evidence in python_imports(root, path):
            rel = target.relative_to(root).as_posix()
            edges.append({"from": raw, "to": rel, "edge_type": "PYTHON_IMPORT", "stage": stage, "condition": "import-time", "evidence": evidence})
            if rel not in nodes:
                queue.append((rel, stage, False, evidence))
    # Source-observed non-AST edges are intentionally explicit: shell variables, MATLAB addpath,
    # subprocess modules, and runtime binaries cannot be recovered faithfully by AST alone.
    observed = [
        ("pipeline/02_PREPROCESS_MATLAB/run_stage.sh", "common/trad/scripts/local_preprocess_all.m", "MATLAB_FUNCTION", "MATLAB -batch addpath/local_preprocess_all"),
        ("common/trad/scripts/local_preprocess_all.m", "common/trad/scripts/local_preprocess_one.m", "MATLAB_FUNCTION", "local_preprocess_all loop"),
        ("common/trad/scripts/local_preprocess_one.m", "common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/preprocess_stack_v1.m", "MATLAB_FUNCTION", "addpath preprocessing_v1; preprocess_stack_v1"),
        ("pipeline/04_SPARSE_PROFILES/run_stage.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/build_sparse_profiles.sh", "SHELL_BASH", "REPO-bound bash invocation"),
        ("pipeline/05_SCMR_V5_FIT_FIRST/run_stage.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh", "SHELL_BASH", "REPO-bound bash invocation"),
        ("common/paper_experiments/joint_vs_2dfit/phase2_clean/scmr_v5_native_cache.py", "common/trad/evaluation/scmr/matlab/build_scmr_v5_native_maps.m", "MATLAB_FUNCTION", "subprocess MATLAB build_scmr_v5_native_maps"),
        ("pipeline/06_PROPOSED/run_stage.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/proposed.py", "PYTHON_MODULE", "python -m proposed"),
        ("pipeline/07_FINAL_POSE/run_stage.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh", "SHELL_BASH", "REPO-bound bash invocation"),
        ("pipeline/08_EVALUATION_AND_VISUALIZATION/run_stage.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_metrics_qc.sh", "SHELL_BASH", "first evaluation wrapper"),
        ("pipeline/08_EVALUATION_AND_VISUALIZATION/run_stage.sh", "common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_all_stack_roi_metrics.sh", "SHELL_BASH", "second evaluation wrapper"),
    ]
    for frm, to, kind, evidence in observed:
        if frm in nodes and (root / to).exists():
            stage = nodes[frm]["stages"][0]
            if to not in nodes:
                nodes[to] = {"path": to, "language": language(to), "classification": classify(to, stage), "stages": [stage], "functional_area": f"stage_{stage}", "reason": evidence, "evidence": [evidence], "inputs": [], "outputs": []}
            edges.append({"from": frm, "to": to, "edge_type": kind, "stage": stage, "condition": "active invocation", "evidence": evidence})
    protocol_hits = []
    for node in nodes.values():
        path = root / node["path"]
        if path.is_file() and path.suffix in {".py", ".m", ".sh", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"50\s*[, ]\s*150|hhz_v1", text, re.I):
                protocol_hits.append({"path": node["path"], "classification": node["classification"], "evidence": "legacy [50,150]/hhz_v1 reference"})
    summary: dict[str, int] = defaultdict(int)
    for node in nodes.values():
        node["stages"].sort()
        summary[node["classification"]] += 1
    unique_edges = {
        (edge["from"], edge["to"], edge["edge_type"], edge["stage"], edge["condition"], edge["evidence"]): edge
        for edge in edges
    }
    return {"schema_version": "active-dependency-inventory/v1", "generated_from_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "entrypoints": [f"pipeline/{n}/run_stage.sh" for n in ("01_RAW_DICOM", "02_PREPROCESS_MATLAB", "03_PREPARE_OBSERVATIONS", "04_SPARSE_PROFILES", "05_SCMR_V5_FIT_FIRST", "06_PROPOSED", "07_FINAL_POSE", "08_EVALUATION_AND_VISUALIZATION", "09_INTEGRITY_AND_REPORTING")], "nodes": sorted(nodes.values(), key=lambda item: item["path"]), "edges": sorted(unique_edges.values(), key=lambda item: (item["from"], item["to"], item["edge_type"], item["stage"])), "stages": {stage: sorted(set(paths)) for stage, paths in sorted(stage_paths.items())}, "unresolved_dynamic_dependencies": [{"path": "external:matlab", "reason": "server-only MATLAB execution"}, {"path": "external:nesvor", "reason": "server-only NeSVoR executable"}, {"path": "runtime/model_assets/*.pth", "reason": "binary assets intentionally absent locally"}], "classification_summary": dict(sorted(summary.items())), "duplicate_or_proxy_modules": PROXIES, "legacy_protocol_references": protocol_hits, "subject_gating_references": [item["path"] for item in nodes.values() if "validate_active_subject" in item["path"]]}


def markdown(inventory: dict[str, Any]) -> str:
    lines = ["# Active pipeline dependency trace", "", f"Generated deterministically from `{inventory['generated_from_git_sha']}`. This is a static, data-free trace; server MATLAB, model binaries, and NeSVoR are recorded but not executed.", ""]
    for stage, paths in inventory["stages"].items():
        lines += [f"## Stage {stage}", "", "Direct and discovered active paths:", ""]
        lines += [f"- `{path}`" for path in paths]
        lines += [""]
    lines += ["## Compatibility / future cleanup", ""]
    for proxy in inventory["duplicate_or_proxy_modules"]:
        lines.append(f"- `{proxy['wrapper']}` → `{proxy['implementation']}`; {proxy['future_cleanup_recommendation']}.")
    lines += ["", "Historical names such as `hhz` and `cyj11` remain generic-runtime wrappers where reached by stages 07/08; they are rename candidates for a future clean repository, not evidence of a subject-specific admission gate.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--edges-csv", type=Path)
    args = parser.parse_args()
    inventory = scan(args.root.resolve())
    args.json.parent.mkdir(parents=True, exist_ok=True); args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown(inventory), encoding="utf-8")
    if args.edges_csv:
        with args.edges_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["from", "to", "edge_type", "stage", "condition", "evidence"]); writer.writeheader(); writer.writerows(inventory["edges"])
    print(f"ACTIVE DEPENDENCY INVENTORY: PASS ({len(inventory['nodes'])} nodes, {len(inventory['edges'])} edges)")


if __name__ == "__main__":
    main()
