"""Stage the eight-case reliability inputs without copying large observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ismrm_v2.sampling.sparse_profile_manifest import LEGACY_SCHEMA_VERSION, validate_sparse_profile_manifest

STACKS = ("sax", "2ch", "4ch")


def digest(path: Path, cache: dict[Path, str] | None = None) -> str:
    path = path.resolve()
    if cache is not None and path in cache:
        return cache[path]
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    value = h.hexdigest()
    if cache is not None:
        cache[path] = value
    return value


def hardlink(source: Path, target: Path) -> None:
    if source.is_symlink():
        raise RuntimeError(f"refusing symlink observation input: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    try:
        os.link(source, target)
    except OSError as exc:
        raise RuntimeError(f"refusing to copy observation; hardlink failed: {source} -> {target}") from exc


def locate(root: Path, subject: str, case_id: str, filename: str) -> Path:
    candidates = [root / subject / case_id / filename, root / subject / filename]
    candidates += sorted(root.glob(f"**/{subject}/{case_id}/{filename}"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot locate {filename} for {subject}/{case_id} below {root}")


def locate_manifest(root: Path, subject: str, case_id: str) -> Path:
    return locate(root, subject, case_id, "sparse_manifest.json")


def resolve_manifest(root: Path, subject: str, item: dict[str, Any]) -> tuple[Path, str]:
    if isinstance(item.get("case_id"), str):
        path = locate_manifest(root, subject, item["case_id"])
        return path, item["case_id"]
    profile_id, fold_id = item.get("profile_id"), item.get("fold_id", 0)
    if not isinstance(profile_id, str) or not isinstance(fold_id, int):
        raise ValueError("case selector requires case_id or profile_id plus integer fold_id")
    matches = []
    for path in sorted(root.glob(f"**/{subject}/**/sparse_manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("profile_id") != profile_id:
            continue
        payload_fold = payload.get("fold_id", payload.get("phase"))
        # A single-fold profile is represented canonically with case_id equal
        # to profile_id and fold_id/phase set to null.  The case-plan selector
        # uses fold_id=0 uniformly, so treat that query as the sole fold.
        if payload_fold is None and payload.get("fold_count") == 1 and fold_id == 0:
            matches.append((path.resolve(), str(payload.get("case_id") or payload.get("condition"))))
            continue
        try:
            payload_fold = int(payload_fold)
        except (TypeError, ValueError):
            continue
        if payload_fold == fold_id:
            matches.append((path.resolve(), str(payload.get("case_id") or payload.get("condition"))))
    if len(matches) != 1:
        raise ValueError(f"case selector {subject}/{profile_id}/fold{fold_id} matched {len(matches)} manifests")
    return matches[0]


def locate_model_checkpoint(root: Path, subject: str, case_id: str) -> Path:
    """Locate the one frozen Proposed authority containing geometry and model state."""
    candidates = [
        root / subject / "reconstruction" / "proposed" / subject / case_id / "model.pt",
        root / "runs" / "frozen_baseline" / subject / "reconstruction" / "proposed" / subject / case_id / "model.pt",
        root / subject / case_id / "model.pt",
    ]
    candidates += sorted(root.glob(f"**/{subject}/reconstruction/proposed/{subject}/{case_id}/model.pt"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot locate frozen Proposed model.pt for {subject}/{case_id} below {root}")


def _unique_candidates(candidates: list[tuple[int, str, Path]], *, expected_sha256: str, stack: str, role: str, digest_cache: dict[Path, str] | None = None) -> tuple[Path, dict[str, Any]]:
    """Select one role-aware candidate; lower-priority duplicates are provenance only."""
    seen: set[Path] = set()
    matches: list[tuple[int, str, Path]] = []
    for priority, rule, candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file() or candidate.is_symlink():
            continue
        seen.add(candidate)
        if digest(candidate, digest_cache) == expected_sha256:
            matches.append((priority, rule, candidate))
    if not matches:
        raise FileNotFoundError(f"no project-local {role} {stack} observation matches recorded SHA256 {expected_sha256}")
    best_priority = min(item[0] for item in matches)
    best = [item for item in matches if item[0] == best_priority]
    if len(best) != 1:
        raise RuntimeError(f"ambiguous same-priority {role} {stack} observations match recorded SHA256: {[str(item[2]) for item in best]}")
    selected = best[0]
    duplicates = [str(item[2]) for item in matches if item[2] != selected[2]]
    return selected[2], {"role": role, "selected_path": str(selected[2]), "selected_sha256": expected_sha256, "resolution_rule": selected[1], "duplicate_same_sha_candidates": duplicates}


def locate_source_observation(roots: list[Path], *, subject: str, stack: str, expected_sha256: str, digest_cache: dict[Path, str] | None = None) -> tuple[Path, dict[str, Any]]:
    """Resolve source observations from canonical prepared subject layouts only."""
    candidates: list[tuple[int, str, Path]] = []
    for root in roots:
        candidates.extend([
            (0, "canonical_subject_prepared_subject_stack", root / subject / "prepared" / subject / stack / "observations.npz"),
            (0, "canonical_subject_prepared_stack", root / subject / "prepared" / stack / "observations.npz"),
            (1, "canonical_recursive_prepared_subject_stack", root / "**" / subject / "prepared" / subject / stack / "observations.npz"),
            (1, "canonical_recursive_prepared_stack", root / "**" / subject / "prepared" / stack / "observations.npz"),
        ])
        candidates.extend((1, "canonical_recursive_prepared_subject_stack", path) for path in root.glob(f"**/{subject}/prepared/{subject}/{stack}/observations.npz"))
        candidates.extend((1, "canonical_recursive_prepared_stack", path) for path in root.glob(f"**/{subject}/prepared/{stack}/observations.npz"))
        candidates.extend((2, "lower_priority_sparse_copy", path) for path in root.glob(f"**/{subject}/sparse_inputs/**/{stack}/observations.npz"))
    return _unique_candidates(candidates, expected_sha256=expected_sha256, stack=stack, role="source", digest_cache=digest_cache)


def locate_derived_observation(roots: list[Path], *, subject: str, case_id: str, stack: str, expected_sha256: str, digest_cache: dict[Path, str] | None = None) -> tuple[Path, dict[str, Any]]:
    """Resolve derived observations from the exact resolved sparse case only."""
    candidates: list[tuple[int, str, Path]] = []
    for root in roots:
        candidates.extend([
            (0, "canonical_subject_sparse_inputs_resolved_case_stack", root / subject / "sparse_inputs" / subject / case_id / stack / "observations.npz"),
            (0, "canonical_sparse_inputs_resolved_case_stack", root / "sparse_inputs" / subject / case_id / stack / "observations.npz"),
            (0, "canonical_resolved_case_stack", root / subject / case_id / stack / "observations.npz"),
        ])
        candidates.extend((1, "recursive_resolved_case_stack", path) for path in root.glob(f"**/{subject}/sparse_inputs/{subject}/{case_id}/{stack}/observations.npz"))
        candidates.extend((1, "recursive_resolved_case_stack", path) for path in root.glob(f"**/{subject}/{case_id}/{stack}/observations.npz"))
    return _unique_candidates(candidates, expected_sha256=expected_sha256, stack=stack, role="derived", digest_cache=digest_cache)


def stage_case(item: dict[str, Any], *, root: Path, manifest_root: Path, geometry_root: Path, reconstruction_root: Path, observation_roots: list[Path], digest_cache: dict[Path, str] | None = None) -> dict[str, Any]:
    subject = item["subject"]
    source_manifest, case_id = resolve_manifest(manifest_root, subject, item)
    destination = (root / item["input_dir"].format(**(item | {"case_id": case_id}))).resolve()
    if not destination.is_relative_to(root.resolve()) or destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"staging destination must be new and project-local: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if source_manifest.is_symlink():
        raise RuntimeError(f"refusing symlink manifest input: {source_manifest}")
    source_sha = digest(source_manifest, digest_cache)
    # Historical manifests are provenance records, not runtime inputs.  Read
    # them before validation because their absolute paths may be retired.
    raw = json.loads(source_manifest.read_text(encoding="utf-8"))
    schema = raw.get("schema_version")
    condition = str(raw.get("condition") or raw.get("case_id") or case_id).upper()
    use_source = schema == LEGACY_SCHEMA_VERSION and condition.startswith("L100")
    prepared = destination / "prepared"
    prepared.mkdir()
    source_paths: dict[str, Path] = {}
    source_resolution: dict[str, dict[str, Any]] = {}
    derived_paths: dict[str, Path] = {}
    derived_resolution: dict[str, dict[str, Any]] = {}
    active_records = []
    for stack in STACKS:
        spec = raw["stacks"][stack]
        source_record = spec.get("source_observation")
        if not isinstance(source_record, dict) or not isinstance(source_record.get("sha256"), str):
            raise ValueError(f"{source_manifest}: {stack} lacks source SHA256 provenance")
        source_paths[stack], source_resolution[stack] = locate_source_observation(observation_roots, subject=subject, stack=stack, expected_sha256=source_record["sha256"], digest_cache=digest_cache)
        present = bool(spec.get("retained_original_group_ids", [])) if schema == LEGACY_SCHEMA_VERSION else bool(spec.get("training_stack_present"))
        if present and not use_source:
            record = spec.get("derived_observation")
            if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
                raise ValueError(f"{source_manifest}: {stack} lacks derived SHA256 provenance")
            derived_paths[stack], derived_resolution[stack] = locate_derived_observation(observation_roots, subject=subject, case_id=case_id, stack=stack, expected_sha256=record["sha256"], digest_cache=digest_cache)
    source_roots = {path.parent.parent for path in source_paths.values()}
    if len(source_roots) != 1 or any(path != next(iter(source_roots)) / stack / "observations.npz" for stack, path in source_paths.items()):
        raise ValueError("current source observations do not share the canonical subject prepared root")
    source_root = next(iter(source_roots))
    raw["source_prepared_root"] = str(source_root)
    prepared_root = source_root if use_source else prepared
    raw["prepared_root"] = str(prepared_root)
    for stack in STACKS:
        spec = raw["stacks"][stack]
        spec["source_observation"] = {**spec["source_observation"], "path": str(source_paths[stack]), "sha256": digest(source_paths[stack], digest_cache)}
        spec["source_observation_resolution"] = source_resolution[stack]
        if stack in derived_paths:
            target = prepared / stack / "observations.npz"
            hardlink(derived_paths[stack], target)
            spec["derived_observation"] = {**spec["derived_observation"], "path": str(target), "sha256": digest(target, digest_cache)}
            spec["derived_observation_resolution"] = {**derived_resolution[stack], "selected_path": str(target), "selected_sha256": digest(target, digest_cache)}
            active_records.append({"stack": stack, "mode": "derived", "role": "derived", "source": str(derived_paths[stack]), "path": str(target), "sha256": digest(target, digest_cache), "resolution": spec["derived_observation_resolution"], "retained_original_group_ids": spec.get("retained_original_group_ids", []), "derived_to_original_group_idx": spec.get("derived_to_original_group_idx")})
        elif use_source:
            active_records.append({"stack": stack, "mode": "source", "role": "source", "path": str(source_paths[stack]), "sha256": digest(source_paths[stack], digest_cache), "resolution": source_resolution[stack]})
    working_manifest = destination / "sparse_manifest.json" if use_source else prepared / "sparse_manifest.json"
    working_manifest.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_sparse_profile_manifest(working_manifest, subject=subject)
    if digest(source_manifest, digest_cache) != source_sha:
        raise RuntimeError("historical source manifest changed during staging")
    model = locate_model_checkpoint(reconstruction_root, subject, case_id)
    if model.is_symlink():
        raise RuntimeError("refusing symlink frozen model checkpoint")
    model_target = destination / "model.pt"
    shutil.copy2(model, model_target)
    model_sha = digest(model_target, digest_cache)
    runtime = {"schema": "ismrm_v2_2/reliability_audit_case_runtime/v2", "subject": subject, "case_id": case_id, "source_manifest": str(source_manifest), "source_manifest_sha256": source_sha, "sparse_manifest": str(working_manifest), "sparse_manifest_sha256": digest(working_manifest), "prepared_root": str(prepared_root), "model_checkpoint": str(model_target), "model_checkpoint_sha256": model_sha, "geometry_checkpoint": str(model_target), "geometry_checkpoint_sha256": model_sha, "reconstruction_checkpoint": str(model_target), "reconstruction_checkpoint_sha256": model_sha, "input_observation_mode": "source" if use_source else "derived", "active_stacks": [record["stack"] for record in active_records], "active_observations": active_records}
    (destination / "case_runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"label": item.get("label"), "subject": subject, "requested_case": item.get("case_id"), "requested_selector": {key: item[key] for key in ("case_id", "profile_id", "fold_id") if key in item}, "profile_id": item.get("profile_id"), "fold_id": item.get("fold_id"), "resolved_case_id": case_id, "input_dir": str(destination), "case_runtime": str(destination / "case_runtime.json"), "source_manifest": str(source_manifest), "working_manifest": str(working_manifest)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--case-plan", required=True); parser.add_argument("--manifest-root", required=True); parser.add_argument("--geometry-root", required=True); parser.add_argument("--reconstruction-root", required=True); parser.add_argument("--observation-root", action="append", default=[]); parser.add_argument("--staging-id", help="Optional immutable run-scoped staging ID.")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    observation_root_values = args.observation_root or [str(root / "runs" / "frozen_baseline"), str(root / "runs" / "sparse_inputs"), args.manifest_root]
    for value in (args.case_plan, args.manifest_root, args.geometry_root, args.reconstruction_root, *observation_root_values):
        path = Path(value).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"path escapes project root: {path}")
    plan = json.loads(Path(args.case_plan).read_text(encoding="utf-8"))
    if plan.get("schema") != "ismrm_v2_2/reliability_audit_case_plan/v1" or len(plan.get("cases", [])) != 8:
        raise ValueError("case plan must contain exactly eight cases")
    staging_id = args.staging_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest_cache: dict[Path, str] = {}
    resolved_entries = []
    for item in plan["cases"]:
        staged_item = dict(item)
        staged_item["input_dir"] = str(Path("runs/v2_2/00_preflight_inputs") / staging_id / "{subject}" / "{case_id}")
        resolved_entries.append(stage_case(staged_item, root=root, manifest_root=Path(args.manifest_root).resolve(), geometry_root=Path(args.geometry_root).resolve(), reconstruction_root=Path(args.reconstruction_root).resolve(), observation_roots=[Path(value).resolve() for value in observation_root_values], digest_cache=digest_cache))
    resolved_plan_path = root / "runs" / "v2_2" / "00_preflight_inputs" / staging_id / "resolved_case_plan.json"
    resolved_plan_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_plan_path.write_text(json.dumps({"schema": "ismrm_v2_2/reliability_audit_resolved_case_plan/v1", "source_case_plan": str(Path(args.case_plan).resolve()), "cases": resolved_entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(resolved_plan_path)


if __name__ == "__main__":
    main()
