"""Stage the eight-case reliability inputs without copying large observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from ismrm_v2.sampling.sparse_profile_manifest import LEGACY_SCHEMA_VERSION, validate_sparse_profile_manifest

STACKS = ("sax", "2ch", "4ch")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def locate_matching_observation(roots: list[Path], *, subject: str, case_id: str, stack: str, expected_sha256: str) -> Path:
    """Find a current project-local observation whose bytes match historical provenance."""
    candidates: list[Path] = []
    for root in roots:
        candidates.extend((root / subject / case_id / stack / "observations.npz", root / subject / stack / "observations.npz"))
        candidates.extend(sorted(root.glob(f"**/{subject}/**/{stack}/observations.npz")))
        candidates.extend(sorted(root.glob(f"**/{case_id}/**/{stack}/observations.npz")))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file() or candidate.is_symlink():
            continue
        seen.add(candidate)
        if digest(candidate) == expected_sha256:
            return candidate
    raise FileNotFoundError(f"no current project-local {stack} observation matches recorded SHA256 {expected_sha256}")


def stage_case(item: dict[str, Any], *, root: Path, manifest_root: Path, geometry_root: Path, reconstruction_root: Path, observation_roots: list[Path]) -> None:
    subject, case_id = item["subject"], item["case_id"]
    destination = (root / item["input_dir"].format(**item)).resolve()
    if not destination.is_relative_to(root.resolve()) or destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"staging destination must be new and project-local: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    source_manifest = locate_manifest(manifest_root, subject, case_id)
    if source_manifest.is_symlink():
        raise RuntimeError(f"refusing symlink manifest input: {source_manifest}")
    source_sha = digest(source_manifest)
    # Historical manifests are provenance records, not runtime inputs.  Read
    # them before validation because their absolute paths may be retired.
    raw = json.loads(source_manifest.read_text(encoding="utf-8"))
    schema = raw.get("schema_version")
    condition = str(raw.get("condition") or raw.get("case_id") or case_id).upper()
    use_source = schema == LEGACY_SCHEMA_VERSION and condition.startswith("L100")
    prepared = destination / "prepared"
    prepared.mkdir()
    source_paths: dict[str, Path] = {}
    derived_paths: dict[str, Path] = {}
    active_records = []
    for stack in STACKS:
        spec = raw["stacks"][stack]
        source_record = spec.get("source_observation")
        if not isinstance(source_record, dict) or not isinstance(source_record.get("sha256"), str):
            raise ValueError(f"{source_manifest}: {stack} lacks source SHA256 provenance")
        source_paths[stack] = locate_matching_observation(observation_roots, subject=subject, case_id=case_id, stack=stack, expected_sha256=source_record["sha256"])
        present = bool(spec.get("retained_original_group_ids", [])) if schema == LEGACY_SCHEMA_VERSION else bool(spec.get("training_stack_present"))
        if present and not use_source:
            record = spec.get("derived_observation")
            if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
                raise ValueError(f"{source_manifest}: {stack} lacks derived SHA256 provenance")
            derived_paths[stack] = locate_matching_observation(observation_roots, subject=subject, case_id=case_id, stack=stack, expected_sha256=record["sha256"])
    source_roots = {path.parent.parent for path in source_paths.values()}
    if len(source_roots) != 1 or any(path != next(iter(source_roots)) / stack / "observations.npz" for stack, path in source_paths.items()):
        raise ValueError("current source observations do not share the canonical subject prepared root")
    source_root = next(iter(source_roots))
    raw["source_prepared_root"] = str(source_root)
    prepared_root = source_root if use_source else prepared
    raw["prepared_root"] = str(prepared_root)
    for stack in STACKS:
        spec = raw["stacks"][stack]
        spec["source_observation"] = {**spec["source_observation"], "path": str(source_paths[stack]), "sha256": digest(source_paths[stack])}
        if stack in derived_paths:
            target = prepared / stack / "observations.npz"
            hardlink(derived_paths[stack], target)
            spec["derived_observation"] = {**spec["derived_observation"], "path": str(target), "sha256": digest(target)}
            active_records.append({"stack": stack, "mode": "derived", "source": str(derived_paths[stack]), "path": str(target), "sha256": digest(target), "retained_original_group_ids": spec.get("retained_original_group_ids", []), "derived_to_original_group_idx": spec.get("derived_to_original_group_idx")})
        elif use_source:
            active_records.append({"stack": stack, "mode": "source", "path": str(source_paths[stack]), "sha256": digest(source_paths[stack])})
    working_manifest = destination / "sparse_manifest.json" if use_source else prepared / "sparse_manifest.json"
    working_manifest.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_sparse_profile_manifest(working_manifest, subject=subject)
    if digest(source_manifest) != source_sha:
        raise RuntimeError("historical source manifest changed during staging")
    model = locate_model_checkpoint(reconstruction_root, subject, case_id)
    if model.is_symlink():
        raise RuntimeError("refusing symlink frozen model checkpoint")
    model_target = destination / "model.pt"
    shutil.copy2(model, model_target)
    model_sha = digest(model_target)
    runtime = {"schema": "ismrm_v2_2/reliability_audit_case_runtime/v2", "subject": subject, "case_id": case_id, "source_manifest": str(source_manifest), "source_manifest_sha256": source_sha, "sparse_manifest": str(working_manifest), "sparse_manifest_sha256": digest(working_manifest), "prepared_root": str(prepared_root), "model_checkpoint": str(model_target), "model_checkpoint_sha256": model_sha, "geometry_checkpoint": str(model_target), "geometry_checkpoint_sha256": model_sha, "reconstruction_checkpoint": str(model_target), "reconstruction_checkpoint_sha256": model_sha, "input_observation_mode": "source" if use_source else "derived", "active_observations": active_records}
    (destination / "case_runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--case-plan", required=True); parser.add_argument("--manifest-root", required=True); parser.add_argument("--geometry-root", required=True); parser.add_argument("--reconstruction-root", required=True); parser.add_argument("--observation-root", action="append", default=[])
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
    for item in plan["cases"]:
        stage_case(item, root=root, manifest_root=Path(args.manifest_root).resolve(), geometry_root=Path(args.geometry_root).resolve(), reconstruction_root=Path(args.reconstruction_root).resolve(), observation_roots=[Path(value).resolve() for value in observation_root_values])


if __name__ == "__main__":
    main()
