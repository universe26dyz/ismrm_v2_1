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


def stage_case(item: dict[str, Any], *, root: Path, manifest_root: Path, geometry_root: Path, reconstruction_root: Path) -> None:
    subject, case_id = item["subject"], item["case_id"]
    destination = (root / item["input_dir"].format(**item)).resolve()
    if not destination.is_relative_to(root.resolve()) or destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"staging destination must be new and project-local: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    source_manifest = locate_manifest(manifest_root, subject, case_id)
    if source_manifest.is_symlink():
        raise RuntimeError(f"refusing symlink manifest input: {source_manifest}")
    source_sha = digest(source_manifest)
    validated = validate_sparse_profile_manifest(source_manifest, subject=subject)
    raw = json.loads(source_manifest.read_text(encoding="utf-8"))
    schema = raw.get("schema_version")
    condition = str(raw.get("condition") or raw.get("case_id") or case_id).upper()
    use_source = schema == LEGACY_SCHEMA_VERSION and condition.startswith("L100")
    if use_source:
        working_manifest = destination / "sparse_manifest.json"
        working_manifest.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prepared_root = Path(raw["prepared_root"]).resolve()
    else:
        prepared = destination / "prepared"
        prepared.mkdir()
        raw["prepared_root"] = str(prepared)
        active_records = []
        for stack in STACKS:
            spec = raw["stacks"][stack]
            present = bool(spec.get("retained_original_group_ids", []))
            if not present:
                continue
            record = spec.get("derived_observation")
            if not isinstance(record, dict):
                raise ValueError(f"{source_manifest}: {stack} lacks derived_observation")
            source = Path(record["path"]).resolve()
            target = prepared / stack / "observations.npz"
            hardlink(source, target)
            spec["derived_observation"] = {**record, "path": str(target), "sha256": digest(target)}
            active_records.append({"stack": stack, "mode": "derived", "source": str(source), "path": str(target), "sha256": digest(target), "retained_original_group_ids": spec.get("retained_original_group_ids", []), "derived_to_original_group_idx": spec.get("derived_to_original_group_idx")})
        working_manifest = prepared / "sparse_manifest.json"
        working_manifest.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_sparse_profile_manifest(working_manifest, subject=subject)
        prepared_root = prepared
    geometry = locate(geometry_root, subject, case_id, "geometry_checkpoint.pt")
    reconstruction = locate(reconstruction_root, subject, case_id, "reconstruction_checkpoint.pt")
    if geometry.is_symlink() or reconstruction.is_symlink():
        raise RuntimeError("refusing symlink checkpoint input")
    geometry_target = destination / "geometry_checkpoint.pt"; reconstruction_target = destination / "reconstruction_checkpoint.pt"
    shutil.copy2(geometry, geometry_target); shutil.copy2(reconstruction, reconstruction_target)
    runtime = {"schema": "ismrm_v2_2/reliability_audit_case_runtime/v1", "subject": subject, "case_id": case_id, "source_manifest": str(source_manifest), "source_manifest_sha256": source_sha, "sparse_manifest": str(working_manifest), "sparse_manifest_sha256": digest(working_manifest), "prepared_root": str(prepared_root), "geometry_checkpoint": str(geometry_target), "geometry_checkpoint_sha256": digest(geometry_target), "reconstruction_checkpoint": str(reconstruction_target), "reconstruction_checkpoint_sha256": digest(reconstruction_target), "input_observation_mode": "source" if use_source else "derived", "active_observations": active_records if not use_source else [{"stack": stack, "mode": "source", "path": str(Path(raw["stacks"][stack]["source_observation"]["path"]).resolve()), "sha256": raw["stacks"][stack]["source_observation"].get("sha256")} for stack in STACKS]}
    (destination / "case_runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--case-plan", required=True); parser.add_argument("--manifest-root", required=True); parser.add_argument("--geometry-root", required=True); parser.add_argument("--reconstruction-root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    for value in (args.case_plan, args.manifest_root, args.geometry_root, args.reconstruction_root):
        path = Path(value).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"path escapes project root: {path}")
    plan = json.loads(Path(args.case_plan).read_text(encoding="utf-8"))
    if plan.get("schema") != "ismrm_v2_2/reliability_audit_case_plan/v1" or len(plan.get("cases", [])) != 8:
        raise ValueError("case plan must contain exactly eight cases")
    for item in plan["cases"]:
        stage_case(item, root=root, manifest_root=Path(args.manifest_root).resolve(), geometry_root=Path(args.geometry_root).resolve(), reconstruction_root=Path(args.reconstruction_root).resolve())


if __name__ == "__main__":
    main()
