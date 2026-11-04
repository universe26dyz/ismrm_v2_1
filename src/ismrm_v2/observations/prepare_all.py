"""Prepare exactly the server subject/stack entries listed in a deployment manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COMMON_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = COMMON_ROOT.parent
# This file remains executable directly for deployment compatibility.  Place
# the clean package and the legacy common package ahead of ``trad/scripts``.
for import_root in reversed((PROJECT_ROOT / "src", COMMON_ROOT)):
    while str(import_root) in sys.path:
        sys.path.remove(str(import_root))
    sys.path.insert(0, str(import_root))

from ismrm_v2.io.deployment_manifest import load_deployment_manifest
from ismrm_v2.observations.prepare_observations import prepare_observations


def prepare_all_observations(manifest_path: str | Path, preprocessed_root: str | Path, prepared_root: str | Path, protocol_path: str | Path) -> dict[str, object]:
    root, destination = Path(preprocessed_root), Path(prepared_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Preprocessed root does not exist: {root}")
    entries = load_deployment_manifest(manifest_path)
    batch_qc_path = destination / "PREPARED_BATCH_QC.json"
    if batch_qc_path.exists():
        raise FileExistsError(f"Prepared batch QC already exists: {batch_qc_path}")
    reports = []
    for entry in entries:
        subject_id, stack = entry["subject_id"], entry["stack"]
        mat = root / subject_id / stack / "preprocessed.mat"
        if not mat.is_file():
            raise FileNotFoundError(f"Required transferred preprocessed MAT is missing: {mat}")
        output = destination / subject_id / stack
        result = prepare_observations(mat, entry["server_dicom_dir"], stack, output, protocol_path=protocol_path)
        required = ("observations.npz", "manifest.json", "timing.npy", "qc_summary.json")
        missing = [name for name in required if not (output / name).is_file()]
        if missing:
            raise RuntimeError(f"Observation preparation did not create required outputs for {subject_id}/{stack}: {missing}")
        reports.append({"subject_id": subject_id, "stack": stack, "preprocessed_mat": str(mat), "prepared_dir": str(output), **result["qc_summary"]})
    report: dict[str, object] = {"schema_version": "prepared-batch-qc-v1", "prepared_count": len(reports), "entries": reports}
    destination.mkdir(parents=True, exist_ok=True)
    batch_qc_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preprocessed-root", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--protocol", dest="protocol_path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_all_observations(args.manifest, args.preprocessed_root, args.prepared_root, args.protocol_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
