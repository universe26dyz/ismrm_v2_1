#!/usr/bin/env python3
"""Build one deployment-v1.1 manifest from actual subject stack directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_subject_id import validate_subject_id


STACKS = ("sax", "2ch", "4ch")


def build_subject_deployment_manifest(
    raw_root: str | Path, subject: str, output: str | Path, server_raw_root: str | Path | None = None,
) -> Path:
    subject = validate_subject_id(subject)
    local_root = Path(raw_root).resolve()
    server_root = Path(server_raw_root).resolve() if server_raw_root is not None else local_root
    stacks = []
    for stack in STACKS:
        local_dir, server_dir = local_root / subject / stack, server_root / subject / stack
        if not local_dir.is_dir():
            raise FileNotFoundError(f"missing local stack directory: {local_dir}")
        if not server_dir.is_dir():
            raise FileNotFoundError(f"missing server stack directory: {server_dir}")
        stacks.append({"stack": stack, "local_dicom_dir": str(local_dir), "server_dicom_dir": str(server_dir)})
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"schema_version": "deployment-v1.1", "subjects": [{"subject_id": subject, "stacks": stacks}]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--server-raw-root")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build_subject_deployment_manifest(**vars(args)))


if __name__ == "__main__":
    main()
