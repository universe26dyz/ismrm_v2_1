#!/usr/bin/env python3
"""Fail closed for the frozen ISMRM-v2.1 submission cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_contract(root: Path) -> dict:
    return json.loads((root / "configs/scientific_freeze_contract.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subject")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    contract = load_contract(args.root.resolve())
    if args.subject not in contract["active_submission_cohort"]:
        reason = contract.get("rejected_subjects", {}).get(args.subject, "not in the active submission cohort")
        raise SystemExit(f"subject rejected: {args.subject}: {reason}")
    print(f"ACTIVE SUBJECT CONTRACT: PASS ({args.subject})")


if __name__ == "__main__":
    main()
