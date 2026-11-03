#!/usr/bin/env python3
"""Validate a generic subject identifier used in filesystem paths."""

from __future__ import annotations

import argparse
import re


SAFE_SUBJECT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def validate_subject_id(subject: str) -> str:
    if not SAFE_SUBJECT.fullmatch(subject) or subject in {".", ".."} or ".." in subject:
        raise ValueError("subject must be a non-empty safe filename without path traversal")
    return subject


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject")
    args = parser.parse_args()
    print(f"SUBJECT ID: PASS ({validate_subject_id(args.subject)})")


if __name__ == "__main__":
    main()
