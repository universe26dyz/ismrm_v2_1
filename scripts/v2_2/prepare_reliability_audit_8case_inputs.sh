#!/usr/bin/env bash
set -euo pipefail
: "${ROOT:?Set ROOT to project-local checkout}"
: "${CASE_PLAN:?Set CASE_PLAN to project-local plan}"
: "${MANIFEST_ROOT:?Set MANIFEST_ROOT to project-local sparse manifests}"
: "${GEOMETRY_ROOT:?Set GEOMETRY_ROOT to project-local geometry checkpoints}"
: "${RECONSTRUCTION_ROOT:?Set RECONSTRUCTION_ROOT to project-local reconstruction checkpoints}"
for path in "$CASE_PLAN" "$MANIFEST_ROOT" "$GEOMETRY_ROOT" "$RECONSTRUCTION_ROOT"; do
  [[ "$path" == "$ROOT"/* ]] || { echo "staging input escapes ROOT: $path" >&2; exit 2; }
done
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python "$ROOT/scripts/v2_2/prepare_reliability_audit_8case_inputs.py" --root "$ROOT" --case-plan "$CASE_PLAN" --manifest-root "$MANIFEST_ROOT" --geometry-root "$GEOMETRY_ROOT" --reconstruction-root "$RECONSTRUCTION_ROOT"
