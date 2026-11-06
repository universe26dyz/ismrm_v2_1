#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:?Set ROOT to the project-local checkout}"
: "${CASE_PLAN:?Set CASE_PLAN to a project-local JSON plan}"
: "${AUDIT_ROOT:=${ROOT}/runs/v2_2/02_same_world_audit}"
: "${AGGREGATE_DIR:=${ROOT}/runs/v2_2/03_same_world_aggregate/$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "${CASE_PLAN}" == "${ROOT}"/* && -f "${CASE_PLAN}" ]] || { echo "CASE_PLAN must be an existing project-local file" >&2; exit 2; }
[[ "${AUDIT_ROOT}" == "${ROOT}/runs/v2_2"/* && "${AGGREGATE_DIR}" == "${ROOT}/runs/v2_2"/* ]] || { echo "outputs must be project-local under runs/v2_2" >&2; exit 2; }
mapfile -t CASE_DIRS < <(find "${AUDIT_ROOT}" -mindepth 1 -maxdepth 2 -name manifest.json -printf '%h\n' | sort)
[[ "${#CASE_DIRS[@]}" -eq 8 ]] || { echo "expected exactly eight completed case directories" >&2; exit 2; }
for directory in "${CASE_DIRS[@]}"; do
  python - "${directory}/manifest.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("status") not in {"PASS", "PASS_NO_TARGET_STACK"}:
    raise SystemExit(f"case is not PASS/no-target: {sys.argv[1]}: {payload.get('status')}")
PY
done
args=()
for directory in "${CASE_DIRS[@]}"; do args+=(--audit-dir "${directory}"); done
cd "${ROOT}"
PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python -m ismrm_v2.v2_2.preflight.aggregate_reliability_audits "${args[@]}" --output-dir "${AGGREGATE_DIR}"
