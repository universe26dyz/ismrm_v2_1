#!/usr/bin/env bash
set -euo pipefail

write_failure_status() {
  local code=$?
  set +e
  mkdir -p "${OUTPUT_DIR:-.}"
  python - "${OUTPUT_DIR:-.}/failure_status.json" "${SUBJECT:-}" "${CASE:-}" "${code}" <<'PY'
import json, sys, traceback
path, subject, case_id, code = sys.argv[1:]
json.dump({"status": "BLOCKED_RUNTIME", "stage": "single_case_audit", "subject": subject, "resolved_case": case_id, "exception_type": "shell_command_failure", "message": f"exit code {code}"}, open(path, "w", encoding="utf-8"), indent=2)
PY
  exit "${code}"
}
trap write_failure_status ERR

: "${ROOT:?Set ROOT to the project-local checkout}"
: "${SUBJECT:?Set SUBJECT}"; : "${CASE:?Set CASE}"
: "${PREPARED_ROOT:?Set PREPARED_ROOT}"; : "${GEOMETRY_CHECKPOINT:?Set GEOMETRY_CHECKPOINT}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR under ${ROOT}/runs/v2_2}"
[[ -d "${ROOT}" && -d "${PREPARED_ROOT}" && -f "${GEOMETRY_CHECKPOINT}" ]] || { echo "missing local input" >&2; exit 2; }
[[ "$(git -C "${ROOT}" branch --show-current)" == "ismrm_v2_2" ]] || { echo "wrong branch" >&2; exit 2; }
[[ "${OUTPUT_DIR}" == "${ROOT}/runs/v2_2"/* ]] || { echo "OUTPUT_DIR must be project-local under runs/v2_2" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}" || -z "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || { echo "refusing to overwrite existing output" >&2; exit 2; }

cd "${ROOT}"
args=(--subject "${SUBJECT}" --case "${CASE}" --prepared-root "${PREPARED_ROOT}" --geometry-checkpoint "${GEOMETRY_CHECKPOINT}" --output-dir "${OUTPUT_DIR}" --device "${DEVICE:-cpu}")
[[ -n "${SPARSE_MANIFEST:-}" ]] && args+=(--sparse-manifest "${SPARSE_MANIFEST}")
[[ -n "${SIGNAL_DECODER_CHECKPOINT:-}" ]] && args+=(--signal-decoder-checkpoint "${SIGNAL_DECODER_CHECKPOINT}")
[[ -n "${PROTOCOL_YAML:-}" ]] && args+=(--protocol-yaml "${PROTOCOL_YAML}")
[[ -n "${TIMING_ONLY_PARAMETERS_NPZ:-}" ]] && args+=(--timing-only-parameters-npz "${TIMING_ONLY_PARAMETERS_NPZ}")
PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python -m ismrm_v2.v2_2.preflight.same_world_reliability_audit "${args[@]}"
