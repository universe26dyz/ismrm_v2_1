#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:?Set ROOT to the project-local checkout}"
: "${CASE_PLAN:?Set CASE_PLAN to a project-local JSON plan}"
: "${RUN_ROOT:=${ROOT}/runs/v2_2/02_same_world_audit/$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "${CASE_PLAN}" == "${ROOT}"/* && -f "${CASE_PLAN}" ]] || { echo "CASE_PLAN must be an existing project-local file" >&2; exit 2; }
[[ "${RUN_ROOT}" == "${ROOT}/runs/v2_2"/* && ! -e "${RUN_ROOT}" ]] || { echo "RUN_ROOT must be a new project-local directory" >&2; exit 2; }
[[ "$(git -C "${ROOT}" branch --show-current)" == "ismrm_v2_2" ]] || { echo "wrong branch" >&2; exit 2; }

mapfile -t CASE_ROWS < <(python - "${CASE_PLAN}" "${ROOT}" <<'PY'
import json, sys
from pathlib import Path
plan = json.load(open(sys.argv[1], encoding="utf-8"))
if plan.get("schema") != "ismrm_v2_2/reliability_audit_case_plan/v1": raise SystemExit("unsupported case-plan schema")
cases = plan.get("cases")
if not isinstance(cases, list) or len(cases) != 8: raise SystemExit("case plan must contain exactly eight cases")
seen = set()
for item in cases:
    for key in ("label", "subject", "case_id", "input_dir"):
        if not isinstance(item.get(key), str): raise SystemExit(f"case missing {key}")
    key = (item["subject"], item["case_id"])
    if key in seen: raise SystemExit(f"duplicate case: {key}")
    seen.add(key)
    root = Path(sys.argv[2])
    input_dir = (root / item["input_dir"].format(**item)).resolve()
    if not input_dir.is_relative_to(root.resolve()): raise SystemExit(f"case input_dir escapes project root: {input_dir}")
    print("\t".join((item["label"], item["subject"], item["case_id"], str(input_dir), str((input_dir / "sparse_manifest.json").resolve()), str((input_dir / "prepared").resolve()), str((input_dir / "geometry_checkpoint.pt").resolve()), str((input_dir / "timing_only_parameters.npz").resolve()), str((input_dir / "reconstruction_checkpoint.pt").resolve()), str((input_dir / "b6_model.pt").resolve()), str((input_dir / "protocol.yaml").resolve()))))
PY
)
[[ "${#CASE_ROWS[@]}" -eq 8 ]] || { echo "case plan did not resolve exactly eight cases" >&2; exit 2; }

mkdir -p "${RUN_ROOT}"
OUTPUT_DIRS=()
for row in "${CASE_ROWS[@]}"; do
  IFS=$'\t' read -r label subject case_id input_dir sparse_manifest prepared_root geometry_checkpoint timing_npz reconstruction_checkpoint b6_model protocol_yaml <<< "${row}"
  [[ -d "${input_dir}" && -f "${sparse_manifest}" && -d "${prepared_root}" && -f "${geometry_checkpoint}" ]] || { echo "missing case input for ${label}: ${input_dir}" >&2; exit 2; }
  output_dir="${RUN_ROOT}/${subject}/${case_id}"
  export SUBJECT="${subject}" CASE="${case_id}" PREPARED_ROOT="${prepared_root}" GEOMETRY_CHECKPOINT="${geometry_checkpoint}" OUTPUT_DIR="${output_dir}" SPARSE_MANIFEST="${sparse_manifest}"
  if [[ -f "${timing_npz}" ]]; then export TIMING_ONLY_PARAMETERS_NPZ="${timing_npz}"; else unset TIMING_ONLY_PARAMETERS_NPZ || true; fi
  if [[ "${EXPORT_TIMING_ONLY:-0}" == "1" ]]; then
    [[ -f "${reconstruction_checkpoint}" && -f "${b6_model}" && -f "${protocol_yaml}" ]] || { echo "timing export inputs missing for ${label}" >&2; exit 2; }
    PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python -m ismrm_v2.v2_2.preflight.export_timing_only_anchor_parameters --subject "${subject}" --case "${case_id}" --prepared-root "${prepared_root}" --sparse-manifest "${sparse_manifest}" --geometry-checkpoint "${geometry_checkpoint}" --b6-model "${b6_model}" --protocol-yaml "${protocol_yaml}" --reconstruction-checkpoint "${reconstruction_checkpoint}" --output "${input_dir}/timing_only_parameters.npz" --device "${DEVICE:-cpu}"
    export TIMING_ONLY_PARAMETERS_NPZ="${input_dir}/timing_only_parameters.npz"
  fi
  "${ROOT}/scripts/v2_2/run_reliability_audit_case.sh"
  status=$(python - "${output_dir}/manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("status"))
PY
)
  [[ "${status}" == "PASS" || "${status}" == "PASS_NO_TARGET_STACK" ]] || { echo "case ${label} failed with ${status}" >&2; exit 2; }
  OUTPUT_DIRS+=("${output_dir}")
done

aggregate_dir="${ROOT}/runs/v2_2/03_same_world_aggregate/$(basename "${RUN_ROOT}")"
args=()
for directory in "${OUTPUT_DIRS[@]}"; do args+=(--audit-dir "${directory}"); done
PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python -m ismrm_v2.v2_2.preflight.aggregate_reliability_audits "${args[@]}" --output-dir "${aggregate_dir}"
