#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:?Set ROOT to the project-local checkout}"
: "${CASE_PLAN:?Set CASE_PLAN to a project-local JSON plan}"
: "${RUN_ROOT:=${ROOT}/runs/v2_2/02_same_world_audit/$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "${CASE_PLAN}" == "${ROOT}"/* && -f "${CASE_PLAN}" ]] || { echo "CASE_PLAN must be an existing project-local file" >&2; exit 2; }
[[ "${RUN_ROOT}" == "${ROOT}/runs/v2_2"/* && ! -e "${RUN_ROOT}" ]] || { echo "RUN_ROOT must be a new project-local directory" >&2; exit 2; }
[[ "$(git -C "${ROOT}" branch --show-current)" == "ismrm_v2_2" ]] || { echo "wrong branch" >&2; exit 2; }

DECODER_PATH="${ROOT}/runtime/model_assets/decoder_scmr_v5_10_100.pth"
B6_PATH="${ROOT}/runtime/model_assets/b6_model.pt"
PROTOCOL_PATH="${ROOT}/configs/protocol_scmr_v5_10_100.yaml"
verify_sha() { [[ -f "$1" ]] || { echo "missing canonical asset: $1" >&2; exit 2; }; [[ "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]] || { echo "canonical asset SHA256 mismatch: $1" >&2; exit 2; }; }
verify_sha "${DECODER_PATH}" 338f2c0b8db90851f733ed31caa2c99914605fed3f7215ec2f69f19eba59583b
verify_sha "${B6_PATH}" 15981fbe93d2c70e14755ea756a02856c9383481e7d92d239619268a628a29b5
verify_sha "${PROTOCOL_PATH}" 3865b6bde1154750d4ed687456823fafad7b0cad5a61f763decb27dd1a4ed1c7
export SIGNAL_DECODER_CHECKPOINT="${DECODER_PATH}" PROTOCOL_YAML="${PROTOCOL_PATH}"

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
    runtime = input_dir / "case_runtime.json"
    if runtime.is_file():
        metadata = json.load(runtime.open(encoding="utf-8"))
        fields = (metadata["sparse_manifest"], metadata["prepared_root"], metadata["geometry_checkpoint"], metadata.get("timing_only_parameters", str(input_dir / "timing_only_parameters.npz")), metadata.get("reconstruction_checkpoint", str(input_dir / "reconstruction_checkpoint.pt")))
    else:
        fields = (str((input_dir / "sparse_manifest.json").resolve()), str((input_dir / "prepared").resolve()), str((input_dir / "geometry_checkpoint.pt").resolve()), str((input_dir / "timing_only_parameters.npz").resolve()), str((input_dir / "reconstruction_checkpoint.pt").resolve()))
    print("\t".join((item["label"], item["subject"], item["case_id"], str(input_dir), *fields)))
PY
)
[[ "${#CASE_ROWS[@]}" -eq 8 ]] || { echo "case plan did not resolve exactly eight cases" >&2; exit 2; }

mkdir -p "${RUN_ROOT}"
OUTPUT_DIRS=()
for row in "${CASE_ROWS[@]}"; do
  IFS=$'\t' read -r label subject case_id input_dir sparse_manifest prepared_root geometry_checkpoint timing_npz reconstruction_checkpoint <<< "${row}"
  [[ -d "${input_dir}" && -f "${sparse_manifest}" && -d "${prepared_root}" && -f "${geometry_checkpoint}" ]] || { echo "missing case input for ${label}: ${input_dir}" >&2; exit 2; }
  output_dir="${RUN_ROOT}/${subject}/${case_id}"
  export SUBJECT="${subject}" CASE="${case_id}" PREPARED_ROOT="${prepared_root}" GEOMETRY_CHECKPOINT="${geometry_checkpoint}" OUTPUT_DIR="${output_dir}" SPARSE_MANIFEST="${sparse_manifest}"
  if [[ -f "${timing_npz}" ]]; then export TIMING_ONLY_PARAMETERS_NPZ="${timing_npz}"; else unset TIMING_ONLY_PARAMETERS_NPZ || true; fi
  if [[ "${EXPORT_TIMING_ONLY:-0}" == "1" ]]; then
    [[ -f "${reconstruction_checkpoint}" ]] || { echo "timing export reconstruction checkpoint missing for ${label}" >&2; exit 2; }
    PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python -m ismrm_v2.v2_2.preflight.export_timing_only_anchor_parameters --subject "${subject}" --case "${case_id}" --prepared-root "${prepared_root}" --sparse-manifest "${sparse_manifest}" --geometry-checkpoint "${geometry_checkpoint}" --b6-model "${B6_PATH}" --protocol-yaml "${PROTOCOL_PATH}" --signal-decoder-checkpoint "${DECODER_PATH}" --reconstruction-checkpoint "${reconstruction_checkpoint}" --output "${input_dir}/timing_only_parameters.npz" --device "${DEVICE:-cpu}"
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
