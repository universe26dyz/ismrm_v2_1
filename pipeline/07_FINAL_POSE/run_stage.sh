#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${1:?usage: $0 RUN_ROOT SUBJECT ROI_BUNDLE CASE}
SUBJECT=${2:?usage: $0 RUN_ROOT SUBJECT ROI_BUNDLE CASE}
ROI_BUNDLE=${3:?usage: $0 RUN_ROOT SUBJECT ROI_BUNDLE CASE}
CASE=${4:?usage: $0 RUN_ROOT SUBJECT ROI_BUNDLE CASE}
PYTHON=${PYTHON:-python}
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m ismrm_v2.io.validate_subject_id "$SUBJECT"
PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/nesvor${PYTHONPATH:+:$PYTHONPATH}" REPO="$PROJECT_ROOT" SUBJECT="$SUBJECT" \
  bash "$PROJECT_ROOT/pipeline/07_FINAL_POSE/internal/hhz_paired_final_pose.sh" \
  "$RUN_ROOT" "$RUN_ROOT" "$ROI_BUNDLE" "$CASE"
