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
  bash "$PROJECT_ROOT/pipeline/08_EVALUATION_AND_VISUALIZATION/internal/hhz_common_native_metrics_qc.sh" \
  "$RUN_ROOT" "$RUN_ROOT" "$ROI_BUNDLE" "$CASE"
OUTPUT_NAME=all_stack_roi_metrics_v2_blood_pool_zero_audit PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/nesvor${PYTHONPATH:+:$PYTHONPATH}" REPO="$PROJECT_ROOT" SUBJECT="$SUBJECT" \
  bash "$PROJECT_ROOT/pipeline/08_EVALUATION_AND_VISUALIZATION/internal/hhz_common_native_all_stack_roi_metrics.sh" \
  "$RUN_ROOT" "$RUN_ROOT" "$ROI_BUNDLE" "$CASE"
