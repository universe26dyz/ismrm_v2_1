#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${1:?usage: $0 RUN_ROOT SUBJECT [MATLAB]}
SUBJECT=${2:?usage: $0 RUN_ROOT SUBJECT [MATLAB]}
MATLAB=${3:-matlab}; PYTHON=${PYTHON:-python}; NESVOR=${NESVOR:-nesvor}
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m ismrm_v2.io.validate_subject_id "$SUBJECT"
PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/nesvor${PYTHONPATH:+:$PYTHONPATH}" REPO="$PROJECT_ROOT" PY="$PYTHON" NESVOR="$NESVOR" \
  bash "$PROJECT_ROOT/pipeline/05_SCMR_V5_FIT_FIRST/internal/run_scmr_v5_fit_first.sh" \
  "$RUN_ROOT" "$RUN_ROOT/preprocessed" "$SUBJECT" "$RUN_ROOT/prepared" "$MATLAB"
