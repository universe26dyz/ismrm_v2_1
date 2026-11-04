#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${1:?usage: $0 RUN_ROOT SUBJECT [PYTHON]}
SUBJECT=${2:?usage: $0 RUN_ROOT SUBJECT [PYTHON]}
PYTHON=${3:-python}
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m ismrm_v2.io.validate_subject_id "$SUBJECT"
PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/nesvor${PYTHONPATH:+:$PYTHONPATH}" REPO="$PROJECT_ROOT" \
  bash "$PROJECT_ROOT/pipeline/04_SPARSE_PROFILES/internal/build_sparse_profiles.sh" \
  "$RUN_ROOT/prepared" "$SUBJECT" "$RUN_ROOT/sparse_inputs" "$PYTHON"
