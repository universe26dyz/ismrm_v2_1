#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${1:?usage: $0 RUN_ROOT SUBJECT MANIFEST [MATLAB] [PYTHON]}
SUBJECT=${2:?usage: $0 RUN_ROOT SUBJECT MANIFEST [MATLAB] [PYTHON]}
MANIFEST=${3:?usage: $0 RUN_ROOT SUBJECT MANIFEST [MATLAB] [PYTHON]}
MATLAB=${4:-matlab}; PYTHON=${5:-python}
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m ismrm_v2.io.validate_subject_id "$SUBJECT"
PREPROCESSED_ROOT="$RUN_ROOT/preprocessed"
[ ! -e "$PREPROCESSED_ROOT" ] || { echo "refusing to overwrite: $PREPROCESSED_ROOT" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
"$MATLAB" -batch "addpath('$PROJECT_ROOT/matlab/preprocessing'); local_preprocess_all('$MANIFEST','$PREPROCESSED_ROOT');" \
  2>&1 | tee "$RUN_ROOT/01_server_preprocess.log"
PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/nesvor${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m ismrm_v2.preprocessing.validation \
  --manifest "$MANIFEST" --preprocessed-root "$PREPROCESSED_ROOT" \
  --output "$PREPROCESSED_ROOT/$SUBJECT/preprocess_validation.json" \
  2>&1 | tee "$RUN_ROOT/02_preprocessed_validation.log"
echo "Server-only MATLAB preprocessing: PASS"
