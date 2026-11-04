#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${1:?usage: $0 RUN_ROOT SUBJECT MANIFEST [PYTHON]}
SUBJECT=${2:?usage: $0 RUN_ROOT SUBJECT MANIFEST [PYTHON]}
MANIFEST=${3:?usage: $0 RUN_ROOT SUBJECT MANIFEST [PYTHON]}
PYTHON=${4:-python}
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m ismrm_v2.io.validate_subject_id "$SUBJECT"
PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/common${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m ismrm_v2.observations.prepare_all \
  --manifest "$MANIFEST" --preprocessed-root "$RUN_ROOT/preprocessed" \
  --prepared-root "$RUN_ROOT/prepared" --protocol "$PROJECT_ROOT/configs/protocol_scmr_v5_10_100.yaml"
