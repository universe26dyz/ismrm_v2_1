#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SUBJECT=${1:?usage: $0 SUBJECT [RAW_ROOT]}
RAW_ROOT=${2:-"$PROJECT_ROOT/data/raw_dicom"}
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m ismrm_v2.io.validate_subject_id "$SUBJECT"
MANIFEST_ROOT=${MANIFEST_ROOT:-"$PROJECT_ROOT/subjects"}
MANIFEST="$MANIFEST_ROOT/$SUBJECT/deployment_manifest.json"
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m ismrm_v2.io.build_subject_deployment_manifest \
  --raw-root "$RAW_ROOT" --subject "$SUBJECT" --output "$MANIFEST"
for stack in sax 2ch 4ch; do
  count=$(find "$RAW_ROOT/$SUBJECT/$stack" -maxdepth 1 -type f -name '*.dcm' | wc -l)
  [ "$count" -gt 0 ] && [ $((count % 10)) -eq 0 ] || { echo "invalid $stack DICOM count: $count" >&2; exit 1; }
  echo "$stack DICOM files: $count"
done
echo "RAW DICOM: PASS"
