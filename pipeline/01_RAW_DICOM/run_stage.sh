#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SUBJECT=${1:?usage: $0 SUBJECT RAW_ROOT}
RAW_ROOT=${2:?usage: $0 SUBJECT RAW_ROOT}
python "$PROJECT_ROOT/scripts/validate_subject_id.py" "$SUBJECT"
MANIFEST_ROOT=${MANIFEST_ROOT:-"$PROJECT_ROOT/subjects"}
MANIFEST="$MANIFEST_ROOT/$SUBJECT/manifest.json"
python "$PROJECT_ROOT/scripts/build_subject_deployment_manifest.py" \
  --raw-root "$RAW_ROOT" --subject "$SUBJECT" --output "$MANIFEST"
for stack in sax 2ch 4ch; do
  count=$(find "$RAW_ROOT/$SUBJECT/$stack" -maxdepth 1 -type f -name '*.dcm' | wc -l)
  [ "$count" -gt 0 ] && [ $((count % 10)) -eq 0 ] || { echo "invalid $stack DICOM count: $count" >&2; exit 1; }
  echo "$stack DICOM files: $count"
done
echo "RAW DICOM: PASS"
