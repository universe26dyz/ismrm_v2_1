#!/usr/bin/env bash
set -euo pipefail

# Read-only audit of the server-local, project-contained frozen baseline.
# It never runs reconstruction or changes project data.

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RAW_ROOT="$PROJECT_ROOT/data/raw_dicom"
ROI_ROOT="$PROJECT_ROOT/data/roi"
BASE="$PROJECT_ROOT/runs/frozen_baseline"
MODEL_ROOT="$PROJECT_ROOT/runtime/model_assets"
DECODER_SHA=338f2c0b8db90851f733ed31caa2c99914605fed3f7215ec2f69f19eba59583b
B6_SHA=15981fbe93d2c70e14755ea756a02856c9383481e7d92d239619268a628a29b5

CASES=(
  L100_fold0 L25_fold0 L50_fold0
  SAX100_2CH0_4CH0 SAX100_2CH0_4CH100
  SAX100_2CH100_4CH0 S2CH100_4CH50_fold0
  S2CH50_4CH100_fold0
)

require_model() {
  local path=$1 expected=$2 label=$3 actual
  if [ ! -f "$path" ]; then
    printf 'MODEL %s MISSING\n' "$label"
    return
  fi
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [ "$actual" = "$expected" ]; then
    printf 'MODEL %s PASS\n' "$label"
  else
    printf 'MODEL %s SHA_MISMATCH\n' "$label"
  fi
}

printf '===== PROJECT =====\n'
printf 'repo: %s\n' "$PROJECT_ROOT"
printf 'commit: %s\n' "$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
printf 'raw_root: %s\nroi_root: %s\nfrozen_root: %s\n' "$RAW_ROOT" "$ROI_ROOT" "$BASE"
require_model "$MODEL_ROOT/decoder_scmr_v5_10_100.pth" "$DECODER_SHA" decoder_scmr_v5_10_100.pth
require_model "$MODEL_ROOT/b6_model.pt" "$B6_SHA" b6_model.pt

for SUBJECT in CYJ DYL_2 DYZ HHZ HJL; do
  RUN="$BASE/$SUBJECT"
  PRE="$RUN/preprocessed/$SUBJECT"
  PREP="$RUN/prepared/$SUBJECT"
  PROPOSED="$RUN/reconstruction/proposed/$SUBJECT"
  FIT_FIRST="$RUN/reconstruction/fit_first/$SUBJECT"
  EVALUATION="$RUN/evaluation"
  ROI_BUNDLE="$ROI_ROOT/$SUBJECT/20261026_scmr_native_roi_bundle"

  printf '\n===== %s =====\n' "$SUBJECT"
  [ -d "$ROI_BUNDLE" ] && printf 'ROI        PASS\n' || printf 'ROI        MISSING\n'
  for STACK in sax 2ch 4ch; do
    N=$(find "$RAW_ROOT/$SUBJECT/$STACK" -type f -name '*.dcm' 2>/dev/null | wc -l)
    if [ "$N" -gt 0 ]; then
      printf 'DICOM      %-4s PASS n=%s\n' "$STACK" "$N"
    else
      printf 'DICOM      %-4s MISSING\n' "$STACK"
    fi
    if [ -f "$PRE/$STACK/preprocessed.mat" ] && [ -f "$PRE/$STACK/preprocess_qc.json" ]; then
      printf 'PREPROCESS %-4s PASS\n' "$STACK"
    else
      printf 'PREPROCESS %-4s MISSING\n' "$STACK"
    fi
    if [ -f "$PREP/$STACK/observations.npz" ] && [ -f "$PREP/$STACK/manifest.json" ]; then
      printf 'PREPARED   %-4s PASS\n' "$STACK"
    else
      printf 'PREPARED   %-4s MISSING\n' "$STACK"
    fi
  done

  COMPLETE=0
  for CASE in "${CASES[@]}"; do
    P="$PROPOSED/$CASE"
    F="$FIT_FIRST/$CASE"
    FP="$EVALUATION/final_pose/$SUBJECT"
    M="$EVALUATION/common_native/$SUBJECT/$CASE"
    A="$M/all_stack_roi_metrics_v2_blood_pool_zero_audit/aggregate_statistics.json"
    P_OK=N; F_OK=N; FP_OK=N; M_OK=N; A_OK=N
    [ -f "$P/T1_3D.nii.gz" ] && [ -f "$P/T2_3D.nii.gz" ] && P_OK=Y
    [ -f "$F/nesvor/t1/volume_ms.nii.gz" ] && [ -f "$F/nesvor/t2/volume_ms.nii.gz" ] && F_OK=Y
    [ -d "$FP/bundles/proposed/$CASE" ] && [ -d "$FP/bundles/fit_first/$CASE" ] && [ -f "$FP/roi/mapping/$CASE.json" ] && FP_OK=Y
    [ -f "$M/manifest.json" ] && M_OK=Y
    [ -f "$A" ] && A_OK=Y
    [ "$P_OK$F_OK$FP_OK$M_OK$A_OK" = YYYYY ] && COMPLETE=$((COMPLETE + 1))
    printf '%-26s P=%s F=%s FP=%s M=%s A=%s\n' "$CASE" "$P_OK" "$F_OK" "$FP_OK" "$M_OK" "$A_OK"
  done
  [ "$COMPLETE" -eq 8 ] && printf 'PIPELINE %s/8 PASS\n' "$COMPLETE" || printf 'PIPELINE %s/8 INCOMPLETE\n' "$COMPLETE"
done
