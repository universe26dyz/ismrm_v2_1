#!/usr/bin/env bash
set -euo pipefail

# Read-only summary audit. This does not run MATLAB/CUDA/reconstruction and
# does not copy, move, delete, or modify subject data.

DATA_ROOT=${DATA_ROOT:-/data/dengyz/dataset/Code_10w_v1}
RAW_ROOT=${RAW_ROOT:-/data/dengyz/dataset/mapping_subject_dicom}

CASES=(
  L100_fold0 L25_fold0 L50_fold0
  SAX100_2CH0_4CH0 SAX100_2CH0_4CH100
  SAX100_2CH100_4CH0 S2CH100_4CH50_fold0
  S2CH50_4CH100_fold0
)

printf '===== PROJECT =====\n'
printf 'repo: %s\n' "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
printf 'commit: %s\n' "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

for SUBJECT in HHZ CYJ DYZ DYL_2 HJL; do
  if [ "$SUBJECT" = HHZ ]; then
    PRE="$DATA_ROOT/Code_10w_preprocessed/HHZ"
    PREP="$DATA_ROOT/Code_10w_prepared_scmr_v5/HHZ"
    RUN="$DATA_ROOT/Code_10w_runs/ismrm_v2_scmr_v5"
    # Verified supplemental source; formal HHZ run has no Proposed subtree.
    PROPOSED="$DATA_ROOT/Code_10w_runs/ismrm_v2/reconstruction/proposed/HHZ"
  elif [ "$SUBJECT" = DYL_2 ]; then
    RUN="$DATA_ROOT/Code_10w_runs/scmr_v5_rebuild_20261102_corrected/$SUBJECT"
    PRE="$RUN/preprocessed/$SUBJECT"
    PREP="$RUN/prepared/$SUBJECT"
    PROPOSED="$RUN/reconstruction/proposed/$SUBJECT"
  else
    RUN="$DATA_ROOT/Code_10w_runs/scmr_v5_rebuild_20261102_corrected/$SUBJECT"
    PRE="$DATA_ROOT/Code_10w_preprocessed/$SUBJECT"
    PREP="$RUN/prepared/$SUBJECT"
    PROPOSED="$RUN/reconstruction/proposed/$SUBJECT"
  fi

  printf '\n===== %s =====\n' "$SUBJECT"
  for STACK in sax 2ch 4ch; do
    N=$(find "$RAW_ROOT/$SUBJECT/$STACK" -type f 2>/dev/null | wc -l)
    if [ "$N" -gt 0 ]; then
      printf 'DICOM      %-4s PASS n=%s\n' "$STACK" "$N"
    else
      printf 'DICOM      %-4s MISSING\n' "$STACK"
    fi

    if [ -f "$PRE/$STACK/preprocessed.mat" ] &&
       [ -f "$PRE/$STACK/preprocess_qc.json" ]; then
      printf 'PREPROCESS %-4s PASS\n' "$STACK"
    else
      printf 'PREPROCESS %-4s MISSING\n' "$STACK"
    fi

    if [ -f "$PREP/$STACK/observations.npz" ] &&
       [ -f "$PREP/$STACK/manifest.json" ]; then
      printf 'PREPARED   %-4s PASS\n' "$STACK"
    else
      printf 'PREPARED   %-4s MISSING\n' "$STACK"
    fi
  done

  COMPLETE=0
  for CASE in "${CASES[@]}"; do
    P="$PROPOSED/$CASE"
    F="$RUN/reconstruction/fit_first/$SUBJECT/$CASE"
    FP="$RUN/evaluation/final_pose/$SUBJECT"
    M="$RUN/evaluation/common_native/$SUBJECT/$CASE"
    A="$M/all_stack_roi_metrics_v2_blood_pool_zero_audit/aggregate_statistics.json"

    P_OK=N; F_OK=N; FP_OK=N; M_OK=N; A_OK=N
    [ -f "$P/T1_3D.nii.gz" ] && [ -f "$P/T2_3D.nii.gz" ] && P_OK=Y
    [ -f "$F/nesvor/t1/volume_ms.nii.gz" ] &&
      [ -f "$F/nesvor/t2/volume_ms.nii.gz" ] && F_OK=Y
    [ -d "$FP/bundles/proposed/$CASE" ] &&
      [ -d "$FP/bundles/fit_first/$CASE" ] &&
      [ -f "$FP/roi/mapping/$CASE.json" ] && FP_OK=Y
    [ -f "$M/manifest.json" ] && M_OK=Y
    [ -f "$A" ] && A_OK=Y

    [ "$P_OK$F_OK$FP_OK$M_OK$A_OK" = YYYYY ] && COMPLETE=$((COMPLETE + 1))
    printf '%-26s P=%s F=%s FP=%s M=%s A=%s\n' \
      "$CASE" "$P_OK" "$F_OK" "$FP_OK" "$M_OK" "$A_OK"
  done

  if [ "$COMPLETE" -eq 8 ]; then
    printf 'PIPELINE %s/8 PASS\n' "$COMPLETE"
  else
    printf 'PIPELINE %s/8 INCOMPLETE\n' "$COMPLETE"
  fi
done
