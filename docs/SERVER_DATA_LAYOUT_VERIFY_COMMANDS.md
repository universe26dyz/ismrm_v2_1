# Server data-layout verification commands

这些命令是只读的。请在 CMRServer04 上执行。它们不运行 MATLAB、CUDA、NeSVoR、重建或评估，也不使用 cp、mv、rsync、rm、chmod、chown 或 git clean。

~~~bash
cd /data/dengyz/code/ismrm_v2_1

echo '===== environment ====='
pwd
hostname
whoami
git rev-parse --show-toplevel
git rev-parse HEAD
git branch --show-current
git remote -v
git status --short --untracked-files=no

echo '===== mounts ====='
df -h /data /data/dengyz/dataset /data/dengyz/code 2>&1 || true
df -T /data /data/dengyz/dataset /data/dengyz/code 2>&1 || true

DATA_ROOT=/data/dengyz/dataset/Code_10w_v1
RAW_ROOT=/data/dengyz/dataset/mapping_subject_dicom
CODE_ROOT=/data/dengyz/code/ismrm_v2_1
OLD_CODE_ROOT=/data/dengyz/code/Code_10w
FINAL5_ROOT=$DATA_ROOT/Code_10w_runs/phase4r_final_five_subject_k8_central_20261023

printf '\n===== roots =====\n'
for p in "$CODE_ROOT" "$OLD_CODE_ROOT" "$DATA_ROOT" "$RAW_ROOT" "$FINAL5_ROOT"; do
  if [ -e "$p" ]; then
    printf 'EXISTS %s\n' "$p"
    printf 'REAL   %s\n' "$(readlink -f "$p")"
    stat -c 'STAT   type=%F size=%s mtime=%y path=%n' "$p"
  else
    printf 'MISSING %s\n' "$p"
  fi
done

echo '===== subject roots ====='
for s in CYJ DYL_2 DYZ HHZ HJL; do
  echo "===== $s ====="

  if [ "$s" = HHZ ]; then
    PREPARED=$DATA_ROOT/Code_10w_prepared_scmr_v5/HHZ
    RUN_ROOT=$DATA_ROOT/Code_10w_runs/ismrm_v2_scmr_v5
    PREPROCESSED=$DATA_ROOT/Code_10w_preprocessed/HHZ
  elif [ "$s" = DYL_2 ]; then
    PREPARED=$DATA_ROOT/Code_10w_runs/scmr_v5_rebuild_20261102_corrected/$s/prepared/$s
    RUN_ROOT=$DATA_ROOT/Code_10w_runs/scmr_v5_rebuild_20261102_corrected/$s
    PREPROCESSED=$RUN_ROOT/preprocessed/$s
  else
    PREPARED=$DATA_ROOT/Code_10w_runs/scmr_v5_rebuild_20261102_corrected/$s/prepared/$s
    RUN_ROOT=$DATA_ROOT/Code_10w_runs/scmr_v5_rebuild_20261102_corrected/$s
    PREPROCESSED=$DATA_ROOT/Code_10w_preprocessed/$s
  fi

  for p in \
    "$RAW_ROOT/$s" \
    "$PREPROCESSED" \
    "$PREPARED" \
    "$RUN_ROOT/sparse_inputs/$s" \
    "$RUN_ROOT/native_cache" \
    "$RUN_ROOT/mapping" \
    "$RUN_ROOT/reconstruction/proposed/$s" \
    "$RUN_ROOT/reconstruction/fit_first/$s" \
    "$RUN_ROOT/evaluation/final_pose/$s" \
    "$RUN_ROOT/evaluation/common_native/$s" \
    "$DATA_ROOT/Code_10w_runs/ismrm_five_subject_analysis_v1/roi/derived/$s/20261026_scmr_native_roi_bundle"; do
    if [ -e "$p" ]; then
      printf 'EXISTS %s\n' "$p"
      printf 'REAL   %s\n' "$(readlink -f "$p")"
      stat -c 'STAT   type=%F size=%s mtime=%y path=%n' "$p"
    else
      printf 'MISSING %s\n' "$p"
    fi
  done

  if [ -d "$RAW_ROOT/$s" ]; then
    for stack in sax 2ch 4ch; do
      printf 'DICOM_COUNT %s/%s ' "$s" "$stack"
      find "$RAW_ROOT/$s/$stack" -maxdepth 1 -type f 2>/dev/null | wc -l
    done
  fi
done

echo '===== historical Final5 structure ====='
if [ -d "$FINAL5_ROOT" ]; then
  for d in 00_inventory 01_preprocessed 02_standardized_12_15_10 03_sparse_inputs 04_reconstruction 05_formal_endpoints 06_metrics 07_publication_visualization; do
    printf '\n--- %s ---\n' "$d"
    find "$FINAL5_ROOT/$d" -maxdepth 3 -type d -print 2>/dev/null | sort
  done
else
  echo "FINAL5_ROOT NOT FOUND: $FINAL5_ROOT"
fi

echo '===== key files under candidate roots ====='
find "$DATA_ROOT" -maxdepth 7 -type f \
  \( -name observations.npz -o -name sparse_manifest.json -o -name native_maps.npz \
     -o -name geometry.npz -o -name model.pt -o -name T1_3D.nii.gz \
     -o -name T2_3D.nii.gz -o -name volume_ms.nii.gz \
     -o -name roi_derivation_manifest.json -o -name sax_myocardium_masks.npz \) \
  -print 2>/dev/null | sort

echo '===== Git relationship ====='
git status --ignored --short -- subjects runs
for p in subjects runs; do
  if [ -e "$p" ]; then
    git check-ignore -v "$p" "$p/.gitkeep" 2>&1 || true
  fi
done
~~~

Interpretation rules:

- CURRENT_RUNTIME_REFERENCE is justified only when the path is present in current scripts/configs or the current formal server run.
- DOCUMENTED_HISTORICAL covers old Final5 and old standardization trees.
- INFERRED_FROM_STRUCTURE must not be treated as a confirmed server fact.
- SERVER_VERIFICATION_REQUIRED remains unresolved until the read-only command returns evidence.
- The existence of 02_standardized_12_15_10 does not reactivate slice standardization; the current scientific contract says it is inactive.
