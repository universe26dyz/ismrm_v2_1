# 07 Final pose

Project both completed methods into one acquisition-native final pose. This stage does not retrain or reconstruct. It consumes completed volumes, fit-first native maps, prepared geometry and the subject ROI bundle.

`run_stage.sh` invokes the paired final-pose bundle for one completed case.

```bash
ROI_BUNDLE="$PROJECT_ROOT/data/roi/$SUBJECT/20261026_scmr_native_roi_bundle"
bash "$PROJECT_ROOT/pipeline/07_FINAL_POSE/run_stage.sh" \
  "$RUN_ROOT" "$SUBJECT" "$ROI_BUNDLE" L100_fold0
```

Continue only after native-cache and prepared-SAX provenance checks pass.
