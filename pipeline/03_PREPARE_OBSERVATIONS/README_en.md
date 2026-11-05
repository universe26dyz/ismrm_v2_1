# 03 Prepare observations

This bridge reads the MAT outputs, DICOM identities and the SCMR-v5 10/100 protocol and writes `observations.npz` plus per-stack manifests. The NPZ is generated here, not supplied as raw input.

`run_stage.sh` invokes `prepare_all_observations.py` with explicit manifest and protocol paths.

```bash
bash "$PROJECT_ROOT/pipeline/03_PREPARE_OBSERVATIONS/run_stage.sh" \
  "$RUN_ROOT" "$SUBJECT" \
  "$PROJECT_ROOT/subjects/$SUBJECT/deployment_manifest.json" "$PYTHON"
```

Manifests must bind TR/VPS/TI/T2-prep, ten weight indices and source SHA256. Treat any source-observation mismatch as a provenance error.
