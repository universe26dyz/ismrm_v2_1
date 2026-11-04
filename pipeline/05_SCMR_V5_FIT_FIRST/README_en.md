# 05 SCMR-v5 fitting and 2D-fit-first

Run the SCMR-v5 MATLAB dictionary matcher and then independent T1/T2 NeSVoR/fit-first reconstruction on the server. The native-cache manifests must bind source observations, preprocessed MAT and generated maps by SHA256; this active route does not consume standardized native maps.

`run_stage.sh` invokes the server native-cache/fit-first launcher and retains per-case logs.

```bash
bash "$PROJECT_ROOT/pipeline/05_SCMR_V5_FIT_FIRST/internal/run_scmr_v5_fit_first.sh" \
  "$RUN_ROOT" "$RUN_ROOT/preprocessed" "$SUBJECT" "$RUN_ROOT/prepared" matlab
```

The expected completion marker is `SCMR-v5 Fit-first native maps and reconstructions: PASS`. Do not accept a cache with a different timing contract.
