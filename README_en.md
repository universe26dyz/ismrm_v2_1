# ismrm_v2_1

This is the frozen ISMRM submission baseline for the corrected HHZ SCMR-v5 MultiMap route. Operator-facing code is ordered in `pipeline/01_` through `pipeline/09_`; project Python lives in `src/ismrm_v2/`, MATLAB in `matlab/`, and vendored NeSVoR in `third_party/nesvor/`. `configs/` contains protocol and manifest templates. Server-local project assets are `data/raw_dicom/`, `data/roi/`, `runtime/model_assets/`, and `runs/`; patient data, model binaries, and results are ignored by Git.

## Self-contained project data contract

Use only paths derived from `PROJECT_ROOT`: raw DICOM is in `data/raw_dicom/<SUBJECT>`, final ROI bundles are in `data/roi/<SUBJECT>/20261026_scmr_native_roi_bundle`, and frozen v2.1 reference products are in `runs/frozen_baseline/<SUBJECT>`. The frozen baseline is read-only. New executions must use `runs/reproduction/<SUBJECT>/run_<timestamp>`; v2.2 server-local outputs belong under `runs/v2_2/`.

Run all new experiments from this root, with explicit `SUBJECT`, run roots and manifests. The server performs DICOM preprocessing, observation preparation, SCMR-v5 MATLAB fitting, 2D-fit-first, Proposed, final-pose and metrics. Local work is limited to manual ROI/mask drawing and copying the ROI bundle to the server.

The immutable contract is TR 2.61 ms, VPS 87, TI `[10,100]`, T2-preparation `[35,45,55]`, FA `[45,45,45]`, ramp-up 10, ten weights per group. Proposed accepts only decoder SHA `338f2c...`; the historical `[50,150]` decoder is invalid. The active submission cohort is `CYJ`, `DYL_2`, `DYZ`, `HHZ`, `HJL`; legacy `DYL` is rejected.

The active route never uses 12/15/10 slice standardization. Proposed starts from prepared signal observations and sparse profiles, while fit-first performs SCMR-v5 dictionary matching followed by independent T1/T2 NeSVoR reconstructions. Historical code remains for provenance only. New method development must fork/branch from this freeze rather than changing its scientific definition.

Start a new run with `RUN_ROOT="$PROJECT_ROOT/runs/reproduction/$SUBJECT/run_$(date +%Y%m%d_%H%M%S)"` and `MANIFEST="$PROJECT_ROOT/subjects/$SUBJECT/deployment_manifest.json"`, then follow modules 01–09. External dependencies are software/runtime only: MATLAB, CUDA/NVIDIA, the Python environment, and required executables.
