# Phase 2D scientific runtime migration

Phase 2D relocates the active scientific Python authority recorded by the
Phase 2C inventory from `common/` into `src/ismrm_v2/`.  Baseline:
`861b0c4898476fe91c227ec1424a67648d71ef13`.

## Authority groups

- `src/ismrm_v2/proposed/`: 26 files — formal Proposed/Q002 route,
  reconstruction authority, quantitative INR, frozen decoder, rigid PSF,
  training, inference, export, and profiling.
- `src/ismrm_v2/fit_first/`: 12 files — SCMR-v5 dictionary matching,
  Fit-first route controls, and independent T1/T2 NeSVoR Python authority.
- `src/ismrm_v2/observations/geometry.py`: deferred shared geometry bridge.
- `src/ismrm_v2/evaluation/`: 39 files — Stage 07/08 AHA, common-native,
  myocardial, and evaluation runtime.

All 78 moves are recorded with pre/post source SHA256 values and exact
old-to-new paths in [`clean_code_migration_manifest.json`](clean_code_migration_manifest.json)
as `MIGRATED_PHASE_2D`.  The source move is a `git mv`; changes inside moved
files are only import, executable-module-path, or repository-local resource-path adaptations.

## Caller adaptation

174 dependent runtime, compatibility, wrapper, and test callers now resolve
the clean-package authority and are recorded as `IMPORT_PATH_ADAPTATION_ONLY`.
No compatibility-only module was deleted, no MLP/TRAD duplicate was merged,
and historical `cyj11`, `hhz`, `final5`, and `phase2_clean` names remain where
they are not the migrated active authority.

The Stage 05, 07, and 08 wrappers remain in their existing locations and pass
repository-local `PYTHONPATH` containing both `src` and `common` to their
unchanged shell workflows.

## Freeze and inventory

The scientific contracts remain unchanged: Q002S640_B6_roi_same_anchor_mse,
K8, 640 anchors, 10 weights, 6400 residuals, A2000/B4000, seed 20260911, and
the frozen decoder/B6/protocol SHA256 bindings.  Fit-first dictionary matching,
independent T1/T2 NeSVoR settings, ROI/AHA/common-native metrics, decoder
gradient flow, and PSF semantics are unchanged.

Inventory is 153 nodes / 671 edges before and after; all edge changes are
module-path replacements.  Missing internal active dependencies remain zero,
and migration-manifest coverage remains 100% (153 of 153 current nodes).

MATLAB, CUDA, NeSVoR, and server-data execution are not run locally.  The
historical Q002 K256 shell-wrapper test is marked `TEST_ONLY_FIX` skipped:
the asserted wrapper files do not exist anywhere in this repository, so this
migration does not fabricate replacement scripts.
