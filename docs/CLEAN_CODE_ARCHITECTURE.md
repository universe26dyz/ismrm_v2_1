# Clean-code target architecture

## Phase 1 status

This is a target architecture, not an implementation migration.  Formal
runtime remains on the audited existing paths until a later, validated move.
The Phase 1 `src/ismrm_v2` package is intentionally an import-free skeleton
and is not a second scientific authority.

## Target package

```text
src/ismrm_v2/
  io/ preprocessing/ observations/ sampling/ fit_first/
  proposed/ evaluation/ integrity/
```

The future pipeline names are `01_INPUT`, `02_PREPROCESS`,
`03_OBSERVATIONS`, `04_SAMPLING`, `05_FIT_FIRST`, `06_PROPOSED`,
`07_NATIVE_PROJECTION`, `08_EVALUATION`, and `09_INTEGRITY`.  The current
`pipeline/01_RAW_DICOM` through `pipeline/09_INTEGRITY_AND_REPORTING`
entrypoints remain authoritative during Phase 1.

## Proposed target

`proposed/` will eventually own `dataset.py`, `quantitative_inr.py`,
`signal_decoder.py`, `rigid_psf.py`, `losses.py`, `training.py`, `export.py`,
and `cli.py`.  Its source mapping begins with Q002, `common/mlp/`,
`common/trad/`, `common/reconstruction_core/`, and
`phase2_clean/proposed.py`; Phase 1 moves none of them.

The frozen formal authority remains `Q002S640_B6_roi_same_anchor_mse`: PSF 8,
anchor batch 640, 10 weights per anchor, 6400 residuals, Stage A 2000, Stage B
4000, seed 20260911, HashGrid, shared quantitative decoder, T1/T2/B1/amplitude,
group-shared rigid pose, PSF-aware forward model, frozen signal MLP, and
signal-space loss.

## Fit-first target

`fit_first/` will contain `native_mapping.py`, `native_cache.py`, `nesvor.py`,
`reconstruction.py`, `geometry.py`, and `cli.py`.  It maps from `baseline.py`,
`scmr_v5_native_cache.py`, `fit_first*.py`, SCMR MATLAB mapping, and NeSVoR.
The formal route remains SCMR-v5 native dictionary matching to native T1/T2/B1,
followed by independent T1 and independent T2 NeSVoR reconstruction.  The
inactive standardized-slice route is not a target runtime route.

## Evaluation target

`evaluation/` will contain native projection, common support, ROI, AHA16,
metrics, visualization, and CLI modules.  Historical `cyj11`, `final5`,
`hhz`, `legacy`, and `paired` names are migration labels, never deletion
criteria.  Stage 07/08 implementations are unchanged in Phase 1.

## Migration rule

Phase 2 must move an implementation once, update imports, validate, and only
then delete the old path.  Duplicate MLP/TRAD quantitative INR and rigid-PSF
implementations remain intact until a validated merge; the MLP trainer facade
remains intact until its imports are decoupled.
