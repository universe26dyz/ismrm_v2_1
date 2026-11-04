# Phase 2C foundation runtime migration

Phase 2C moved the low-risk Stage 02/03 Python data bridge and Stage 04
sampling authority into `src/ismrm_v2/`.  Baseline: `3ca73f9e1c3c141ebb1a7af5f0d70362f7e72548`.
No MATLAB, server-local data, CUDA, NeSVoR, Proposed, Fit-first algorithm, AHA,
metrics, decoder, reconstruction, or `geometry.py` implementation was moved.

## Moved authorities

| Old path | New path | Pre SHA256 | Post SHA256 | Classification |
|---|---|---|---|---|
| `common/trad/scripts/validate_transferred_preprocessed.py` | `src/ismrm_v2/preprocessing/validation.py` | `93400c8fda77a8dbd1f5b46c827b11765743e88e45b45df771775b2a60be2450` | `541d86a5d86dc9bdecc361423bf699bb536e1faf3bcca2e453f57d7f1d6a6b07` | MOVE_WITH_IMPORT_ADAPTATION |
| `common/trad/scripts/prepare_all_observations.py` | `src/ismrm_v2/observations/prepare_all.py` | `43175da3871b7cf28f198d620387a52355cd1fc82425d83c1ed398b70467741b` | `af03e2f0b9d097e368527a440d0d2b4c43e2a16da7eb685d63bf0b7d839d3a33` | MOVE_WITH_IMPORT_ADAPTATION |
| `common/trad/modules/module_02_data_bridge/build_manifest.py` | `src/ismrm_v2/observations/build_manifest.py` | `f6eec3c122ef699abfcc100ccf24708107fbe25d47c23088bb224e736bb08b5a` | `934e43c4b4e3ce90bb2354fa631625be8c21c41e8803cd75d750f0b9d476d437` | MOVE_WITH_IMPORT_ADAPTATION |
| `common/trad/modules/module_02_data_bridge/mat_v73.py` | `src/ismrm_v2/observations/mat_v73.py` | `6b6ea2922fdd3b2f306263acb25a96b2bbcbf5972e0dd29d558075ea45ea9e23` | `6b6ea2922fdd3b2f306263acb25a96b2bbcbf5972e0dd29d558075ea45ea9e23` | MOVE |
| `common/trad/modules/module_02_data_bridge/timing.py` | `src/ismrm_v2/observations/timing.py` | `4670dce30fe404fd0e1811c8aad56b9b1aa981c7223a4f3c046aa14a25c60c8f` | `4670dce30fe404fd0e1811c8aad56b9b1aa981c7223a4f3c046aa14a25c60c8f` | MOVE |
| `common/trad/modules/module_02_data_bridge/prepare_observations.py` | `src/ismrm_v2/observations/prepare_observations.py` | `53bc69ea6d5d2053518f00bdee7fde2c92856d1a1da9573a7ae51745642a6e85` | `53bc69ea6d5d2053518f00bdee7fde2c92856d1a1da9573a7ae51745642a6e85` | MOVE |
| `common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse.py` | `src/ismrm_v2/sampling/sparse.py` | `04d030ca77e2ecbcb5e6744bdbc18e0e518734a4655adb0e38428d9326abfec6` | `f95be554f97d4795b5f5d8c56fd1bfaa9f0ba0414a50c011bf6ccbab889ae0c4` | MOVE_WITH_IMPORT_ADAPTATION |
| `common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse_profile_manifest.py` | `src/ismrm_v2/sampling/sparse_profile_manifest.py` | `ef903d6649dcb8be7f3f3eaeaf4935c6e1ec1dde489bfaff497e6dc00e65e497` | `53b31de20fa60d424fc4a8479e5cb748d81ef6a66e76c26158c6d3b3ceeca0bd` | MOVE_WITH_IMPORT_ADAPTATION |

The Stage 02, 03, and 04 wrappers now invoke the clean package with
`PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/common${PYTHONPATH:+:$PYTHONPATH}"`.

## Import-only caller adaptations

The following callers changed only the named import path, from the former
`phase2_clean` sampling module to the corresponding `ismrm_v2.sampling`
authority.  Saved pre-migration copies were diffed; every diff contains only
its import line.

- `common/quality_experiments/Q002_B6_roi_same_anchor_mse/contracts.py`
- `common/quality_experiments/Q002_B6_roi_same_anchor_mse/run_q002_reconstruction.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/baseline.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_intensity_preserving.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/formal_l100_sparse_manifest.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/reuse_standardized_native_dictionary_maps.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/standardize_subject_slices.py`

All eight are `IMPORT_PATH_ADAPTATION_ONLY`.  In particular, Q002 constants,
K8/640 anchors/10 weights/6400 residuals, A2000/B4000, sparse validation,
formal provenance checks, CUDA/training/export behavior, function bodies, and
control flow are unchanged.

## Inventory and verification

The active dependency graph remains 153 nodes / 671 edges before and after the
migration, with zero missing internal active dependencies.  The old sampling
paths are absent; the new sampling paths carry the equivalent Stage 04--08
roles.  Focused tests cover MAT v7.3 loading, timing parsing, observation
outputs, deployment-manifest preparation, sparse profiles, SAX retention,
retained/held-out partitioning, and L100/L50 behavior.

`common/trad/modules/module_02_data_bridge/geometry.py` remains in place.
Scientific-core Proposed, Fit-first implementation, reconstruction, decoder,
rigid PSF, and Stage 07/08 evaluation/AHA modules remain deferred.
