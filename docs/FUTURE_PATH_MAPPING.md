# Future path mapping

This is a Phase 1 migration map.  Each current runtime path remains active
until the corresponding import migration and validation are complete.

| Current path group | Future module | Action | Risk | Phase |
|---|---|---|---|---|
| Generic subject validator | `src/ismrm_v2/io/validate_subject_id.py` | **COMPLETED** — moved from `scripts/validate_subject_id.py` | generic admission only | 2A |
| Subject deployment-manifest builder | `src/ismrm_v2/io/build_subject_deployment_manifest.py` | **COMPLETED** — moved from `scripts/build_subject_deployment_manifest.py` | generic filesystem I/O only | 2A |
| Deployment-manifest loader contract | `src/ismrm_v2/io/deployment_manifest.py` | **COMPLETED** — moved from `common/scripts/deployment_manifest.py` | Stage 02/03 manifest contract | 2B |
| Stage 02 preprocessed-MAT validator | `src/ismrm_v2/preprocessing/validation.py` | **COMPLETED** — moved from `common/trad/scripts/validate_transferred_preprocessed.py` | Stage 02 validation only | 2C |
| Stage 03 observation preparation and data bridge | `src/ismrm_v2/observations/{prepare_all,build_manifest,mat_v73,timing,prepare_observations}.py` | **COMPLETED** — moved from the Stage 02 data bridge and Stage 03 wrapper | MAT v7.3 / observation-output contract | 2C |
| Stage 04 sparse sampling and profile manifest | `src/ismrm_v2/sampling/{sparse,sparse_profile_manifest}.py` | **COMPLETED** — moved from `phase2_clean/` | sparse selection and profile validation | 2C |
| Formal Proposed/Q002 scientific runtime | `src/ismrm_v2/proposed/` | **COMPLETED** — migrated active Q002, reconstruction, INR, decoder, PSF, training, inference, and export authorities | Q002S640 scientific freeze | 2D |
| SCMR-v5 Fit-first and NeSVoR Python route | `src/ismrm_v2/fit_first/` | **COMPLETED** — migrated active Fit-first authority | dictionary matching and independent T1/T2 route | 2D |
| Deferred geometry bridge | `src/ismrm_v2/observations/geometry.py` | **COMPLETED** — migrated from `module_02_data_bridge/geometry.py` | geometry contract shared across stages | 2D |
| Stage 07/08 evaluation and AHA runtime | `src/ismrm_v2/evaluation/` | **COMPLETED** — migrated active AHA, common-native, and evaluation authorities | ROI/AHA/common-native semantics | 2D |
| `phase2_clean/proposed.py` and Q002 | `src/ismrm_v2/proposed/` | Move implementation once | formal Proposed authority | 2 |
| `phase2_clean/baseline.py`, `fit_first*.py` | `src/ismrm_v2/fit_first/` | Move after CLI/import migration | SCMR-v5 and NeSVoR route | 2 |
| `common/mlp/` | `src/ismrm_v2/proposed/` | Merge later with TRAD only after validation | frozen decoder and checkpoint compatibility | 2 |
| `common/trad/` | domain package under `src/ismrm_v2/` | Move later | MATLAB/data-bridge dependencies | 2 |
| `common/reconstruction_core/` | `src/ismrm_v2/proposed/` | Collapse facade later | sole reconstruction engine | 2 |
| `aha/` | `src/ismrm_v2/evaluation/` | Rename/move later | Stage 07/08 output contracts | 2 |
| `server_commands/` | future pipeline CLI layer | Move later | server command semantics | 2 |
| `pipeline/01_*` through `pipeline/09_*` | future named pipeline directories | Rename later | formal entrypoints | 3 |

The detailed, node-level mapping is
[`clean_code_migration_manifest.json`](clean_code_migration_manifest.json).
