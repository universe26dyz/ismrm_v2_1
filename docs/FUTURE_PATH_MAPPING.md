# Future path mapping

This is a Phase 1 migration map.  Each current runtime path remains active
until the corresponding import migration and validation are complete.

| Current path group | Future module | Action | Risk | Phase |
|---|---|---|---|---|
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
