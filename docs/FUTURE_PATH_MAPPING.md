# Final path mapping

All completed project-owned Python runtime moves now terminate in
`src/ismrm_v2/`. The node-level old-to-new SHA256 records are maintained in
[`clean_code_migration_manifest.json`](clean_code_migration_manifest.json).

| Runtime group | Final authority | Status |
|---|---|---|
| Subject I/O and deployment manifests | `src/ismrm_v2/io/` | completed 2A/2B |
| Preprocess validation and observations bridge | `src/ismrm_v2/preprocessing/`, `src/ismrm_v2/observations/` | completed 2C/2D |
| Sparse profiles and sampling | `src/ismrm_v2/sampling/` | completed 2C |
| SCMR-v5 Fit-first and NeSVoR Python route | `src/ismrm_v2/fit_first/` | completed 2D/final cleanup |
| Q002/Proposed, decoder, INR, PSF, training, inference, export | `src/ismrm_v2/proposed/` | completed 2D/final cleanup |
| Stage 07/08 common-native, ROI, AHA, metrics, visualization | `src/ismrm_v2/evaluation/` | completed 2D |
| Pipeline formal shell entrypoints | `pipeline/` | retained; wrapper paths only |
| MATLAB preprocessing/SCMR matcher and third-party NeSVoR | `common/trad/` | intentionally retained |

The final cleanup removes unused Q003 launcher/evaluator modules and moves the
retained compatibility/legacy call targets out of `common/`. No project-owned
active scientific Python authority remains under `common/`.

MLP/TRAD duplicate implementations are explicitly deferred rather than merged:
their numerical merge requires independent validation and is outside the frozen
runtime refactor.
