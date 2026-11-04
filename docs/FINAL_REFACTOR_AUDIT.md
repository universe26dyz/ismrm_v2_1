# Final refactor audit

Baseline: `71ece1ab12ba1508aa585f738c34afac3b7b162a`.

## Final authority tree

Project-owned Python runtime authority is under `src/ismrm_v2/` in the I/O,
preprocessing, observations, sampling, Fit-first, Proposed, and evaluation
packages. Formal `pipeline/` shell wrappers remain in place and execute the
same stage contracts using repository-local imports. `common/` contains only
third-party NeSVoR, MATLAB, historical/non-runtime material, and shell/runtime
support; it contains no active project-owned scientific Python authority.

## Cleanup record

- Moved Stage 05 compatibility modules to `src/ismrm_v2/fit_first/`:
  `fit_first_intensity_preserving.py`,
  `reuse_standardized_native_dictionary_maps.py`, and
  `standardize_subject_slices.py`. These preserve compatibility behavior but
  are not selected by the formal standardization-free route.
- Moved the retained historical Q004 launcher to
  `src/ismrm_v2/proposed/q002/run_q004_s640_k1_reconstruction.py` and updated
  its two historical facades by module path only.
- Deleted the audited-unused Q003 launcher and 10k evaluator; their only
  callers were obsolete test assertions, which were removed without changing
  Q002/Proposed behavior.
- Deferred the MLP/TRAD quantitative-INR and rigid-PSF duplicates. They still
  require an independently validated numerical merge and are not removed by a
  path-only refactor.

## Inventory and freeze

Inventory changed from 153 nodes / 671 edges to 151 nodes / 671 edges. The two
node removals are the audited-unused Q003 modules; the other cleanup paths are
moves. The inventory generator reports no missing internal active dependency.
Migration-manifest coverage is 151 of 151 current inventory nodes.

The scientific freeze remains Q002S640_B6_roi_same_anchor_mse, K8, 640
anchors, 10 weights, 6400 residuals, A2000/B4000, seed 20260911, and the frozen
decoder/B6/protocol SHA256 bindings. Fit-first SCMR-v5 matching, independent
T1/T2 NeSVoR settings, ROI/AHA/common-native metrics, decoder gradient flow,
and PSF semantics are unchanged.

## Deferred execution

MATLAB, CUDA, NeSVoR, and server-data execution are intentionally not run on
the local CPU environment. Deployment validation is documented in
[`FINAL_SERVER_DEPLOYMENT_COMMANDS.md`](FINAL_SERVER_DEPLOYMENT_COMMANDS.md).
The integrity checker permits a source-only local checkout without ignored
`subjects/` or `runs/`; the deployment commands require those protected server
directories before formal execution.

## Local test status

The formal targeted subject/I/O, observations, sampling, Q002/Proposed,
Fit-first, geometry, evaluation/AHA, inventory, migration, and freeze suites
pass locally. A best-effort whole-repository pytest aggregation runs with
`--import-mode=importlib` and excludes third-party MDM tests; it is blocked by
86 failures in retained historical MLP/TRAD tests that assume their original
per-subtree current working directory and untracked legacy artifact paths.
Those tests are outside the formal runtime inventory and are not rewritten to
change scientific or historical behavior for local aggregation.
