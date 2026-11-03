# Phase 2B deployment-manifest migration

Baseline: `f9179829ef6ddbc3b48a0d25b2cc1c8236223141`.

Phase 2B moves the deployment-manifest loader contract only:

- `common/scripts/deployment_manifest.py` →
  `src/ismrm_v2/io/deployment_manifest.py`

`common/trad/scripts/validate_transferred_preprocessed.py` and
`common/trad/scripts/prepare_all_observations.py` now import the clean module.
Both retain repository-local `src` and `common` import setup for direct script
execution.

The loader remains byte-for-byte equivalent: it accepts only
`deployment-v1.1`, rejects ambiguous `deployment-v1`, requires explicit local
and server DICOM paths, preserves entry order, and never discovers filesystem
subjects or stacks.

The active dependency inventory remains 153 nodes / 671 edges. The old loader
node is replaced by the clean I/O node, which is `ACTIVE_TRANSITIVE` in Stage
02 and Stage 03. No scientific runtime source or dependency changed.

Deferred: the data bridge, MATLAB preprocessing, sampling, Fit-first,
Proposed, Q002, decoder, rigid PSF, Stage 07/08 evaluation, compatibility
imports, and duplicate implementations remain outside this phase.
