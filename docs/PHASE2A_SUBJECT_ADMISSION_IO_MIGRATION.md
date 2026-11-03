# Phase 2A subject-admission I/O migration

Baseline: `e4a91778270e6df9ca973d39ed6efd713d741654`.

Phase 2A moves only two generic I/O implementations:

- `scripts/validate_subject_id.py` → `src/ismrm_v2/io/validate_subject_id.py`
- `scripts/build_subject_deployment_manifest.py` → `src/ismrm_v2/io/build_subject_deployment_manifest.py`

The validator is now invoked as `python -m ismrm_v2.io.validate_subject_id`
by Stage 01 through Stage 08, with repository-local `src` added to
`PYTHONPATH`. Stage 01 invokes the builder as
`python -m ismrm_v2.io.build_subject_deployment_manifest` with unchanged
arguments and manifest schema.

The active dependency inventory changes from 153 nodes / 664 edges to 153
nodes / 671 edges. The two script nodes are replaced by the two package nodes;
the seven net new edges record direct validator invocation for Stages 02--08.

Behavior tests cover generic valid and invalid subject identifiers, manifest
creation with local and explicit server roots, missing-stack and no-overwrite
failures, module CLI smoke tests, path authority, and shell syntax. The
scientific freeze contract remains unchanged: no Proposed, Fit-first, MATLAB,
AHA, metrics, decoder, B6, protocol, or scientific algorithm source was
modified.

`common/scripts/deployment_manifest.py` is deliberately deferred and was not
moved or modified in this phase.
