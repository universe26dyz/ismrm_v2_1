# Proposed code deletions

Phase 1 performs **zero** Git-tracked runtime-code deletions.

## SAFE_AFTER_NEXT_VALIDATION

Only nodes already classified `LEGACY_UNUSED_BY_PIPELINE` or `TEST_ONLY` are
candidates after a later validation.  They remain present in Phase 1 and are
recorded as `AUDITED_UNUSED_CANDIDATE` in the migration manifest.

## SAFE_AFTER_IMPORT_DECOUPLING

`fit_first_intensity_preserving.py`,
`reuse_standardized_native_dictionary_maps.py`, and
`standardize_subject_slices.py` remain compatibility imports.  They may only
be removed after all importing callers are decoupled; none is deleted now.

## DO_NOT_DELETE_YET

All active closure nodes, historical-name active modules, MLP/TRAD duplicate
implementations, the MLP trainer facade, Stage 07/08 code, Q002, and SCMR-v5
mapping are retained.  Historical names and duplicate implementations are not
evidence that a path is safe to delete.
