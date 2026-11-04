# Phase 3 common retirement

`common/` has been removed after the active dependency inventory reached zero
nodes beneath that path. Runtime assets moved without algorithm changes:

- 10 sparse-profile JSON files: `configs/sparse_profiles/`.
- Five stage shell helpers: their owning `pipeline/*/internal/` directories.
- MATLAB preprocessing and SCMR-v5 matcher trees: `matlab/preprocessing/` and
  `matlab/scmr_v5/`.
- Vendored NeSVoR: `third_party/nesvor/`.
- Formal reconstruction YAML assets: `configs/proposed/reconstruction/`.

The remaining `common/` contents were historical, compatibility, duplicate, or
test-only material already decoupled from the active graph and were deleted.
The generated final graph records 149 nodes and 671 edges, with no active
`common/` node or runtime reference. MATLAB, CUDA, NeSVoR, and server-data
execution were not run locally.
