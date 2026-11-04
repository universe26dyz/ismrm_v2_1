# Clean-code architecture

`src/ismrm_v2/` is the single authority for project-owned Python runtime.
Formal pipeline shell entrypoints remain in `pipeline/`; they set a
repository-local `PYTHONPATH` containing `src` and retain their server-facing
CLI and output contracts.

```text
src/ismrm_v2/
  io/                 subject admission and deployment manifests
  preprocessing/      Stage 02 validation
  observations/       MAT/observation bridge and shared geometry
  sampling/           sparse-profile selection and manifests
  fit_first/          SCMR-v5 dictionary / independent T1,T2 NeSVoR route
  proposed/           Q002, decoder, INR, PSF, training, inference, export
  evaluation/         Stage 07/08 native, ROI, AHA, metrics, visualization
```

The frozen Proposed authority is `Q002S640_B6_roi_same_anchor_mse`: K8, 640
anchors, 10 weights, 6400 residuals, A2000/B4000, seed 20260911, and the
frozen decoder/B6/protocol bindings. The Fit-first route remains SCMR-v5
dictionary matching followed by independent T1 and T2 NeSVoR reconstruction.
No standardization route is selected by the formal Stage 05 wrapper.

`common/` is retained only for third-party NeSVoR, MATLAB, historical material,
and required shell/runtime support. Historical provenance names such as
`cyj11`, `hhz`, `final5`, and `phase2_clean` remain where they identify an
output or historical workflow; they are not cleanup criteria.

The MLP/TRAD quantitative-INR and rigid-PSF duplicates remain deferred. They
have independent compatibility-tree callers and freeze-contract evidence, so a
numerically validated merge is required before either implementation can be
removed. The node-level evidence is in
[`clean_code_migration_manifest.json`](clean_code_migration_manifest.json).
