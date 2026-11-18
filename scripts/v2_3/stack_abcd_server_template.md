# v2.3 stack-balance mechanism workflow — server template

Run this template only in the server environment after supplying its real paths.
The local worktree and `/data/dengyz/code/ismrm_v2_3` are different environments;
this template does not copy code or data between them. Confirm that the server
worktree contains the local commits before running it.

`stack_abcd_server_template.sh` runs A, the four new B/C training routes,
the B/C native evaluations, and D, in that order. It reuses the historical
EQUAL authorities. It does not generate timestamped output paths. Each output
must be absent; do not reuse output folders from previous attempts. Set
`SUBJECTS` to override the initial `CYJ DYZ HJL` list.

Export these paths before invoking the script:

- `PROJECT_ROOT`: server v2.3 worktree.
- `CASE_ROOT`: parent containing `<subject>/sparse_manifest.json` and the
  manifest-declared prepared stack files under that subject's directory.
- `OUTPUT_ROOT`: parent for new per-subject audit and training outputs.
- `NATIVE_ROOT`, `ROI_ROOT`: parents containing each subject's native-cache
  and clean ROI bundle directory, respectively.
- `MSE_EQUAL_ROOT`, `H10_EQUAL_ROOT`, `DETACH_EQUAL_ROOT`: parents containing
  each subject's existing completed historical training root.
- `B6_MODEL`, `DECODER`, `PROTOCOL`: exact frozen dependency files.
- `GENERATOR_SHA`: independently established sparse-generator commit.
- `SPARSE_PIN_ROOT`: directory containing independently established
  `<subject>.sha256` files; each starts with that subject’s manifest SHA256.
- `PYTHON` (optional): Python executable in the existing server environment;
  `DEVICE` (optional) defaults to `cuda:0`.

Example invocation after setting these values:

```bash
export SUBJECTS='CYJ DYZ HJL'
bash "$PROJECT_ROOT/scripts/v2_3/stack_abcd_server_template.sh"
```

The template checks each supplied sparse manifest against the independent pin,
and passes the independently established generator SHA to the strict validator.
It computes dependency SHA256 values from the supplied files; the training
launcher also enforces its existing immutable B6/decoder contracts. The
historical roots must include the deterministic-Torch `run.log` accepted by the
existing evaluator. The template stores each new log both beside and inside its
training root after successful completion.

The spatial audit evaluates actual checkpoint variance heads on native planes.
Its sigma is pointwise, not PSF-aggregated. Its coverage uses retained joint
anchors in original physical geometry; other-view counts exclude the queried
view. Correlations are diagnostic associations, not causal proof. The edge
abnormality and observed SAX/native structural mismatch remain separate
scientific questions.
