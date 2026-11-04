# Final server deployment commands

Run these commands manually after pushing the reviewed local `main` commit.
They operate in place at `/data/dengyz/code/ismrm_v2_1`; they never replace the
repository directory, run `git clean`, or delete `subjects/`, `runs/`, or
`runtime/model_assets/`.

```bash
cd /data/dengyz/code/ismrm_v2_1
git status --short
git fetch origin --tags
git switch main
git pull --ff-only origin main
git rev-parse HEAD
git rev-parse refs/tags/ismrm-v2.1-frozen-baseline^{commit}
```

Confirm that the protected server-local state remains present before running
any pipeline command:

```bash
test -d subjects
test -d runs
test -d runtime/model_assets
git status --short
```

Run the tracked-code checks in place:

```bash
python scripts/verify_server_local_state_policy.py
python scripts/verify_clean_refactor_invariants.py
python scripts/verify_scientific_freeze.py --root .
python scripts/verify_project_integrity.py .
python scripts/verify_no_legacy_code_path.py .
python scripts/build_active_dependency_inventory.py --root . --json /tmp/ismrm_final_inventory.json --markdown /tmp/ismrm_final_trace.md
```

Only after the checks pass, run the site-specific MATLAB preprocessing and
SCMR-v5 matcher smoke checks, NeSVoR Stage 05 smoke check, and CUDA Stage 06/07
formal validation using the existing protected `subjects/`, `runs/`, and model
assets. Do not run `git clean -fd`, `git clean -fdx`, directory replacement, or
any command that deletes those protected paths.
