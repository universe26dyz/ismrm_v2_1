# Future server code-update procedure

This document describes a future safe update; it does not authorize a server
operation during Phase 1.

## Do

1. Enter the existing repository directory and inspect `git status`.
2. Fetch/pull Git-tracked changes using a fast-forward-only update.
3. Preserve `subjects/`, `runs/`, `runtime/model_assets/`, and root audit logs.
4. Validate the scientific freeze, runtime contracts, and model hashes after
   the tracked-code update.

## Do not

Do not remove the repository directory, run destructive `git clean` variants,
replace `runtime/` wholesale, or clone into an empty replacement after deleting
the existing repository.  These operations can destroy ignored archives and
model binaries.

## Precondition

The server-local state policy and `.gitignore` protections must be present
before any update.  Deployment changes tracked code in place only.
