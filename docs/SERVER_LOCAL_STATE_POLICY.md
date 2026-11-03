# Server-local state policy

## Protected server-local state

The following paths are operational state, not refactor inputs:

- `subjects/`: archived subject inputs, manifests, checksums, and provenance;
- `runs/`: formal reconstruction and evaluation outputs;
- `runtime/model_assets/*.pt` and `runtime/model_assets/*.pth`: frozen model binaries;
- repository-root `materialize*.log` and `verify*.log`: preserve during refactor, optional archive later.

They must not be moved, deleted, renamed, overwritten, migrated, or copied into Git by a code refactor.

## Deployment rule

Update Git-tracked code, configurations, and documentation in place.  Preserve
`subjects/`, `runs/`, model assets, and root audit logs exactly as they are.
The policy intentionally uses repository-relative paths; it does not make a
server absolute path part of runtime logic.

## Forbidden operations

Never use destructive repository replacement, `git clean` with either `-fd`
or `-fdx`, or wholesale replacement of `runtime/`.  In particular, a future
deployment must not remove the repository root before updating tracked files.

## Required invariant

Every code refactor and server deployment must preserve subject archives, run
outputs, and model assets unchanged.  The machine-readable companion is
[`configs/server_local_state_policy.json`](../configs/server_local_state_policy.json).
