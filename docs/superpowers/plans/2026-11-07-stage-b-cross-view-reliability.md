# Stage-B Cross-View Reliability Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the frozen raw-cosine, same-world Stage-B reliability gate to the isolated v2.2 Q002 route while preserving the v2.1 loss and all Stage-A behavior when disabled.

**Architecture:** A v2.2-specific reliability module will own validated configuration, detached gate math, per-stack normalization, fixed-geometry precomputation, and a compact cache/manifest. The Q002 loop will consume cache values by `JointAnchorSampler.anchor_idx`; route IDs will select independent stack-balance and reliability switches without changing the decoder, PSF, optimizer, or schedule.

**Tech Stack:** Python, PyTorch, NumPy, YAML/JSON/NPZ manifests, pytest.

**Spec:** `/home/universe/.codex/attachments/01dde67a-397e-44fd-92a6-cc9bb2bd7d66/pasted-text.txt`

## Global Constraints

- Use raw cosine discrepancy `1 - cosine(observed target, observed same-world SAX)`; normalized L1 is not the decision metric.
- Use fixed `group_axisangle_post_stack_init_physical`; never use trainable Stage-B pose or rerun HB1.
- Keep PSF, decoder, backbone, optimizer, iteration budgets, Stage-A, and Q003 prediction-vs-observation cosine objective unchanged/off for reliability routes.
- Invalid same-world reference support is neutral (`g=1`), never zero or rejected.
- Reliability is detached/no-grad and precomputed once; no per-step interpolation.
- Preserve stack-balance scale by normalizing reliability multipliers within each sampled stack.
- Local CPU validation only; do not access or launch server data/training; commit but do not push.

## Review Focus

- Disabled/all-one reliability must be numerically identical to the frozen `joint_vector_mse` path (tests A/B).
- Per-stack normalization must preserve each represented stack's mean multiplier and stack prefactor (test C).
- Stage A must ignore reliability and Stage-B annealing must be local 0.15→0 (tests E/F).
- Cache identity and fixed geometry must fail closed or remain invariant (tests H/I).
- Missing/invalid reference support and SAX-only runs must be neutral no-ops (tests D/J).

### Task 1: Add the v2.2 reliability core

**Files:**
- Create: `src/ismrm_v2/v2_2/stage_b_reliability.py`
- Test: `tests/v2_2/test_stage_b_reliability.py`

**Interfaces:**
- `ReliabilityConfig.from_mapping(mapping) -> ReliabilityConfig`
- `stage_b_progress(iteration, iterations) -> float`
- `gate_floor(progress, config) -> float`
- `compute_reliability_gate(discrepancy, reference_valid, stack_idx, tau_by_stack, progress, config) -> torch.Tensor`
- `normalize_reliability_gate(gate, stack_idx) -> torch.Tensor`
- `reliability_weighted_joint_mse(prediction, observed, stack_idx, stack_weights, gate, enabled, stage, stack_balance_enabled=True) -> tuple[torch.Tensor, dict[str, Any]]`
- `ReliabilityCache.validate_anchor_order(anchor_order_sha256)` and compact NPZ/JSON serialization helpers.

- [ ] Write tests covering A–K math, detachment, cache identity, and SAX-only neutral behavior.
- [ ] Implement validated config, detached raw-cosine gate formula, Q90/floor tau helper, per-stack normalization, and compact cache/provenance schema.
- [ ] Run `pytest -q tests/v2_2/test_stage_b_reliability.py`.

### Task 2: Integrate independent route switches and training consumption

**Files:**
- Modify: `src/ismrm_v2/proposed/q002/contracts.py`
- Modify: `src/ismrm_v2/proposed/q002/training.py`
- Modify: `src/ismrm_v2/proposed/q002/run_q002_reconstruction.py`
- Test: `tests/v2_2/test_stage_b_reliability.py`

**Interfaces:**
- Add route IDs `B0_R0`, `B1_R0`, `B0_R1`, `B1_R1`; B1_R0 is frozen v2.1 and B1_R1 is the primary candidate.
- Pass route reliability and stack-balance mappings into `train_q002_reconstruction` without changing legacy callers.
- Build fixed-geometry cache before optimization, consume it via `anchor_idx`, and return reliability provenance for manifests.

- [ ] Add route metadata/config defaults and CLI choices while preserving existing routes.
- [ ] Add the optional provider path in the Q002 loop; use the original objective for R0 and Stage A, and the normalized detached Stage-B objective for R1.
- [ ] Add lightweight training-log diagnostics and run-level reliability manifest/cache fields.
- [ ] Add route/integration assertions for Q003 cosine exclusion and no-op semantics.

### Task 3: Validate and commit

- [ ] Run focused v2.2 tests, full `pytest -q tests/v2_2`, compileall, and `git diff --check`.
- [ ] Run existing scientific-freeze, clean-refactor, project-integrity, and self-contained/no-legacy-path checks available locally.
- [ ] Review diff for server paths or training launches.
- [ ] Commit with `v2.2: integrate Stage-B cross-view reliability`; do not push.
