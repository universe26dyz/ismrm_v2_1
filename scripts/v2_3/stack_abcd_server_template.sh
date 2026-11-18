#!/usr/bin/env bash
# Server template. Export the paths and hashes documented in stack_abcd_server_template.md.
set -euo pipefail
: "${PROJECT_ROOT:?}" "${CASE_ROOT:?}" "${OUTPUT_ROOT:?}" "${NATIVE_ROOT:?}" "${ROI_ROOT:?}"
: "${B6_MODEL:?}" "${DECODER:?}" "${PROTOCOL:?}" "${GENERATOR_SHA:?}" "${SPARSE_PIN_ROOT:?}" "${MSE_EQUAL_ROOT:?}" "${H10_EQUAL_ROOT:?}" "${DETACH_EQUAL_ROOT:?}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/nesvor${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20260911
: "${PYTHON:=python}"
for SUBJECT in ${SUBJECTS:-CYJ DYZ HJL}; do
    PREPARED="$CASE_ROOT/$SUBJECT"
    SPARSE="$PREPARED/sparse_manifest.json"
    DEST="$OUTPUT_ROOT/$SUBJECT"
    mkdir -p "$DEST"
    read -r SPARSE_SHA _ < "$SPARSE_PIN_ROOT/$SUBJECT.sha256"
    if [ "$(sha256sum "$SPARSE" | cut -d' ' -f1)" != "$SPARSE_SHA" ]; then
        printf 'Sparse manifest pin mismatch for %s\n' "$SUBJECT" >&2; exit 1
    fi
    B6_SHA="$(sha256sum "$B6_MODEL" | cut -d' ' -f1)"
    DECODER_SHA="$(sha256sum "$DECODER" | cut -d' ' -f1)"
    PROTOCOL_SHA="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
    "$PYTHON" -m ismrm_v2.evaluation.v2_3_stack_balance_audit --prepared-root "$PREPARED" --sparse-manifest "$SPARSE" --output "$DEST/stack_audit"
    for ROUTE in B1_R0_STACK_OFF B1_R0_STACK_SLICE B1_R0_HGNLL_STACK_SLICE B1_R0_HGNLL_DETACH_STACK_SLICE; do
        LOG="$DEST/$ROUTE.run.log"
        "$PYTHON" -u scripts/v2_3/run_stack_policy_torch.py --experiment-id "$ROUTE" --anchor-batch-size 640 --subject-id "$SUBJECT" --cropped-prepared-root "$PREPARED" --sparse-manifest "$SPARSE" --expected-sparse-generator-git-sha "$GENERATOR_SHA" --expected-sparse-manifest-sha256 "$SPARSE_SHA" --b6-model "$B6_MODEL" --expected-b6-model-sha256 "$B6_SHA" --signal-simulator "$DECODER" --expected-decoder-sha256 "$DECODER_SHA" --protocol "$PROTOCOL" --expected-protocol-sha256 "$PROTOCOL_SHA" --output "$DEST/$ROUTE" --device "${DEVICE:-cuda:0}" 2>&1 | tee "$LOG"
        cp "$LOG" "$DEST/$ROUTE/run.log"
    done
    "$PYTHON" -m ismrm_v2.evaluation.v2_3_stack_policy_native_metrics --mode mse --method "mse_stack_off=$DEST/B1_R0_STACK_OFF" --method "mse_stack_equal=$MSE_EQUAL_ROOT/$SUBJECT" --method "mse_stack_slice=$DEST/B1_R0_STACK_SLICE" --r0-native-maps-root "$NATIVE_ROOT/$SUBJECT" --sparse-manifest "$SPARSE" --roi-bundle "$ROI_ROOT/$SUBJECT" --output "$DEST/mse_native_eval"
    "$PYTHON" -m ismrm_v2.evaluation.v2_3_stack_policy_native_metrics --mode hgnll --method "h10_equal=$H10_EQUAL_ROOT/$SUBJECT" --method "h10_slice=$DEST/B1_R0_HGNLL_STACK_SLICE" --method "detach_equal=$DETACH_EQUAL_ROOT/$SUBJECT" --method "detach_slice=$DEST/B1_R0_HGNLL_DETACH_STACK_SLICE" --r0-native-maps-root "$NATIVE_ROOT/$SUBJECT" --sparse-manifest "$SPARSE" --roi-bundle "$ROI_ROOT/$SUBJECT" --output "$DEST/hgnll_native_eval"
    "$PYTHON" -m ismrm_v2.evaluation.v2_3_spatial_uncertainty_geometry_audit --mse-root "$MSE_EQUAL_ROOT/$SUBJECT" --h10-root "$H10_EQUAL_ROOT/$SUBJECT" --detach-root "$DETACH_EQUAL_ROOT/$SUBJECT" --prepared-root "$PREPARED" --r0-native-maps-root "$NATIVE_ROOT/$SUBJECT" --sparse-manifest "$SPARSE" --roi-bundle "$ROI_ROOT/$SUBJECT" --output "$DEST/spatial_audit" --device "${DEVICE:-cuda:0}"
done
