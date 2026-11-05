# 08 评估与可视化

metrics、AHA、slice-pair 和 QC 必须使用 07 的 final-pose 输出。LV 使用 `outer_filled`（圆圈及内部），心肌同时保留 `myocardium_core_1px` 与 `myocardium_core_legacy`，血池使用 legacy 内膜填充。统计要区分 all-slice、held-out-only 和 stack，并输出 voxel count、mean、SD、RMSE、MAE、NMAE、bias、CC/NCC、PSNR 及 zero audit。

`run_stage.sh`：按 case 先做 common-native metrics/QC，再做 all-stack ROI metrics。

AHA16、final-five publication figures 与 multi-subject aggregation 属于可选 publication reporting，不是 Stage 08/09 自动生成的输出。

```bash
ROI_BUNDLE="$PROJECT_ROOT/data/roi/$SUBJECT/20261026_scmr_native_roi_bundle"
bash "$PROJECT_ROOT/pipeline/08_EVALUATION_AND_VISUALIZATION/run_stage.sh" \
  "$RUN_ROOT" "$SUBJECT" "$ROI_BUNDLE" L100_fold0
```

图像与 metrics 必须引用同一 final-pose mask；若 shape 不同，停止并检查 plane/pose，不要裁剪数组来掩盖错误。
