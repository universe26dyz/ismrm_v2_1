# 06 Proposed

Proposed 使用原始 sparse observations、正确的 `[10,100]` decoder、B6、protocol 和 sparse manifest；它不读取 fitting map 作为输入。L100 必须使用 `$RUN_ROOT/prepared/$SUBJECT`，其他 case 使用对应 sparse case root。输出 root 必须不存在，避免 formal sparse execution 的 `fresh output root` 错误。

`run_stage.sh`：每次只执行一个 case，并在调用前检查 manifest 和 fresh output root。

```bash
bash "$PROJECT_ROOT/pipeline/06_PROPOSED/run_stage.sh" \
  "$RUN_ROOT" "$SUBJECT" "$CASE" "$PYTHON"
```

必须传入 decoder、B6、protocol、sparse manifest 和四个 expected SHA。完成后每 case 应有 T1/T2 NIfTI、experiment manifest 和 run provenance。
