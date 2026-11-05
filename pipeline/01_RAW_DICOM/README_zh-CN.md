# 01 原始 DICOM

## 作用与顺序

这是第一个模块，只负责确认三种 stack 的原始输入和部署清单；不修改 DICOM。必须先于 MATLAB 预处理执行。

## 文件清单

`run_stage.sh`：生成单 subject manifest，并检查三种 stack 的 DICOM 数量。

## 输入与输出

- 输入：`sax/`、`2ch/`、`4ch/` 三个服务器 DICOM 目录；每个 group 必须恰好十个 weight。
- 输出：`subjects/<SUBJECT>/deployment_manifest.json` 以及 DICOM 数量检查日志。
- 运行位置：服务器。原始 DICOM 不需要上传到 GitHub。

## 操作

```bash
PROJECT_ROOT=/path/to/ismrm_v2_1
SUBJECT=NEW_SUBJECT_001
RAW_ROOT="$PROJECT_ROOT/data/raw_dicom"
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python -m ismrm_v2.io.build_subject_deployment_manifest \
  --raw-root "$RAW_ROOT" --subject "$SUBJECT" \
  --output "$PROJECT_ROOT/subjects/$SUBJECT/deployment_manifest.json"
for s in sax 2ch 4ch; do
  find "$RAW_ROOT/$SUBJECT/$s" -maxdepth 1 -name '*.dcm' | wc -l
done
```

清单中的 `local_dicom_dir` 和 `server_dicom_dir` 从实际 stack 目录解析为绝对路径。不要用 NIfTI 代替 DICOM，因为 UID、采集时间和空间几何会在后续使用。

## 文件说明与故障排除

`build_subject_deployment_manifest.py` 只从指定 subject 的 `sax/2ch/4ch` 实际目录生成清单。数量不是 10 的整数倍、路径不存在或 stack 缺失时不要进入 02。
