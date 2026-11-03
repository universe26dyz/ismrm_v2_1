# ISMRM Server Data Layout

审计日期：2026-11-03
审计范围：Code_10w / ismrm_v2_1 的五个正式 subject：CYJ、DYL_2、DYZ、HHZ、HJL

## Scope and evidence policy

本次 Codex 运行环境不能访问 CMRServer04 的 /data 文件系统。因此本文档不是一次实时服务器扫描，而是依据：

1. 当前 Git repository 中的代码、配置和 README；
2. 当前会话中已经返回的服务器终端结果；
3. 当前 repository 的 Git history。

状态只使用以下分类：

- CURRENT_RUNTIME_REFERENCE：当前代码或本次会话已经明确用于正式路线的路径；
- DOCUMENTED_HISTORICAL：旧文档、旧 handoff 或旧实验记录中的路径；
- INFERRED_FROM_STRUCTURE：由当前目录模式推断，尚未直接验证；
- SERVER_VERIFICATION_REQUIRED：本地证据不足，必须在服务器只读核验；
- DEPRECATED：已明确不属于当前 active scientific route 的路径。

除本文档和核验命令外，本次没有连接服务器、移动数据、复制数据、删除数据、运行 MATLAB/GPU、修改 ROI/metrics 或修改 scientific pipeline。

## Repository examined

| Item | Value | Evidence |
|---|---|---|
| Local repository | /home/universe/SVR/multimap_postprogramming/ismrm_v2_1 | Local Git repository |
| Git remote | https://github.com/universe26dyz/ismrm_v2_1.git | git remote -v |
| HEAD | 4523e16d5b3061ba04becee2025ad18d33008ff8 | Local Git |
| Branch | main | Local Git |
| Historical server code root | /data/dengyz/code/Code_10w | Session history; historical |
| Current server project root | /data/dengyz/code/ismrm_v2_1 | Session history; server verification should be repeated |
| Server host | CMRServer04 | Session history; not reconnected during this audit |

The repository is intended to be the code/deployment project. Large subject inputs and outputs are external runtime data and are not part of the GitHub code snapshot.

## Active scientific route

The frozen active route is:

~~~text
01 raw DICOM
02 MATLAB preprocessing -> preprocessed.mat
03 prepared observations -> observations.npz, timing.npy, manifests, QC
04 sparse profiles -> sparse_inputs/<SUBJECT>/<CASE>
05 SCMR-v5 native cache + Fit-first reconstruction
06 Proposed reconstruction
07 final-pose bundles and AHA mapping
08 common-native metrics and visual QC
09 integrity and reporting
~~~

The active route uses SCMR-v5 timing TR=2.61 ms, VPS=87, TI=[10,100] ms, T2prep=[35,45,55] ms, FA=[45,45,45], ten weights per group. The active route does not use 12/15/10 slice standardization.

## Canonical external roots

| Purpose | Path | Status | Protected |
|---|---|---|---|
| Raw DICOM | /data/dengyz/dataset/mapping_subject_dicom/<SUBJECT> | CURRENT_RUNTIME_REFERENCE (session) | Yes |
| Shared data root | /data/dengyz/dataset/Code_10w_v1 | CURRENT_RUNTIME_REFERENCE | Yes |
| Standard preprocessed root | /data/dengyz/dataset/Code_10w_v1/Code_10w_preprocessed/<SUBJECT> | CURRENT_RUNTIME_REFERENCE for CYJ/DYZ/HHZ/HJL; verify per subject | Yes |
| DYL_2 preprocessed root | /data/dengyz/dataset/Code_10w_v1/Code_10w_runs/scmr_v5_rebuild_20261102_corrected/DYL_2/preprocessed/DYL_2 | CURRENT_RUNTIME_REFERENCE (session) | Yes |
| HHZ prepared root | /data/dengyz/dataset/Code_10w_v1/Code_10w_prepared_scmr_v5/HHZ | CURRENT_RUNTIME_REFERENCE | Yes |
| Other prepared roots | /data/dengyz/dataset/Code_10w_v1/Code_10w_runs/scmr_v5_rebuild_20261102_corrected/<SUBJECT>/prepared/<SUBJECT> | CURRENT_RUNTIME_REFERENCE | Yes |
| HHZ formal run | /data/dengyz/dataset/Code_10w_v1/Code_10w_runs/ismrm_v2_scmr_v5 | CURRENT_RUNTIME_REFERENCE | Yes |
| Other formal runs | /data/dengyz/dataset/Code_10w_v1/Code_10w_runs/scmr_v5_rebuild_20261102_corrected/<SUBJECT> | CURRENT_RUNTIME_REFERENCE | Yes |
| ROI bundle | /data/dengyz/dataset/Code_10w_v1/Code_10w_runs/ismrm_five_subject_analysis_v1/roi/derived/<SUBJECT>/20261026_scmr_native_roi_bundle | CURRENT_RUNTIME_REFERENCE | Yes; manual/derived asset |
| Archived project inputs | /data/dengyz/code/ismrm_v2_1/subjects/<SUBJECT>/inputs | CURRENT_RUNTIME_REFERENCE archive copy | Yes |
| Archived project outputs | /data/dengyz/code/ismrm_v2_1/runs/<SUBJECT> | CURRENT_RUNTIME_REFERENCE archive copy | Yes |

The ismrm_v2_1/subjects and runs trees are archive copies. They do not replace the original external source roots.

## Five-subject layout

For CYJ, DYZ, DYL_2 and HJL, the formal run pattern is:

~~~text
/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/scmr_v5_rebuild_20261102_corrected/<SUBJECT>/
  prepared/<SUBJECT>/
  sparse_inputs/<SUBJECT>/<CASE>/
  mapping/
  native_cache/
  reconstruction/proposed/<SUBJECT>/<CASE>/
  reconstruction/fit_first/<SUBJECT>/<CASE>/
  evaluation/final_pose/<SUBJECT>/
  evaluation/common_native/<SUBJECT>/
~~~

For HHZ, the formal SCMR-v5 run pattern is:

~~~text
/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/ismrm_v2_scmr_v5/
  sparse_inputs/HHZ/
  mapping/
  native_cache/
  reconstruction/fit_first/HHZ/
  evaluation/final_pose/HHZ/
  evaluation/common_native/HHZ/
~~~

The formal HHZ Proposed source was recorded at:

~~~text
/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/ismrm_v2/reconstruction/proposed/HHZ/
~~~

That source is historical/supplemental provenance and must be verified before reuse. The final archived copy is under /data/dengyz/code/ismrm_v2_1/runs/HHZ/reconstruction/proposed/HHZ.

The five subject names are a paper-cohort record only. Active runtime admission must remain manifest/parameter based, not a subject-name whitelist.

## Historical Final5 and standardization

The previously recorded historical root is:

~~~text
/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/phase4r_final_five_subject_k8_central_20261023
~~~

Possible subtrees include 00_inventory, 01_preprocessed, 02_standardized_12_15_10, 03_sparse_inputs, 04_reconstruction, 05_formal_endpoints, 06_metrics, and 07_publication_visualization.

This root is DOCUMENTED_HISTORICAL / SERVER_VERIFICATION_REQUIRED. The presence of 02_standardized_12_15_10 must not reconnect that historical route to the active pipeline. The frozen scientific contract says slice standardization is inactive.

## ROI and protected manual assets

The following are protected manual/derived assets:

~~~text
sax_myocardium_masks.npz
roi_derivation_manifest.json
myocardium_core_1px / myocardium_core_legacy masks
outer-filled LV masks
blood-pool masks
AHA mapping and bullseye inputs
~~~

They must not be regenerated, moved, renamed, or deleted during a code-only refactor.

## Repository relationship and Git policy

The repository contains code, configs, manifests/templates, tests, READMEs, and runtime contracts. The following are data-sized and must remain outside Git tracking:

~~~text
raw DICOM
preprocessed.mat and preprocessing outputs
observations.npz and timing arrays
native_maps.npz / geometry.npz / native cache
model.pt / decoder binaries
NIfTI reconstructions
large metrics and visualization trees
manual ROI archives
~~~

The local archive keeps these under subjects/ and runs/ for reproducibility, but they are gitignored. Small manifest.json and sha256.jsonl files may appear as untracked metadata; that does not mean the large data are tracked.

## Code refactor boundary

Allowed for future in-place refactoring:

~~~text
common/
pipeline/
scripts/
configs/
runtime/runtime_contract.json
tests/
docs/
README_*.md
~~~

Protected external-data roots:

~~~text
/data/dengyz/dataset/mapping_subject_dicom
/data/dengyz/dataset/Code_10w_v1/Code_10w_preprocessed
/data/dengyz/dataset/Code_10w_v1/Code_10w_prepared_scmr_v5
/data/dengyz/dataset/Code_10w_v1/Code_10w_runs
/data/dengyz/code/ismrm_v2_1/subjects
/data/dengyz/code/ismrm_v2_1/runs
~~~

Future code must receive these locations through explicit deployment configuration, manifests, or CLI arguments. Server-specific paths must not be added to scientific algorithms or protocol semantics.

## Unresolved items

1. Real-time readlink -f, stat, filesystem/mount, and size for every server root.
2. Whether any historical Final5 subtree remains active for a particular output.
3. Exact per-stack native-cache and mapping locations for every subject outside the archive copies.
4. Exact server model-binary location and filesystem identity.
5. Whether /data/dengyz/code/Code_10w is still used by any active job.

Use docs/SERVER_DATA_LAYOUT_VERIFY_COMMANDS.md to resolve these items without mutating data.

## Safety record

~~~text
No server access was attempted.
No subject data was moved.
No subject data was copied.
No subject data was deleted.
No DICOM, ROI, reconstruction, or metrics file was modified.
No scientific code or protocol was modified.
No Git tag was modified.
No push was performed.
~~~
