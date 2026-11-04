# Active pipeline dependency trace

Generated deterministically from runtime-source fingerprint `14ab1def2a47afbe6ab64b0a2a8887284b3c3608293936a99e3d84c048f5b6af` (generator `d03e8b49ae05c5c25cfac525c581ac2a04da46a40cba4d74b4b6f5376c5a3894`). Static only: no DICOM, MATLAB, NeSVoR, GPU, or model binary is executed.

## Stage 01

Entrypoint: `pipeline/01_RAW_DICOM/run_stage.sh`

Inputs:
- `RAW_ROOT/SUBJECT/{sax,2ch,4ch}/*.dcm`

Outputs:
- `subjects/SUBJECT/manifest.json`

Direct execution chain:
- `src/ismrm_v2/io/build_subject_deployment_manifest.py`
- `src/ismrm_v2/io/validate_subject_id.py`

Conditional direct execution:

Transitive implementation dependencies:

Compatibility import closure:

Runtime configs/assets:

Third-party runtime closure:

External runtime boundaries:

Historical-name cleanup candidates:

## Stage 02

Entrypoint: `pipeline/02_PREPROCESS_MATLAB/run_stage.sh`

Inputs:
- `deployment manifest`
- `DICOM`

Outputs:
- `RUN_ROOT/preprocessed/SUBJECT/{stack}/preprocessed.mat`

Direct execution chain:
- `common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/preprocess_stack_v1.m`
- `common/trad/scripts/local_preprocess_all.m`
- `common/trad/scripts/local_preprocess_one.m`
- `src/ismrm_v2/io/validate_subject_id.py`
- `src/ismrm_v2/preprocessing/validation.py`

Conditional direct execution:

Transitive implementation dependencies:
- `common/trad/modules/module_01_preprocess_matlab/hhz_original/MIND_registration/MIND_descriptor2D.m`
- `common/trad/modules/module_01_preprocess_matlab/hhz_original/MIND_registration/deformableReg2Dmind_asym_nodisplay.m`
- `common/trad/modules/module_01_preprocess_matlab/hhz_original/Utility/HHMMSS2Sec.m`
- `common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/mppca_denoise_multimap_stack.m`
- `common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/preprocess_options_v1.m`
- `src/ismrm_v2/io/deployment_manifest.py`
- `src/ismrm_v2/observations/mat_v73.py`

Compatibility import closure:

Runtime configs/assets:

Third-party runtime closure:

External runtime boundaries:
- `matlab`

Historical-name cleanup candidates:
- `common/trad/modules/module_01_preprocess_matlab/hhz_original/MIND_registration/MIND_descriptor2D.m`
- `common/trad/modules/module_01_preprocess_matlab/hhz_original/MIND_registration/deformableReg2Dmind_asym_nodisplay.m`
- `common/trad/modules/module_01_preprocess_matlab/hhz_original/Utility/HHMMSS2Sec.m`

## Stage 03

Entrypoint: `pipeline/03_PREPARE_OBSERVATIONS/run_stage.sh`

Inputs:
- `preprocessed MAT`
- `deployment manifest`
- `protocol YAML`

Outputs:
- `observations.npz`
- `manifest.json`
- `timing.npy`
- `qc_summary.json`

Direct execution chain:
- `src/ismrm_v2/io/validate_subject_id.py`
- `src/ismrm_v2/observations/prepare_all.py`

Conditional direct execution:

Transitive implementation dependencies:
- `src/ismrm_v2/io/deployment_manifest.py`
- `src/ismrm_v2/observations/build_manifest.py`
- `src/ismrm_v2/observations/geometry.py`
- `src/ismrm_v2/observations/mat_v73.py`
- `src/ismrm_v2/observations/prepare_observations.py`
- `src/ismrm_v2/observations/timing.py`

Compatibility import closure:

Runtime configs/assets:
- `configs/protocol_scmr_v5_10_100.yaml`

Third-party runtime closure:
- `common/trad/third_party/nesvor/nesvor/transform/__init__.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform_convert.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform_convert_torch.py`
- `common/trad/third_party/nesvor/nesvor/utils/__init__.py`
- `common/trad/third_party/nesvor/nesvor/utils/logger.py`
- `common/trad/third_party/nesvor/nesvor/utils/loss.py`
- `common/trad/third_party/nesvor/nesvor/utils/misc.py`
- `common/trad/third_party/nesvor/nesvor/utils/psf.py`
- `common/trad/third_party/nesvor/nesvor/utils/types.py`

External runtime boundaries:

Historical-name cleanup candidates:

## Stage 04

Entrypoint: `pipeline/04_SPARSE_PROFILES/run_stage.sh`

Inputs:
- `prepared observations`
- `sparse profile`

Outputs:
- `sparse_inputs/SUBJECT/CASE/sparse_manifest.json`

Direct execution chain:
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/build_sparse_profiles.sh`
- `src/ismrm_v2/io/validate_subject_id.py`

Conditional direct execution:
- `src/ismrm_v2/sampling/sparse.py`

Transitive implementation dependencies:
- `src/ismrm_v2/fit_first/contracts.py`
- `src/ismrm_v2/observations/geometry.py`
- `src/ismrm_v2/proposed/rigid_psf/hb1_stack_adapter.py`

Compatibility import closure:

Runtime configs/assets:
- `common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/S2CH100_4CH50.json`
- `common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/S2CH50_4CH100.json`
- `common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/SAX100_2CH0_4CH0.json`
- `common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/SAX100_2CH0_4CH100.json`
- `common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/SAX100_2CH100_4CH0.json`

Third-party runtime closure:
- `common/trad/third_party/nesvor/nesvor/image/__init__.py`
- `common/trad/third_party/nesvor/nesvor/image/image.py`
- `common/trad/third_party/nesvor/nesvor/image/image_utils.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/__init__.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/slice_acq.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/slice_acq_torch.py`
- `common/trad/third_party/nesvor/nesvor/svr/registration.py`
- `common/trad/third_party/nesvor/nesvor/transform/__init__.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform_convert.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform_convert_torch.py`
- `common/trad/third_party/nesvor/nesvor/utils/__init__.py`
- `common/trad/third_party/nesvor/nesvor/utils/logger.py`
- `common/trad/third_party/nesvor/nesvor/utils/loss.py`
- `common/trad/third_party/nesvor/nesvor/utils/misc.py`
- `common/trad/third_party/nesvor/nesvor/utils/psf.py`
- `common/trad/third_party/nesvor/nesvor/utils/types.py`

External runtime boundaries:

Historical-name cleanup candidates:

## Stage 05

Entrypoint: `pipeline/05_SCMR_V5_FIT_FIRST/run_stage.sh`

Inputs:
- `preprocessed MAT`
- `prepared observations`
- `sparse inputs`

Outputs:
- `native cache`
- `fit_first reconstruction`

Direct execution chain:
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh`
- `common/trad/evaluation/scmr/matlab/build_scmr_v5_native_maps.m`
- `src/ismrm_v2/fit_first/scmr_v5_native_cache.py`
- `src/ismrm_v2/io/validate_subject_id.py`

Conditional direct execution:
- `src/ismrm_v2/fit_first/baseline.py`

Transitive implementation dependencies:
- `common/trad/evaluation/scmr/matlab/scmr_v5_matcher/function_T1T2_10HB_bssfp_qc.m`
- `common/trad/evaluation/scmr/matlab/scmr_v5_matcher/matlab/dictionary/sim_T1T2_10HB_bssfp.m`
- `common/trad/evaluation/scmr/matlab/scmr_v5_matcher/matlab/utils/d2r.m`
- `src/ismrm_v2/fit_first/contracts.py`
- `src/ismrm_v2/fit_first/dictionary.py`
- `src/ismrm_v2/fit_first/fit_first.py`
- `src/ismrm_v2/fit_first/fit_first_native_geometry.py`
- `src/ismrm_v2/fit_first/fit_first_native_pose_audit.py`
- `src/ismrm_v2/fit_first/simulate_hhz_signals.py`
- `src/ismrm_v2/observations/geometry.py`
- `src/ismrm_v2/proposed/rigid_psf/hb1_stack_adapter.py`
- `src/ismrm_v2/sampling/sparse.py`
- `src/ismrm_v2/sampling/sparse_profile_manifest.py`

Compatibility import closure:
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_intensity_preserving.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/reuse_standardized_native_dictionary_maps.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/standardize_subject_slices.py`

Runtime configs/assets:

Third-party runtime closure:
- `common/trad/third_party/nesvor/nesvor/image/__init__.py`
- `common/trad/third_party/nesvor/nesvor/image/image.py`
- `common/trad/third_party/nesvor/nesvor/image/image_utils.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/__init__.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/slice_acq.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/slice_acq_torch.py`
- `common/trad/third_party/nesvor/nesvor/svr/registration.py`
- `common/trad/third_party/nesvor/nesvor/transform/__init__.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform_convert.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform_convert_torch.py`
- `common/trad/third_party/nesvor/nesvor/utils/__init__.py`
- `common/trad/third_party/nesvor/nesvor/utils/logger.py`
- `common/trad/third_party/nesvor/nesvor/utils/loss.py`
- `common/trad/third_party/nesvor/nesvor/utils/misc.py`
- `common/trad/third_party/nesvor/nesvor/utils/psf.py`
- `common/trad/third_party/nesvor/nesvor/utils/types.py`

External runtime boundaries:
- `matlab`
- `nesvor`

Historical-name cleanup candidates:
- `src/ismrm_v2/fit_first/simulate_hhz_signals.py`

## Stage 06

Entrypoint: `pipeline/06_PROPOSED/run_stage.sh`

Inputs:
- `prepared or sparse observations`
- `decoder/B6 assets`
- `protocol`

Outputs:
- `model.pt`
- `T1_3D.nii.gz`
- `T2_3D.nii.gz`
- `B1_3D.nii.gz`
- `amplitude_3D.nii.gz`
- `final_rigid_poses.json`

Direct execution chain:
- `src/ismrm_v2/io/validate_subject_id.py`
- `src/ismrm_v2/proposed/proposed.py`
- `src/ismrm_v2/proposed/q002/run_q002_reconstruction.py`
- `src/ismrm_v2/proposed/q002/training.py`
- `src/ismrm_v2/proposed/reconstruction/orchestration.py`

Conditional direct execution:

Transitive implementation dependencies:
- `src/ismrm_v2/fit_first/contracts.py`
- `src/ismrm_v2/observations/geometry.py`
- `src/ismrm_v2/proposed/inference/export_quantitative.py`
- `src/ismrm_v2/proposed/inference/reprojection.py`
- `src/ismrm_v2/proposed/mlp_trainer.py`
- `src/ismrm_v2/proposed/profiling/summarize_timing_profile.py`
- `src/ismrm_v2/proposed/q001_config.py`
- `src/ismrm_v2/proposed/q002/contracts.py`
- `src/ismrm_v2/proposed/q002/joint_dataset.py`
- `src/ismrm_v2/proposed/q002/training_curves.py`
- `src/ismrm_v2/proposed/quantitative_point_dataset.py`
- `src/ismrm_v2/proposed/rigid_psf/hb1_stack_adapter.py`
- `src/ismrm_v2/proposed/rigid_psf/rigid_psf_forward.py`
- `src/ismrm_v2/proposed/signal_decoder/checkpoint_loader.py`
- `src/ismrm_v2/proposed/signal_decoder/decoder_factory.py`
- `src/ismrm_v2/proposed/signal_decoder/frozen_decoder.py`
- `src/ismrm_v2/proposed/signal_decoder/mlp_model.py`
- `src/ismrm_v2/proposed/signal_decoder/provenance.py`
- `src/ismrm_v2/proposed/signal_decoder/trad_teacher/trad_signal_simulator.py`
- `src/ismrm_v2/proposed/trad_quantitative_inr.py`
- `src/ismrm_v2/proposed/trad_signal_decoder/decoder_factory.py`
- `src/ismrm_v2/proposed/trad_signal_decoder/trad_signal_simulator.py`
- `src/ismrm_v2/proposed/training/experiment_infrastructure.py`
- `src/ismrm_v2/proposed/training/training_space.py`
- `src/ismrm_v2/sampling/sparse.py`
- `src/ismrm_v2/sampling/sparse_profile_manifest.py`

Compatibility import closure:

Runtime configs/assets:
- `configs/models_manifest.json`
- `configs/protocol_scmr_v5_10_100.yaml`
- `runtime/runtime_contract.json`

Third-party runtime closure:
- `common/trad/third_party/nesvor/nesvor/image/__init__.py`
- `common/trad/third_party/nesvor/nesvor/image/image.py`
- `common/trad/third_party/nesvor/nesvor/image/image_utils.py`
- `common/trad/third_party/nesvor/nesvor/inr/__init__.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/__init__.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/slice_acq.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/slice_acq_torch.py`
- `common/trad/third_party/nesvor/nesvor/svr/registration.py`
- `common/trad/third_party/nesvor/nesvor/transform/__init__.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform_convert.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform_convert_torch.py`
- `common/trad/third_party/nesvor/nesvor/utils/__init__.py`
- `common/trad/third_party/nesvor/nesvor/utils/logger.py`
- `common/trad/third_party/nesvor/nesvor/utils/loss.py`
- `common/trad/third_party/nesvor/nesvor/utils/misc.py`
- `common/trad/third_party/nesvor/nesvor/utils/psf.py`
- `common/trad/third_party/nesvor/nesvor/utils/types.py`

External runtime boundaries:
- `cuda`
- `decoder_scmr_v5_10_100.pth`
- `b6_model.pt`

Historical-name cleanup candidates:

## Stage 07

Entrypoint: `pipeline/07_FINAL_POSE/run_stage.sh`

Inputs:
- `completed Proposed/Fit-first outputs`
- `ROI bundle`

Outputs:
- `evaluation/final_pose/SUBJECT`

Direct execution chain:
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh`
- `src/ismrm_v2/io/validate_subject_id.py`

Conditional direct execution:
- `src/ismrm_v2/evaluation/aha/current_fit_first_final_pose_bundle.py`
- `src/ismrm_v2/evaluation/aha/cyj11_r3_direct_final_pose_bundle.py`
- `src/ismrm_v2/evaluation/aha/derive_final_pose_mapping.py`
- `src/ismrm_v2/evaluation/aha/paired_aha_sd.py`
- `src/ismrm_v2/evaluation/aha/paired_final_pose_postprocess.py`
- `src/ismrm_v2/evaluation/aha/paired_final_pose_through_plane.py`

Transitive implementation dependencies:
- `src/ismrm_v2/evaluation/aha/cyj11_adapter.py`
- `src/ismrm_v2/evaluation/aha/cyj11_aha16.py`
- `src/ismrm_v2/evaluation/aha/cyj11_legacy_roi.py`
- `src/ismrm_v2/evaluation/aha/cyj11_myocardial_roi_metrics.py`
- `src/ismrm_v2/evaluation/aha/cyj11_postprocess.py`
- `src/ismrm_v2/evaluation/aha/cyj11_r3_fit_first_final_pose_bundle.py`
- `src/ismrm_v2/evaluation/aha/cyj11_runner.py`
- `src/ismrm_v2/evaluation/aha/final5_adapter.py`
- `src/ismrm_v2/evaluation/aha/final5_aha16.py`
- `src/ismrm_v2/evaluation/aha/final5_myocardial_roi_metrics.py`
- `src/ismrm_v2/evaluation/aha/legacy_aha16.py`
- `src/ismrm_v2/evaluation/aha/paired_final_pose_lv_metrics.py`
- `src/ismrm_v2/evaluation/current_native_central_component_mask.py`
- `src/ismrm_v2/evaluation/direct_native_bundles.py`
- `src/ismrm_v2/evaluation/direct_native_endpoint_audit.py`
- `src/ismrm_v2/evaluation/evaluation.py`
- `src/ismrm_v2/evaluation/final5_partitioned_native_metrics.py`
- `src/ismrm_v2/evaluation/final5_publication_visualization.py`
- `src/ismrm_v2/evaluation/final_cyj_reporting.py`
- `src/ismrm_v2/evaluation/final_native_grid_metrics.py`
- `src/ismrm_v2/evaluation/final_sparse_native_metrics.py`
- `src/ismrm_v2/evaluation/heldout.py`
- `src/ismrm_v2/evaluation/l100_mask_paired_eval.py`
- `src/ismrm_v2/evaluation/matched_output.py`
- `src/ismrm_v2/evaluation/myocardium/build_native_bundle.py`
- `src/ismrm_v2/evaluation/myocardium/legacy_masks.py`
- `src/ismrm_v2/evaluation/scmr/quality_control.py`
- `src/ismrm_v2/evaluation/scmr/reference_2d.py`
- `src/ismrm_v2/evaluation/visualization.py`
- `src/ismrm_v2/fit_first/contracts.py`
- `src/ismrm_v2/fit_first/dictionary.py`
- `src/ismrm_v2/fit_first/fit_first.py`
- `src/ismrm_v2/fit_first/fit_first_mask_audit.py`
- `src/ismrm_v2/fit_first/fit_first_native_geometry.py`
- `src/ismrm_v2/fit_first/fit_first_route_contract.py`
- `src/ismrm_v2/fit_first/formal_l100_sparse_manifest.py`
- `src/ismrm_v2/fit_first/frozen_native_support.py`
- `src/ismrm_v2/fit_first/simulate_hhz_signals.py`
- `src/ismrm_v2/observations/geometry.py`
- `src/ismrm_v2/proposed/inference/export_quantitative.py`
- `src/ismrm_v2/proposed/mlp_trainer.py`
- `src/ismrm_v2/proposed/profiling/summarize_timing_profile.py`
- `src/ismrm_v2/proposed/quantitative_point_dataset.py`
- `src/ismrm_v2/proposed/reconstruction/orchestration.py`
- `src/ismrm_v2/proposed/rigid_psf/hb1_stack_adapter.py`
- `src/ismrm_v2/proposed/rigid_psf/rigid_psf_forward.py`
- `src/ismrm_v2/proposed/signal_decoder/checkpoint_loader.py`
- `src/ismrm_v2/proposed/signal_decoder/decoder_factory.py`
- `src/ismrm_v2/proposed/signal_decoder/frozen_decoder.py`
- `src/ismrm_v2/proposed/signal_decoder/mlp_model.py`
- `src/ismrm_v2/proposed/signal_decoder/provenance.py`
- `src/ismrm_v2/proposed/signal_decoder/trad_teacher/trad_signal_simulator.py`
- `src/ismrm_v2/proposed/trad_quantitative_inr.py`
- `src/ismrm_v2/proposed/trad_signal_decoder/decoder_factory.py`
- `src/ismrm_v2/proposed/trad_signal_decoder/trad_signal_simulator.py`
- `src/ismrm_v2/proposed/training/experiment_infrastructure.py`
- `src/ismrm_v2/proposed/training/training_space.py`
- `src/ismrm_v2/sampling/sparse.py`
- `src/ismrm_v2/sampling/sparse_profile_manifest.py`

Compatibility import closure:

Runtime configs/assets:

Third-party runtime closure:
- `common/trad/third_party/nesvor/nesvor/image/__init__.py`
- `common/trad/third_party/nesvor/nesvor/image/image.py`
- `common/trad/third_party/nesvor/nesvor/image/image_utils.py`
- `common/trad/third_party/nesvor/nesvor/inr/__init__.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/__init__.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/slice_acq.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/slice_acq_torch.py`
- `common/trad/third_party/nesvor/nesvor/svr/registration.py`
- `common/trad/third_party/nesvor/nesvor/transform/__init__.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform_convert.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform_convert_torch.py`
- `common/trad/third_party/nesvor/nesvor/utils/__init__.py`
- `common/trad/third_party/nesvor/nesvor/utils/logger.py`
- `common/trad/third_party/nesvor/nesvor/utils/loss.py`
- `common/trad/third_party/nesvor/nesvor/utils/misc.py`
- `common/trad/third_party/nesvor/nesvor/utils/psf.py`
- `common/trad/third_party/nesvor/nesvor/utils/types.py`

External runtime boundaries:
- `cuda`

Historical-name cleanup candidates:
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh`
- `src/ismrm_v2/evaluation/aha/cyj11_adapter.py`
- `src/ismrm_v2/evaluation/aha/cyj11_aha16.py`
- `src/ismrm_v2/evaluation/aha/cyj11_legacy_roi.py`
- `src/ismrm_v2/evaluation/aha/cyj11_myocardial_roi_metrics.py`
- `src/ismrm_v2/evaluation/aha/cyj11_postprocess.py`
- `src/ismrm_v2/evaluation/aha/cyj11_r3_direct_final_pose_bundle.py`
- `src/ismrm_v2/evaluation/aha/cyj11_r3_fit_first_final_pose_bundle.py`
- `src/ismrm_v2/evaluation/aha/cyj11_runner.py`
- `src/ismrm_v2/evaluation/aha/final5_adapter.py`
- `src/ismrm_v2/evaluation/aha/final5_aha16.py`
- `src/ismrm_v2/evaluation/aha/final5_myocardial_roi_metrics.py`
- `src/ismrm_v2/evaluation/aha/legacy_aha16.py`
- `src/ismrm_v2/evaluation/final5_partitioned_native_metrics.py`
- `src/ismrm_v2/evaluation/final5_publication_visualization.py`
- `src/ismrm_v2/evaluation/myocardium/legacy_masks.py`
- `src/ismrm_v2/fit_first/simulate_hhz_signals.py`

## Stage 08

Entrypoint: `pipeline/08_EVALUATION_AND_VISUALIZATION/run_stage.sh`

Inputs:
- `completed outputs`
- `ROI bundle`
- `final-pose/native cache`

Outputs:
- `evaluation/common_native/SUBJECT`

Direct execution chain:
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_all_stack_roi_metrics.sh`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_metrics_qc.sh`
- `src/ismrm_v2/io/validate_subject_id.py`

Conditional direct execution:
- `src/ismrm_v2/evaluation/aha/common_native_all_stack_roi_metrics.py`
- `src/ismrm_v2/evaluation/aha/common_native_metrics_qc.py`

Transitive implementation dependencies:
- `src/ismrm_v2/evaluation/aha/common_native.py`
- `src/ismrm_v2/evaluation/aha/cyj11_adapter.py`
- `src/ismrm_v2/evaluation/aha/cyj11_aha16.py`
- `src/ismrm_v2/evaluation/aha/cyj11_legacy_roi.py`
- `src/ismrm_v2/evaluation/aha/cyj11_myocardial_roi_metrics.py`
- `src/ismrm_v2/evaluation/aha/cyj11_runner.py`
- `src/ismrm_v2/evaluation/aha/final5_adapter.py`
- `src/ismrm_v2/evaluation/aha/final5_aha16.py`
- `src/ismrm_v2/evaluation/aha/final5_myocardial_roi_metrics.py`
- `src/ismrm_v2/evaluation/aha/legacy_aha16.py`
- `src/ismrm_v2/evaluation/aha/paired_final_pose_lv_metrics.py`
- `src/ismrm_v2/evaluation/current_native_central_component_mask.py`
- `src/ismrm_v2/evaluation/evaluation.py`
- `src/ismrm_v2/evaluation/final5_partitioned_native_metrics.py`
- `src/ismrm_v2/evaluation/final5_publication_visualization.py`
- `src/ismrm_v2/evaluation/final_cyj_reporting.py`
- `src/ismrm_v2/evaluation/final_native_grid_metrics.py`
- `src/ismrm_v2/evaluation/final_sparse_native_metrics.py`
- `src/ismrm_v2/evaluation/final_stratified_native_metrics.py`
- `src/ismrm_v2/evaluation/heldout.py`
- `src/ismrm_v2/evaluation/l100_mask_paired_eval.py`
- `src/ismrm_v2/evaluation/matched_output.py`
- `src/ismrm_v2/evaluation/myocardium/build_native_bundle.py`
- `src/ismrm_v2/evaluation/myocardium/legacy_masks.py`
- `src/ismrm_v2/evaluation/scmr/quality_control.py`
- `src/ismrm_v2/evaluation/scmr/reference_2d.py`
- `src/ismrm_v2/evaluation/visualization.py`
- `src/ismrm_v2/fit_first/contracts.py`
- `src/ismrm_v2/fit_first/dictionary.py`
- `src/ismrm_v2/fit_first/fit_first.py`
- `src/ismrm_v2/fit_first/fit_first_mask_audit.py`
- `src/ismrm_v2/fit_first/fit_first_route_contract.py`
- `src/ismrm_v2/fit_first/formal_l100_sparse_manifest.py`
- `src/ismrm_v2/fit_first/frozen_native_support.py`
- `src/ismrm_v2/fit_first/simulate_hhz_signals.py`
- `src/ismrm_v2/observations/geometry.py`
- `src/ismrm_v2/proposed/rigid_psf/hb1_stack_adapter.py`
- `src/ismrm_v2/sampling/sparse.py`
- `src/ismrm_v2/sampling/sparse_profile_manifest.py`

Compatibility import closure:

Runtime configs/assets:

Third-party runtime closure:
- `common/trad/third_party/nesvor/nesvor/image/__init__.py`
- `common/trad/third_party/nesvor/nesvor/image/image.py`
- `common/trad/third_party/nesvor/nesvor/image/image_utils.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/__init__.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/slice_acq.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/slice_acq_torch.py`
- `common/trad/third_party/nesvor/nesvor/svr/registration.py`
- `common/trad/third_party/nesvor/nesvor/transform/__init__.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform_convert.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform_convert_torch.py`
- `common/trad/third_party/nesvor/nesvor/utils/__init__.py`
- `common/trad/third_party/nesvor/nesvor/utils/logger.py`
- `common/trad/third_party/nesvor/nesvor/utils/loss.py`
- `common/trad/third_party/nesvor/nesvor/utils/misc.py`
- `common/trad/third_party/nesvor/nesvor/utils/psf.py`
- `common/trad/third_party/nesvor/nesvor/utils/types.py`

External runtime boundaries:

Historical-name cleanup candidates:
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_all_stack_roi_metrics.sh`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_metrics_qc.sh`
- `src/ismrm_v2/evaluation/aha/cyj11_adapter.py`
- `src/ismrm_v2/evaluation/aha/cyj11_aha16.py`
- `src/ismrm_v2/evaluation/aha/cyj11_legacy_roi.py`
- `src/ismrm_v2/evaluation/aha/cyj11_myocardial_roi_metrics.py`
- `src/ismrm_v2/evaluation/aha/cyj11_runner.py`
- `src/ismrm_v2/evaluation/aha/final5_adapter.py`
- `src/ismrm_v2/evaluation/aha/final5_aha16.py`
- `src/ismrm_v2/evaluation/aha/final5_myocardial_roi_metrics.py`
- `src/ismrm_v2/evaluation/aha/legacy_aha16.py`
- `src/ismrm_v2/evaluation/final5_partitioned_native_metrics.py`
- `src/ismrm_v2/evaluation/final5_publication_visualization.py`
- `src/ismrm_v2/evaluation/myocardium/legacy_masks.py`
- `src/ismrm_v2/fit_first/simulate_hhz_signals.py`

## Stage 09

Entrypoint: `pipeline/09_INTEGRITY_AND_REPORTING/run_stage.sh`

Inputs:
- `repository contracts`

Outputs:
- `integrity and freeze reports`

Direct execution chain:
- `scripts/verify_no_legacy_code_path.py`
- `scripts/verify_project_integrity.py`
- `scripts/verify_runtime_contract.py`
- `scripts/verify_scientific_freeze.py`

Conditional direct execution:

Transitive implementation dependencies:

Compatibility import closure:

Runtime configs/assets:

Third-party runtime closure:

External runtime boundaries:

Historical-name cleanup candidates:
- `scripts/verify_no_legacy_code_path.py`

## Historical-name cleanup candidates

Historical `cyj11`, `final5`, `hhz`, and `legacy` names are recorded as metadata; reachability, not spelling, decides classification.
