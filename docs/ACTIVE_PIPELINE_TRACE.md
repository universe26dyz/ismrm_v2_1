# Active pipeline dependency trace

Generated deterministically from runtime-source fingerprint `fc14774cc657e7d1a262cd488ad371854607f33186f60b15a970fa2db7d47061` (generator `2e13540679c1f4a714a9ccf705444d0f81799be4f8fd09d8501d5182a778596f`). Static only: no DICOM, MATLAB, NeSVoR, GPU, or model binary is executed.

## Stage 01

Entrypoint: `pipeline/01_RAW_DICOM/run_stage.sh`

Inputs:
- `RAW_ROOT/SUBJECT/{sax,2ch,4ch}/*.dcm`

Outputs:
- `subjects/SUBJECT/manifest.json`

Direct execution chain:
- `scripts/build_subject_deployment_manifest.py`
- `scripts/validate_subject_id.py`

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
- `common/trad/scripts/validate_transferred_preprocessed.py`

Conditional direct execution:

Transitive implementation dependencies:
- `common/scripts/deployment_manifest.py`
- `common/trad/modules/module_01_preprocess_matlab/hhz_original/MIND_registration/MIND_descriptor2D.m`
- `common/trad/modules/module_01_preprocess_matlab/hhz_original/MIND_registration/deformableReg2Dmind_asym_nodisplay.m`
- `common/trad/modules/module_01_preprocess_matlab/hhz_original/Utility/HHMMSS2Sec.m`
- `common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/mppca_denoise_multimap_stack.m`
- `common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/preprocess_options_v1.m`
- `common/trad/modules/module_02_data_bridge/mat_v73.py`

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
- `common/trad/scripts/prepare_all_observations.py`

Conditional direct execution:

Transitive implementation dependencies:
- `common/scripts/deployment_manifest.py`
- `common/trad/modules/module_02_data_bridge/build_manifest.py`
- `common/trad/modules/module_02_data_bridge/geometry.py`
- `common/trad/modules/module_02_data_bridge/mat_v73.py`
- `common/trad/modules/module_02_data_bridge/prepare_observations.py`
- `common/trad/modules/module_02_data_bridge/timing.py`

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

Conditional direct execution:
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse.py`

Transitive implementation dependencies:
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/contracts.py`
- `common/trad/modules/module_02_data_bridge/geometry.py`
- `common/trad/modules/module_06_rigid_psf/hb1_stack_adapter.py`

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
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/scmr_v5_native_cache.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh`
- `common/trad/evaluation/scmr/matlab/build_scmr_v5_native_maps.m`

Conditional direct execution:
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/baseline.py`

Transitive implementation dependencies:
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/contracts.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/dictionary.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_native_geometry.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_native_pose_audit.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse_profile_manifest.py`
- `common/paper_experiments/joint_vs_2dfit/simulation/simulate_hhz_signals.py`
- `common/trad/evaluation/scmr/matlab/scmr_v5_matcher/function_T1T2_10HB_bssfp_qc.m`
- `common/trad/evaluation/scmr/matlab/scmr_v5_matcher/matlab/dictionary/sim_T1T2_10HB_bssfp.m`
- `common/trad/evaluation/scmr/matlab/scmr_v5_matcher/matlab/utils/d2r.m`
- `common/trad/modules/module_02_data_bridge/geometry.py`
- `common/trad/modules/module_06_rigid_psf/hb1_stack_adapter.py`

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
- `common/paper_experiments/joint_vs_2dfit/simulation/simulate_hhz_signals.py`

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
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/proposed.py`
- `common/quality_experiments/Q002_B6_roi_same_anchor_mse/run_q002_reconstruction.py`
- `common/quality_experiments/Q002_B6_roi_same_anchor_mse/training.py`
- `common/reconstruction_core/orchestration.py`

Conditional direct execution:

Transitive implementation dependencies:
- `common/mlp/modules/module_05_signal_decoder/checkpoint_loader.py`
- `common/mlp/modules/module_05_signal_decoder/decoder_factory.py`
- `common/mlp/modules/module_05_signal_decoder/frozen_decoder.py`
- `common/mlp/modules/module_05_signal_decoder/mlp_model.py`
- `common/mlp/modules/module_05_signal_decoder/provenance.py`
- `common/mlp/modules/module_05_signal_decoder/trad_teacher/trad_signal_simulator.py`
- `common/mlp/modules/module_07_objective_training/mlp_trainer.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/contracts.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse_profile_manifest.py`
- `common/quality_experiments/Q001_recon/config.py`
- `common/quality_experiments/Q002_B6_roi_same_anchor_mse/contracts.py`
- `common/quality_experiments/Q002_B6_roi_same_anchor_mse/joint_dataset.py`
- `common/quality_experiments/Q002_B6_roi_same_anchor_mse/training_curves.py`
- `common/trad/evaluation/profiling/summarize_timing_profile.py`
- `common/trad/modules/module_02_data_bridge/geometry.py`
- `common/trad/modules/module_03_dataset_geometry/quantitative_point_dataset.py`
- `common/trad/modules/module_04_quantitative_inr/quantitative_inr.py`
- `common/trad/modules/module_05_signal_decoder/decoder_factory.py`
- `common/trad/modules/module_05_signal_decoder/trad_signal_simulator.py`
- `common/trad/modules/module_06_rigid_psf/hb1_stack_adapter.py`
- `common/trad/modules/module_06_rigid_psf/rigid_psf_forward.py`
- `common/trad/modules/module_07_objective_training/experiment_infrastructure.py`
- `common/trad/modules/module_07_objective_training/training_space.py`
- `common/trad/modules/module_08_inference_export/export_quantitative.py`
- `common/trad/modules/module_08_inference_export/reprojection.py`

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

Conditional direct execution:
- `common/paper_experiments/joint_vs_2dfit/aha/current_fit_first_final_pose_bundle.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_r3_direct_final_pose_bundle.py`
- `common/paper_experiments/joint_vs_2dfit/aha/derive_final_pose_mapping.py`
- `common/paper_experiments/joint_vs_2dfit/aha/paired_aha_sd.py`
- `common/paper_experiments/joint_vs_2dfit/aha/paired_final_pose_postprocess.py`
- `common/paper_experiments/joint_vs_2dfit/aha/paired_final_pose_through_plane.py`

Transitive implementation dependencies:
- `common/evaluation_common/myocardium/build_native_bundle.py`
- `common/evaluation_common/myocardium/legacy_masks.py`
- `common/mlp/modules/module_05_signal_decoder/checkpoint_loader.py`
- `common/mlp/modules/module_05_signal_decoder/decoder_factory.py`
- `common/mlp/modules/module_05_signal_decoder/frozen_decoder.py`
- `common/mlp/modules/module_05_signal_decoder/mlp_model.py`
- `common/mlp/modules/module_05_signal_decoder/provenance.py`
- `common/mlp/modules/module_05_signal_decoder/trad_teacher/trad_signal_simulator.py`
- `common/mlp/modules/module_07_objective_training/mlp_trainer.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_adapter.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_aha16.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_legacy_roi.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_myocardial_roi_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_postprocess.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_r3_fit_first_final_pose_bundle.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_runner.py`
- `common/paper_experiments/joint_vs_2dfit/aha/final5_adapter.py`
- `common/paper_experiments/joint_vs_2dfit/aha/final5_aha16.py`
- `common/paper_experiments/joint_vs_2dfit/aha/final5_myocardial_roi_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/aha/legacy_aha16.py`
- `common/paper_experiments/joint_vs_2dfit/aha/paired_final_pose_lv_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/contracts.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/current_native_central_component_mask.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/dictionary.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/direct_native_bundles.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/direct_native_endpoint_audit.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/evaluation.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final5_partitioned_native_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final5_publication_visualization.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final_cyj_reporting.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final_native_grid_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final_sparse_native_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_mask_audit.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_native_geometry.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_route_contract.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/formal_l100_sparse_manifest.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/frozen_native_support.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/heldout.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/l100_mask_paired_eval.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/matched_output.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse_profile_manifest.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/visualization.py`
- `common/paper_experiments/joint_vs_2dfit/simulation/simulate_hhz_signals.py`
- `common/reconstruction_core/orchestration.py`
- `common/trad/evaluation/profiling/summarize_timing_profile.py`
- `common/trad/evaluation/scmr/quality_control.py`
- `common/trad/evaluation/scmr/reference_2d.py`
- `common/trad/modules/module_02_data_bridge/geometry.py`
- `common/trad/modules/module_03_dataset_geometry/quantitative_point_dataset.py`
- `common/trad/modules/module_04_quantitative_inr/quantitative_inr.py`
- `common/trad/modules/module_05_signal_decoder/decoder_factory.py`
- `common/trad/modules/module_05_signal_decoder/trad_signal_simulator.py`
- `common/trad/modules/module_06_rigid_psf/hb1_stack_adapter.py`
- `common/trad/modules/module_06_rigid_psf/rigid_psf_forward.py`
- `common/trad/modules/module_07_objective_training/experiment_infrastructure.py`
- `common/trad/modules/module_07_objective_training/training_space.py`
- `common/trad/modules/module_08_inference_export/export_quantitative.py`

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
- `common/evaluation_common/myocardium/legacy_masks.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_adapter.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_aha16.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_legacy_roi.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_myocardial_roi_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_postprocess.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_r3_direct_final_pose_bundle.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_r3_fit_first_final_pose_bundle.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_runner.py`
- `common/paper_experiments/joint_vs_2dfit/aha/final5_adapter.py`
- `common/paper_experiments/joint_vs_2dfit/aha/final5_aha16.py`
- `common/paper_experiments/joint_vs_2dfit/aha/final5_myocardial_roi_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/aha/legacy_aha16.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final5_partitioned_native_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final5_publication_visualization.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh`
- `common/paper_experiments/joint_vs_2dfit/simulation/simulate_hhz_signals.py`

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

Conditional direct execution:
- `common/paper_experiments/joint_vs_2dfit/aha/common_native_all_stack_roi_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/aha/common_native_metrics_qc.py`

Transitive implementation dependencies:
- `common/evaluation_common/myocardium/build_native_bundle.py`
- `common/evaluation_common/myocardium/legacy_masks.py`
- `common/paper_experiments/joint_vs_2dfit/aha/common_native.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_adapter.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_aha16.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_legacy_roi.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_myocardial_roi_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_runner.py`
- `common/paper_experiments/joint_vs_2dfit/aha/final5_adapter.py`
- `common/paper_experiments/joint_vs_2dfit/aha/final5_aha16.py`
- `common/paper_experiments/joint_vs_2dfit/aha/final5_myocardial_roi_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/aha/legacy_aha16.py`
- `common/paper_experiments/joint_vs_2dfit/aha/paired_final_pose_lv_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/contracts.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/current_native_central_component_mask.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/dictionary.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/evaluation.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final5_partitioned_native_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final5_publication_visualization.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final_cyj_reporting.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final_native_grid_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final_sparse_native_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final_stratified_native_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_mask_audit.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_route_contract.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/formal_l100_sparse_manifest.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/frozen_native_support.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/heldout.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/l100_mask_paired_eval.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/matched_output.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse_profile_manifest.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/visualization.py`
- `common/paper_experiments/joint_vs_2dfit/simulation/simulate_hhz_signals.py`
- `common/trad/evaluation/scmr/quality_control.py`
- `common/trad/evaluation/scmr/reference_2d.py`
- `common/trad/modules/module_02_data_bridge/geometry.py`
- `common/trad/modules/module_06_rigid_psf/hb1_stack_adapter.py`

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
- `common/evaluation_common/myocardium/legacy_masks.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_adapter.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_aha16.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_legacy_roi.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_myocardial_roi_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_runner.py`
- `common/paper_experiments/joint_vs_2dfit/aha/final5_adapter.py`
- `common/paper_experiments/joint_vs_2dfit/aha/final5_aha16.py`
- `common/paper_experiments/joint_vs_2dfit/aha/final5_myocardial_roi_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/aha/legacy_aha16.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final5_partitioned_native_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final5_publication_visualization.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_all_stack_roi_metrics.sh`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_metrics_qc.sh`
- `common/paper_experiments/joint_vs_2dfit/simulation/simulate_hhz_signals.py`

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
