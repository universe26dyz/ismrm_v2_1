# Active pipeline dependency trace

Generated deterministically from `c92fde00ec65187b8cc172d5212dedfe6178ae1a`. This is a static, data-free trace; server MATLAB, model binaries, and NeSVoR are recorded but not executed.

## Stage 01

Direct and discovered active paths:

- `common/scripts/extract_subject_deployment_manifest.py`
- `configs/subject_stack_manifest.example.json`
- `pipeline/01_RAW_DICOM/run_stage.sh`
- `scripts/validate_active_subject.py`

## Stage 02

Direct and discovered active paths:

- `common/scripts/deployment_manifest.py`
- `common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/deformableReg2Dmind_adaptive.m`
- `common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/mppca_denoise_multimap_stack.m`
- `common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/preprocess_options_v1.m`
- `common/trad/modules/module_01_preprocess_matlab/preprocessing_v1/preprocess_stack_v1.m`
- `common/trad/modules/module_02_data_bridge/mat_v73.py`
- `common/trad/scripts/local_preprocess_all.m`
- `common/trad/scripts/local_preprocess_one.m`
- `common/trad/scripts/validate_transferred_preprocessed.py`
- `pipeline/02_PREPROCESS_MATLAB/run_stage.sh`

## Stage 03

Direct and discovered active paths:

- `common/scripts/deployment_manifest.py`
- `common/trad/modules/module_02_data_bridge/build_manifest.py`
- `common/trad/modules/module_02_data_bridge/geometry.py`
- `common/trad/modules/module_02_data_bridge/mat_v73.py`
- `common/trad/modules/module_02_data_bridge/prepare_observations.py`
- `common/trad/modules/module_02_data_bridge/timing.py`
- `common/trad/scripts/prepare_all_observations.py`
- `common/trad/third_party/nesvor/nesvor/transform/__init__.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform_convert.py`
- `common/trad/third_party/nesvor/nesvor/transform/transform_convert_torch.py`
- `common/trad/third_party/nesvor/nesvor/utils/__init__.py`
- `common/trad/third_party/nesvor/nesvor/utils/logger.py`
- `common/trad/third_party/nesvor/nesvor/utils/loss.py`
- `common/trad/third_party/nesvor/nesvor/utils/misc.py`
- `configs/protocol_scmr_v5_10_100.yaml`
- `pipeline/03_PREPARE_OBSERVATIONS/run_stage.sh`

## Stage 04

Direct and discovered active paths:

- `common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/CYJ_S2CH100_4CH50.json`
- `common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/CYJ_S2CH50_4CH100.json`
- `common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/CYJ_SAX100_2CH0_4CH0.json`
- `common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/CYJ_SAX100_2CH0_4CH100.json`
- `common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/CYJ_SAX100_2CH100_4CH0.json`
- `common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/S2CH100_4CH50.json`
- `common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/S2CH50_4CH100.json`
- `common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/SAX100_2CH0_4CH0.json`
- `common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/SAX100_2CH0_4CH100.json`
- `common/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/SAX100_2CH100_4CH0.json`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/contracts.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/build_sparse_profiles.sh`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/sparse_profile_manifest.py`
- `common/trad/modules/module_06_rigid_psf/hb1_stack_adapter.py`
- `common/trad/third_party/nesvor/nesvor/image/__init__.py`
- `common/trad/third_party/nesvor/nesvor/image/image.py`
- `common/trad/third_party/nesvor/nesvor/image/image_utils.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/__init__.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/slice_acq.py`
- `common/trad/third_party/nesvor/nesvor/slice_acquisition/slice_acq_torch.py`
- `common/trad/third_party/nesvor/nesvor/svr/registration.py`
- `common/trad/third_party/nesvor/nesvor/utils/__init__.py`
- `common/trad/third_party/nesvor/nesvor/utils/logger.py`
- `common/trad/third_party/nesvor/nesvor/utils/loss.py`
- `common/trad/third_party/nesvor/nesvor/utils/misc.py`
- `pipeline/04_SPARSE_PROFILES/run_stage.sh`

## Stage 05

Direct and discovered active paths:

- `common/paper_experiments/joint_vs_2dfit/phase2_clean/baseline.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/contracts.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/dictionary.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_denormalize.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_intensity_preserving.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_native_pose_audit.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/reuse_standardized_native_dictionary_maps.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/scmr_v5_native_cache.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/standardize_subject_slices.py`
- `common/paper_experiments/joint_vs_2dfit/simulation/simulate_hhz_signals.py`
- `common/trad/evaluation/scmr/matlab/build_scmr_v5_native_maps.m`
- `pipeline/05_SCMR_V5_FIT_FIRST/run_stage.sh`

## Stage 06

Direct and discovered active paths:

- `common/mlp/modules/module_04_quantitative_inr/quantitative_inr.py`
- `common/mlp/modules/module_05_signal_decoder/checkpoint_loader.py`
- `common/mlp/modules/module_05_signal_decoder/decoder_factory.py`
- `common/mlp/modules/module_05_signal_decoder/frozen_decoder.py`
- `common/mlp/modules/module_05_signal_decoder/mlp_model.py`
- `common/mlp/modules/module_05_signal_decoder/provenance.py`
- `common/mlp/modules/module_05_signal_decoder/trad_teacher/trad_signal_simulator.py`
- `common/mlp/modules/module_07_objective_training/mlp_trainer.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/contracts.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/proposed.py`
- `common/quality_experiments/Q001_recon/config.py`
- `common/quality_experiments/Q002_B6_roi_same_anchor_mse/contracts.py`
- `common/quality_experiments/Q002_B6_roi_same_anchor_mse/joint_dataset.py`
- `common/quality_experiments/Q002_B6_roi_same_anchor_mse/run_q002_reconstruction.py`
- `common/quality_experiments/Q002_B6_roi_same_anchor_mse/training.py`
- `common/quality_experiments/Q002_B6_roi_same_anchor_mse/training_curves.py`
- `common/reconstruction_core/orchestration.py`
- `common/trad/evaluation/profiling/summarize_timing_profile.py`
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
- `configs/models_manifest.json`
- `pipeline/06_PROPOSED/run_stage.sh`
- `runtime/runtime_contract.json`

## Stage 07

Direct and discovered active paths:

- `common/evaluation_common/myocardium/build_native_bundle.py`
- `common/evaluation_common/myocardium/legacy_masks.py`
- `common/paper_experiments/joint_vs_2dfit/aha/current_fit_first_final_pose_bundle.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_adapter.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_aha16.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_legacy_roi.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_myocardial_roi_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_postprocess.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_r3_direct_final_pose_bundle.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_r3_fit_first_final_pose_bundle.py`
- `common/paper_experiments/joint_vs_2dfit/aha/cyj11_runner.py`
- `common/paper_experiments/joint_vs_2dfit/aha/derive_final_pose_mapping.py`
- `common/paper_experiments/joint_vs_2dfit/aha/final5_adapter.py`
- `common/paper_experiments/joint_vs_2dfit/aha/final5_aha16.py`
- `common/paper_experiments/joint_vs_2dfit/aha/final5_myocardial_roi_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/aha/legacy_aha16.py`
- `common/paper_experiments/joint_vs_2dfit/aha/paired_aha_sd.py`
- `common/paper_experiments/joint_vs_2dfit/aha/paired_final_pose_lv_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/aha/paired_final_pose_postprocess.py`
- `common/paper_experiments/joint_vs_2dfit/aha/paired_final_pose_through_plane.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/contracts.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/current_native_central_component_mask.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/direct_native_bundles.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/direct_native_endpoint_audit.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/evaluation.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final5_partitioned_native_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final5_publication_visualization.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final_cyj_reporting.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final_native_grid_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/final_sparse_native_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_mask_audit.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_native_geometry.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/fit_first_route_contract.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/formal_l100_sparse_manifest.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/frozen_native_support.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/heldout.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/l100_mask_paired_eval.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/matched_output.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_paired_final_pose.sh`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/visualization.py`
- `common/trad/evaluation/scmr/quality_control.py`
- `common/trad/evaluation/scmr/reference_2d.py`
- `common/trad/third_party/nesvor/nesvor/image/__init__.py`
- `common/trad/third_party/nesvor/nesvor/image/image.py`
- `common/trad/third_party/nesvor/nesvor/image/image_utils.py`
- `common/trad/third_party/nesvor/nesvor/utils/__init__.py`
- `common/trad/third_party/nesvor/nesvor/utils/logger.py`
- `common/trad/third_party/nesvor/nesvor/utils/loss.py`
- `common/trad/third_party/nesvor/nesvor/utils/misc.py`
- `pipeline/07_FINAL_POSE/run_stage.sh`

## Stage 08

Direct and discovered active paths:

- `common/paper_experiments/joint_vs_2dfit/phase2_clean/common_native.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/common_native_all_stack_roi_metrics.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/common_native_metrics_qc.py`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_all_stack_roi_metrics.sh`
- `common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/hhz_common_native_metrics_qc.sh`
- `pipeline/08_EVALUATION_AND_VISUALIZATION/run_stage.sh`

## Stage 09

Direct and discovered active paths:

- `configs/scientific_freeze_contract.json`
- `pipeline/09_INTEGRITY_AND_REPORTING/run_stage.sh`
- `scripts/verify_no_legacy_code_path.py`
- `scripts/verify_project_integrity.py`
- `scripts/verify_runtime_contract.py`
- `scripts/verify_scientific_freeze.py`

## Compatibility / future cleanup

- `common/mlp/modules/module_04_quantitative_inr/quantitative_inr.py` → `common/trad/modules/module_04_quantitative_inr/quantitative_inr.py`; retain facade until package consolidation.
- `common/mlp/modules/module_05_signal_decoder/frozen_decoder.py` → `common/trad/modules/module_05_signal_decoder`; record dynamic/proxy boundary before renaming.

Historical names such as `hhz` and `cyj11` remain generic-runtime wrappers where reached by stages 07/08; they are rename candidates for a future clean repository, not evidence of a subject-specific admission gate.
