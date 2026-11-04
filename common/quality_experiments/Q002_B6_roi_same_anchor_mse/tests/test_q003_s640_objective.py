import pytest
import torch


def _stack_weights():
    return {0: 2.0, 1: 0.5}


def test_q003_cosine_is_anchor_level_dimension_ten_and_uses_stack_weights():
    from ismrm_v2.proposed.q002.training import joint_fingerprint_objective

    prediction = torch.tensor([[1.0] + [0.0] * 9, [0.0, 1.0] + [0.0] * 8])
    observed = torch.tensor([[2.0] + [0.0] * 9, [1.0, 0.0] + [0.0] * 8])
    result = joint_fingerprint_objective(prediction, observed, torch.tensor([0, 1]), _stack_weights(), cosine_weight=1.0, cosine_epsilon=1e-8)
    assert torch.allclose(result["fingerprint_cosine_loss"], torch.tensor(0.25))
    assert torch.allclose(result["weighted_fingerprint_cosine_loss"], torch.tensor(0.25))
    assert result["pred_norm_clamp_count"] == 0
    assert result["obs_norm_clamp_count"] == 0


def test_q003_cosine_has_expected_identity_rescaling_orthogonality_and_separate_clamps():
    from ismrm_v2.proposed.q002.training import joint_fingerprint_objective

    identical = joint_fingerprint_objective(torch.tensor([[3.0] + [0.0] * 9]), torch.tensor([[6.0] + [0.0] * 9]), torch.tensor([0]), {0: 1.0}, cosine_weight=1.0, cosine_epsilon=1e-8)
    orthogonal = joint_fingerprint_objective(torch.tensor([[1.0] + [0.0] * 9]), torch.tensor([[0.0, 1.0] + [0.0] * 8]), torch.tensor([0]), {0: 1.0}, cosine_weight=1.0, cosine_epsilon=1e-8)
    clamped = joint_fingerprint_objective(torch.zeros((1, 10)), torch.tensor([[1.0] + [0.0] * 9]), torch.tensor([0]), {0: 1.0}, cosine_weight=1.0, cosine_epsilon=1e-8)
    assert torch.allclose(identical["fingerprint_cosine_loss"], torch.zeros(()))
    assert torch.allclose(orthogonal["fingerprint_cosine_loss"], torch.ones(()))
    assert clamped["pred_norm_clamp_count"] == 1
    assert clamped["obs_norm_clamp_count"] == 0
    assert clamped["any_norm_clamp_count"] == 1


def test_q003_lambda_zero_matches_q002_mse_value_and_gradients_exactly():
    from ismrm_v2.proposed.q002.training import joint_fingerprint_objective, joint_vector_mse

    prediction = torch.tensor([[1.0] * 10, [2.0] * 10], requires_grad=True)
    observed = torch.tensor([[0.0] * 10, [1.0] * 10])
    stack_idx = torch.tensor([0, 1])
    expected = joint_vector_mse(prediction, observed, stack_idx, _stack_weights())
    expected.backward()
    expected_gradient = prediction.grad.detach().clone()
    prediction.grad = None
    actual = joint_fingerprint_objective(prediction, observed, stack_idx, _stack_weights(), cosine_weight=0.0, cosine_epsilon=1e-8)["total_data_loss"]
    actual.backward()
    assert torch.equal(actual, expected.detach())
    assert torch.equal(prediction.grad, expected_gradient)


def test_q003_rejects_nonfinite_fingerprints_and_has_immutable_cosine_route():
    from ismrm_v2.proposed.q002.contracts import Q003S640_EXPERIMENT_ID, build_q002_route_config
    from ismrm_v2.proposed.q002.training import joint_fingerprint_objective

    with pytest.raises(ValueError, match="finite"):
        joint_fingerprint_objective(torch.full((1, 10), float("nan")), torch.ones((1, 10)), torch.tensor([0]), {0: 1.0}, cosine_weight=1.0, cosine_epsilon=1e-8)
    route = build_q002_route_config({"training": {}, "decoder": {}}, cropped_prepared_root="/cropped", experiment_id=Q003S640_EXPERIMENT_ID, anchor_batch_size=640)
    assert route["route"]["parent_experiment"] == "Q002S640_B6_roi_same_anchor_mse"
    assert route["route"]["fingerprint_cosine"] == {"enabled": True, "weight": 1.0, "epsilon": 1.0e-8}


def test_unknown_stack_id_hard_fails_instead_of_using_uninitialized_weight():
    from ismrm_v2.proposed.q002.training import joint_vector_mse

    with pytest.raises(ValueError, match="stack weights"):
        joint_vector_mse(torch.ones((1, 10)), torch.zeros((1, 10)), torch.tensor([99]), {0: 1.0})


def test_q003_evaluation_provenance_marks_cosine_as_the_training_objective():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.evaluation import evaluation_plan
    from quality_experiments.Q002_B6_roi_same_anchor_mse.evaluate_q003_s640 import build_parser

    assert evaluation_plan("checkpoint", fingerprint_cosine_training_objective=True)["fingerprint_cosine"]["training_objective"] is True
    args = build_parser().parse_args(["--run-root", "/q003", "--q002-s640-run-root", "/q002", "--b6-map-root", "/b6", "--b6-d2-root", "/d2", "--reference-root", "/ref", "--preprocessed-root", "/pre", "--output", "/out"])
    assert args.q002_s640_run_root == "/q002"
