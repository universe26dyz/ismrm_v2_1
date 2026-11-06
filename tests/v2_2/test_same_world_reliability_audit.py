import numpy as np
import torch

from ismrm_v2.v2_2.preflight.discrepancy import (
    fit_joint_robust_normalization,
    normalized_l1_discrepancy,
    srpan_gate,
)
from ismrm_v2.v2_2.preflight.reference_sampling import (
    build_regular_grid_reference,
    sample_regular_grid_trilinear,
)


def test_trilinear_sampling_recovers_ten_channel_linear_field():
    axes = (np.array([0.0, 1.0]), np.array([0.0, 2.0]), np.array([0.0, 3.0]))
    points = np.array([[x, y, z] for z in axes[2] for y in axes[1] for x in axes[0]])
    values = np.column_stack([2.0 * points[:, 0] + (k + 1) * points[:, 1] - 0.5 * points[:, 2] for k in range(10)])
    grid = build_regular_grid_reference(points, values)
    sampled, valid, _ = sample_regular_grid_trilinear(grid, np.array([[0.25, 0.5, 1.5], [9.0, 0.0, 0.0]]))
    expected = np.array([[2.0 * .25 + (k + 1) * .5 - .5 * 1.5 for k in range(10)], [np.nan] * 10])
    assert valid.tolist() == [True, False]
    assert np.allclose(sampled[0], expected[0])
    assert np.isnan(sampled[1]).all()


def test_normalized_l1_and_gate_are_explicit_and_monotone():
    target = np.ones((2, 10))
    reference = np.vstack((target[0], np.full(10, 3.0)))
    stats = fit_joint_robust_normalization(target, reference, low_quantile=0.0, high_quantile=1.0)
    values, valid = normalized_l1_discrepancy(target, reference, stats)
    assert valid.tolist() == [True, True]
    assert np.isclose(values[0], 0.0)
    gate = srpan_gate(np.array([0.0, 1.0, 2.0]), tau=1.0, progress=0.5)
    assert np.isclose(gate[0], 1.0)
    assert np.all(np.diff(gate) < 0)


def test_same_world_geometry_transform_is_cpu_safe():
    points = np.array([[0., 0., 0.], [1., 0., 0.]])
    values = np.ones((2, 10))
    grid = build_regular_grid_reference(points, values, axes=(np.array([0., 1.]), np.array([0.]), np.array([0.])))
    sampled, valid, _ = sample_regular_grid_trilinear(grid, torch.tensor([[1., 0., 0.] ]).numpy())
    assert valid[0]
    assert np.allclose(sampled[0], 1.0)
