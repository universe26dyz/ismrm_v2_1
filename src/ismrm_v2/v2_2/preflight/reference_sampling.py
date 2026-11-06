"""Fail-closed same-world-coordinate reference sampling for reliability audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RegularGridReference:
    """A complete, regular RAS-mm lattice with ten co-registered channels."""

    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    values: np.ndarray  # [nx, ny, nz, 10]
    coordinate_convention: str = "world RAS mm; axis order x,y,z"

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(int(item) for item in self.values.shape)  # type: ignore[return-value]


def _axis(values: np.ndarray, name: str, tolerance: float) -> np.ndarray:
    axis = np.unique(np.asarray(values, dtype=np.float64))
    if axis.size == 0 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
        raise ValueError(f"{name} axis must contain finite increasing coordinates.")
    if axis.size > 2:
        spacing = np.diff(axis)
        if not np.allclose(spacing, spacing[0], rtol=0.0, atol=tolerance):
            raise ValueError("SAX reference geometry is not a regular physical lattice; refusing volume interpolation.")
    return axis


def build_regular_grid_reference(
    points_ras_mm: np.ndarray,
    fingerprints: np.ndarray,
    *,
    axes: Sequence[np.ndarray] | None = None,
    tolerance: float = 1e-5,
) -> RegularGridReference:
    """Admit a complete lattice; duplicate, missing, or irregular points fail closed."""

    points = np.asarray(points_ras_mm, dtype=np.float64)
    values = np.asarray(fingerprints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("reference points must be finite [N,3] RAS-mm coordinates.")
    if values.shape != (points.shape[0], 10) or not np.isfinite(values).all():
        raise ValueError("reference fingerprints must be finite [N,10].")
    if tolerance < 0:
        raise ValueError("lattice tolerance must be non-negative.")
    if axes is None:
        lattice_axes = tuple(_axis(points[:, index], name, tolerance) for index, name in enumerate(("x", "y", "z")))
    else:
        if len(axes) != 3:
            raise ValueError("axes must contain x, y, z arrays.")
        lattice_axes = tuple(_axis(np.asarray(axis, dtype=np.float64), name, tolerance) for axis, name in zip(axes, ("x", "y", "z")))
        if any(not np.all(np.min(np.abs(points[:, i, None] - axis[None, :]), axis=0) <= tolerance) for i, axis in enumerate(lattice_axes)):
            raise ValueError("provided lattice axes do not contain every reference coordinate.")
    expected = int(np.prod([axis.size for axis in lattice_axes]))
    if expected != points.shape[0]:
        raise ValueError("SAX reference geometry is sparse or incomplete; refusing extrapolating interpolation.")
    grid = np.full((*[axis.size for axis in lattice_axes], 10), np.nan, dtype=np.float64)
    occupied: set[tuple[int, int, int]] = set()
    for point, fingerprint in zip(points, values):
        index = tuple(int(np.argmin(np.abs(axis - coordinate))) for axis, coordinate in zip(lattice_axes, point))
        if any(abs(axis[position] - coordinate) > tolerance for axis, position, coordinate in zip(lattice_axes, index, point)):
            raise ValueError("reference point is not on the admitted physical lattice.")
        if index in occupied:
            raise ValueError("duplicate SAX reference coordinate; geometry is not a function on a regular lattice.")
        occupied.add(index)
        grid[index] = fingerprint
    if len(occupied) != expected or not np.isfinite(grid).all():
        raise ValueError("SAX reference lattice has missing or non-finite cells.")
    return RegularGridReference(tuple(axis.copy() for axis in lattice_axes), grid)


def sample_regular_grid_trilinear(
    reference: RegularGridReference,
    query_ras_mm: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Sample only inside the lattice support; invalid queries are never extrapolated."""

    query = np.asarray(query_ras_mm, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 3 or not np.isfinite(query).all():
        raise ValueError("query coordinates must be finite [N,3] RAS-mm points.")
    if tolerance < 0:
        raise ValueError("sampling tolerance must be non-negative.")
    axes = reference.axes
    valid = np.ones(query.shape[0], dtype=bool)
    for coordinate, axis in zip(query.T, axes):
        valid &= (coordinate >= axis[0] - tolerance) & (coordinate <= axis[-1] + tolerance)
    sampled = np.full((query.shape[0], 10), np.nan, dtype=np.float64)
    if not valid.any():
        return sampled, valid, {"support_min_ras_mm": np.array([axis[0] for axis in axes]), "support_max_ras_mm": np.array([axis[-1] for axis in axes]), "invalid_reason": np.where(~valid, "outside_reference_fov", "").astype(object)}
    clipped = np.clip(query[valid], np.array([axis[0] for axis in axes]), np.array([axis[-1] for axis in axes]))
    lower, upper, weights = [], [], []
    for coordinate, axis in zip(clipped.T, axes):
        right = np.searchsorted(axis, coordinate, side="right")
        lo = np.clip(right - 1, 0, axis.size - 1)
        hi = np.clip(right, 0, axis.size - 1)
        denominator = axis[hi] - axis[lo]
        weight = np.divide(coordinate - axis[lo], denominator, out=np.zeros_like(coordinate), where=denominator != 0)
        lower.append(lo); upper.append(hi); weights.append(weight)
    result = np.zeros((clipped.shape[0], 10), dtype=np.float64)
    for bx in (0, 1):
        for by in (0, 1):
            for bz in (0, 1):
                coefficient = ((1 - weights[0]) if bx == 0 else weights[0]) * ((1 - weights[1]) if by == 0 else weights[1]) * ((1 - weights[2]) if bz == 0 else weights[2])
                result += coefficient[:, None] * reference.values[(lower[0] if bx == 0 else upper[0]), (lower[1] if by == 0 else upper[1]), (lower[2] if bz == 0 else upper[2])]
    sampled[valid] = result
    reason = np.where(valid, "inside_reference_fov", "outside_reference_fov").astype(object)
    return sampled, valid, {"support_min_ras_mm": np.array([axis[0] for axis in axes]), "support_max_ras_mm": np.array([axis[-1] for axis in axes]), "invalid_reason": reason}
