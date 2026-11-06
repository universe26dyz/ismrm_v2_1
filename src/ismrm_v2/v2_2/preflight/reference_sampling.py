"""Fail-closed same-world-coordinate reference sampling for reliability audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SlicePlane:
    """One prepared SAX slice in its frozen post-HB1 world geometry."""

    group_idx: int
    values: np.ndarray  # [H,W,C]
    joint_valid: np.ndarray  # [H,W]
    row_spacing_mm: float
    col_spacing_mm: float
    rotation: np.ndarray  # local -> world RAS rotation
    translation: np.ndarray  # trans-first axis-angle translation parameter
    origin_ras_mm: np.ndarray
    normal_ras: np.ndarray


@dataclass(frozen=True)
class SliceStackReference:
    planes: tuple[SlicePlane, ...]
    orientation_spread_deg: float
    median_inter_plane_angle_deg: float
    coordinate_convention: str = "local [column,row,slice] mm -> world RAS mm; frozen post-HB1 pose"


def _pose_matrices(axisangle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert NeSVoR trans-first axis-angle poses to R,T without changing convention."""
    import torch
    from nesvor.transform import RigidTransform

    poses = torch.as_tensor(axisangle, dtype=torch.float64)
    if poses.ndim != 2 or poses.shape[1] != 6 or not torch.isfinite(poses).all():
        raise ValueError("SAX group poses must be finite [G,6] trans-first axis-angle values.")
    matrices = RigidTransform(poses, trans_first=True).matrix(trans_first=True).detach().cpu().numpy()
    return matrices[:, :, :3], matrices[:, :, 3]


def build_slice_stack_reference(
    observation_path: str,
    group_axisangle_physical: np.ndarray,
    *,
    stack_idx: int = 0,
    plane_tolerance_mm: float = 1e-5,
) -> SliceStackReference:
    """Build a geometry-aware SAX sampler from full prepared observation images."""
    path = str(observation_path)
    with np.load(path, allow_pickle=False) as data:
        required = {"images", "masks", "group_idx", "weight_idx", "stack_idx", "pixel_spacing_rc_mm"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"prepared SAX observation lacks {sorted(missing)}")
        images = np.asarray(data["images"], dtype=np.float64)
        masks = np.asarray(data["masks"], dtype=bool)
        groups = np.asarray(data["group_idx"], dtype=np.int64)
        weights = np.asarray(data["weight_idx"], dtype=np.int64)
        stacks = np.asarray(data["stack_idx"], dtype=np.int64)
        spacing = np.asarray(data["pixel_spacing_rc_mm"], dtype=np.float64)
    if images.ndim != 3 or masks.shape != images.shape or groups.shape != (images.shape[0],) or weights.shape != groups.shape or stacks.shape != groups.shape or spacing.shape != (images.shape[0], 2):
        raise ValueError("prepared SAX arrays have inconsistent shapes")
    poses = np.asarray(group_axisangle_physical, dtype=np.float64)
    rotations, translations = _pose_matrices(poses)
    planes: list[SlicePlane] = []
    for group in np.unique(groups):
        rows = np.flatnonzero((groups == group) & (stacks == stack_idx))
        if rows.size != 10 or not np.array_equal(np.sort(weights[rows]), np.arange(10)):
            raise ValueError(f"SAX group {group} must contain exactly one complete weight 0..9")
        if int(group) >= poses.shape[0]:
            raise ValueError(f"SAX group {group} has no matching frozen geometry pose")
        ordered = rows[np.argsort(weights[rows], kind="stable")]
        if not np.allclose(spacing[ordered], spacing[ordered[0]], rtol=0, atol=plane_tolerance_mm):
            raise ValueError(f"SAX group {group} weights disagree on pixel spacing")
        values = np.moveaxis(images[ordered], 0, -1)
        valid = masks[ordered].all(axis=0) & np.isfinite(values).all(axis=-1)
        rotation = rotations[int(group)]
        normal = rotation[:, 2]
        normal /= np.linalg.norm(normal)
        translation = translations[int(group)]
        origin = rotation @ translation
        planes.append(SlicePlane(int(group), values, valid, float(spacing[ordered[0], 0]), float(spacing[ordered[0], 1]), rotation, translation, origin, normal))
    if not planes:
        raise ValueError("prepared SAX observation has no active SAX groups")
    reference_normal = planes[0].normal_ras
    oriented: list[SlicePlane] = []
    angles = []
    for plane in planes:
        normal = plane.normal_ras.copy()
        if float(np.dot(normal, reference_normal)) < 0:
            normal = -normal
        angles.append(np.degrees(np.arccos(np.clip(float(np.dot(normal, reference_normal)), -1.0, 1.0))))
        oriented.append(SlicePlane(plane.group_idx, plane.values, plane.joint_valid, plane.row_spacing_mm, plane.col_spacing_mm, plane.rotation, plane.translation, plane.origin_ras_mm, normal))
    pairwise = []
    for left in range(len(oriented)):
        for right in range(left + 1, len(oriented)):
            pairwise.append(np.degrees(np.arccos(np.clip(abs(float(np.dot(oriented[left].normal_ras, oriented[right].normal_ras))), -1.0, 1.0))))
    return SliceStackReference(tuple(oriented), float(max(angles)), float(np.median(pairwise) if pairwise else 0.0), "local [column,row,slice] mm -> world RAS mm; frozen post-HB1 pose")


def _bilinear_plane(plane: SlicePlane, world_point: np.ndarray, *, tolerance_mm: float) -> tuple[np.ndarray, bool, float]:
    local = plane.rotation.T @ world_point - plane.translation
    height, width = plane.values.shape[:2]
    col = local[0] / plane.col_spacing_mm + (width - 1) / 2.0
    row = local[1] / plane.row_spacing_mm + (height - 1) / 2.0
    if row < -tolerance_mm or row > height - 1 + tolerance_mm or col < -tolerance_mm or col > width - 1 + tolerance_mm:
        return np.full(plane.values.shape[-1], np.nan), False, float(np.dot(world_point - plane.origin_ras_mm, plane.normal_ras))
    row = float(np.clip(row, 0, height - 1)); col = float(np.clip(col, 0, width - 1))
    r0, c0 = int(np.floor(row)), int(np.floor(col)); r1, c1 = min(r0 + 1, height - 1), min(c0 + 1, width - 1)
    wr, wc = row - r0, col - c0
    corners = ((r0, c0, (1 - wr) * (1 - wc)), (r0, c1, (1 - wr) * wc), (r1, c0, wr * (1 - wc)), (r1, c1, wr * wc))
    if any(not plane.joint_valid[r, c] for r, c, _ in corners):
        return np.full(plane.values.shape[-1], np.nan), False, float(np.dot(world_point - plane.origin_ras_mm, plane.normal_ras))
    value = sum(weight * plane.values[r, c] for r, c, weight in corners)
    distance = float(np.dot(world_point - plane.origin_ras_mm, plane.normal_ras))
    return np.asarray(value), bool(np.isfinite(value).all()), distance


def sample_slice_stack(
    reference: SliceStackReference,
    query_ras_mm: np.ndarray,
    *,
    plane_tolerance_mm: float = 1e-5,
    in_plane_tolerance_mm: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Bilinear-in-plane plus distance-weighted bracketing-plane interpolation."""
    query = np.asarray(query_ras_mm, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 3 or not np.isfinite(query).all():
        raise ValueError("query coordinates must be finite [N,3]")
    channels = int(reference.planes[0].values.shape[-1])
    sampled = np.full((query.shape[0], channels), np.nan); valid = np.zeros(query.shape[0], dtype=bool); distances = np.full((query.shape[0], len(reference.planes)), np.nan)
    support_groups = np.full((query.shape[0], 2), -1, dtype=np.int64); support_weights = np.zeros((query.shape[0], 2), dtype=np.float64); in_plane_valid = np.zeros(query.shape[0], dtype=bool); exact_plane = np.zeros(query.shape[0], dtype=bool); bracketed = np.zeros(query.shape[0], dtype=bool)
    for index, point in enumerate(query):
        candidates: list[tuple[float, int, np.ndarray]] = []
        positive: list[tuple[float, int, np.ndarray]] = []; negative: list[tuple[float, int, np.ndarray]] = []
        for plane_index, plane in enumerate(reference.planes):
            value, plane_valid, distance = _bilinear_plane(plane, point, tolerance_mm=in_plane_tolerance_mm)
            distances[index, plane_index] = distance
            if not plane_valid:
                continue
            in_plane_valid[index] = True
            if abs(distance) <= plane_tolerance_mm:
                candidates.append((abs(distance), plane_index, value))
            elif distance < 0:
                negative.append((abs(distance), plane_index, value))
            else:
                positive.append((abs(distance), plane_index, value))
        if candidates:
            selected = min(candidates, key=lambda item: item[0]); selected_index = selected[1]
            sampled[index] = selected[2]; valid[index] = True; exact_plane[index] = True; support_groups[index] = (reference.planes[selected_index].group_idx, -1); support_weights[index] = (1.0, 0.0); continue
        if negative and positive:
            d0, negative_index, v0 = min(negative, key=lambda item: item[0]); d1, positive_index, v1 = min(positive, key=lambda item: item[0])
            sampled[index] = (d1 * v0 + d0 * v1) / (d0 + d1); valid[index] = True; bracketed[index] = True
            ordered_support = [(negative_index, reference.planes[negative_index].group_idx, d1 / (d0 + d1)), (positive_index, reference.planes[positive_index].group_idx, d0 / (d0 + d1))]
            # Keep support metadata in deterministic lower-to-upper plane order;
            # signed distances depend on the normal orientation and are not a
            # stable ordering key across frozen pose conventions.
            reference_axis = reference.planes[0].normal_ras
            ordered_support.sort(key=lambda item: float(np.dot(reference.planes[item[0]].origin_ras_mm, reference_axis)))
            support_groups[index] = (ordered_support[0][1], ordered_support[1][1]); support_weights[index] = (ordered_support[0][2], ordered_support[1][2])
    support = {"signed_plane_distance_mm": distances, "support_group_idx": support_groups, "support_weights": support_weights, "in_plane_valid": in_plane_valid, "exact_plane": exact_plane, "bracketed": bracketed, "orientation_spread_deg": np.asarray(reference.orientation_spread_deg), "median_inter_plane_angle_deg": np.asarray(reference.median_inter_plane_angle_deg), "invalid_reason": np.where(valid, "inside_bilinear_and_bracketed_support", "no_valid_plane_or_bracket").astype(object)}
    return sampled, valid, support


def replace_slice_values(reference: SliceStackReference, values_by_group: dict[int, np.ndarray]) -> SliceStackReference:
    """Reuse frozen slice geometry for an auxiliary co-registered channel tensor."""
    planes = []
    for plane in reference.planes:
        values = np.asarray(values_by_group[plane.group_idx], dtype=np.float64)
        if values.shape[:2] != plane.values.shape[:2] or values.ndim != 3 or not np.isfinite(values).all():
            raise ValueError("auxiliary slice values must be finite [H,W,C] and match each image grid")
        planes.append(SlicePlane(plane.group_idx, values, plane.joint_valid, plane.row_spacing_mm, plane.col_spacing_mm, plane.rotation, plane.translation, plane.origin_ras_mm, plane.normal_ras))
    return SliceStackReference(tuple(planes), reference.orientation_spread_deg, reference.median_inter_plane_angle_deg, reference.coordinate_convention)


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
