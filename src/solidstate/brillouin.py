"""Brillouin-zone construction helpers."""

from __future__ import annotations

import numpy as np


def _clip_polygon_with_half_plane(polygon: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Clip a convex polygon to the half-plane x dot normal <= offset."""

    if len(polygon) == 0:
        return polygon

    clipped: list[np.ndarray] = []
    previous = polygon[-1]
    previous_inside = float(previous @ normal) <= offset + 1.0e-12

    for current in polygon:
        current_inside = float(current @ normal) <= offset + 1.0e-12
        if current_inside != previous_inside:
            direction = current - previous
            denominator = float(direction @ normal)
            if abs(denominator) > 1.0e-14:
                t = (offset - float(previous @ normal)) / denominator
                clipped.append(previous + t * direction)
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside

    if not clipped:
        return np.empty((0, 2), dtype=float)
    return np.asarray(clipped, dtype=float)


def brillouin_zone_2d(reciprocal_basis: np.ndarray, shell: int = 2) -> np.ndarray:
    """Return the first Brillouin zone polygon for a 2D reciprocal lattice.

    The construction clips the plane by perpendicular bisectors to nearby
    reciprocal-lattice points. Returned vertices are ordered around the origin.
    """

    basis = np.asarray(reciprocal_basis, dtype=float)
    if basis.shape != (2, 2):
        raise ValueError("reciprocal_basis must have shape (2, 2)")

    reciprocal_points = []
    for i in range(-shell, shell + 1):
        for j in range(-shell, shell + 1):
            if i == 0 and j == 0:
                continue
            reciprocal_points.append(i * basis[0] + j * basis[1])
    reciprocal_points = sorted(reciprocal_points, key=lambda point: float(point @ point))

    radius = 2.0 * max(float(np.linalg.norm(point)) for point in reciprocal_points)
    polygon = np.array(
        [
            [-radius, -radius],
            [radius, -radius],
            [radius, radius],
            [-radius, radius],
        ],
        dtype=float,
    )

    for point in reciprocal_points:
        offset = 0.5 * float(point @ point)
        polygon = _clip_polygon_with_half_plane(polygon, point, offset)
        if len(polygon) == 0:
            break

    center = np.mean(polygon, axis=0)
    angles = np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0])
    return polygon[np.argsort(angles)]


def square_reciprocal_basis(a: float = 1.0) -> np.ndarray:
    """2D square-lattice reciprocal basis."""

    return 2.0 * np.pi / a * np.array([[1.0, 0.0], [0.0, 1.0]])


def triangular_reciprocal_basis(a: float = 1.0) -> np.ndarray:
    """2D triangular-lattice reciprocal basis."""

    return 2.0 * np.pi / a * np.array([[1.0, -1.0 / np.sqrt(3.0)], [0.0, 2.0 / np.sqrt(3.0)]])


def graphene_reciprocal_basis(a: float = 1.0) -> np.ndarray:
    """2D honeycomb/graphene reciprocal basis for nearest-neighbor spacing a."""

    real_a = np.sqrt(3.0) * a
    return triangular_reciprocal_basis(real_a)


def polygon_area(vertices: np.ndarray) -> float:
    """Area of a 2D polygon using the shoelace formula."""

    vertices = np.asarray(vertices, dtype=float)
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
