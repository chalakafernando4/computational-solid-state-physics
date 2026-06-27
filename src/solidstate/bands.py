"""Simple tight-binding band-structure utilities."""

from __future__ import annotations

import numpy as np


def tight_binding_1d(k: np.ndarray, a: float = 1.0, hopping: float = 1.0, onsite: float = 0.0) -> np.ndarray:
    """Nearest-neighbor 1D tight-binding dispersion."""

    k = np.asarray(k, dtype=float)
    return onsite - 2.0 * hopping * np.cos(k * a)


def square_lattice_dispersion(
    kx: np.ndarray,
    ky: np.ndarray,
    a: float = 1.0,
    hopping: float = 1.0,
    onsite: float = 0.0,
) -> np.ndarray:
    """Nearest-neighbor tight-binding dispersion on a square lattice."""

    return onsite - 2.0 * hopping * (np.cos(kx * a) + np.cos(ky * a))


def graphene_dispersion(kx: np.ndarray, ky: np.ndarray, a: float = 1.0, hopping: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor graphene pi-band dispersion.

    The convention here uses `a` as the nearest-neighbor carbon-carbon spacing.
    Energies are returned as valence and conduction branches.
    """

    root = np.sqrt(
        1.0
        + 4.0 * np.cos(1.5 * kx * a) * np.cos(np.sqrt(3.0) * ky * a / 2.0)
        + 4.0 * np.cos(np.sqrt(3.0) * ky * a / 2.0) ** 2
    )
    return -hopping * root, hopping * root


def high_symmetry_path(
    points: dict[str, tuple[float, float]],
    labels: list[str],
    points_per_segment: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Interpolate a 2D high-symmetry path.

    Parameters
    ----------
    points:
        Mapping from labels to 2D coordinates.
    labels:
        Ordered labels to visit, for example ["G", "X", "M", "G"].
    points_per_segment:
        Number of interpolated points per path segment.
    """

    if len(labels) < 2:
        raise ValueError("At least two labels are required")

    coords = []
    distances = []
    tick_positions = [0.0]
    total = 0.0

    for start_label, end_label in zip(labels[:-1], labels[1:]):
        start = np.asarray(points[start_label], dtype=float)
        end = np.asarray(points[end_label], dtype=float)
        segment = np.linspace(start, end, points_per_segment, endpoint=False)
        if coords:
            step_start = coords[-1]
        else:
            step_start = start
        for point in segment:
            if coords:
                total += float(np.linalg.norm(point - step_start))
            coords.append(point)
            distances.append(total)
            step_start = point
        total += float(np.linalg.norm(end - step_start))
        tick_positions.append(total)

    coords.append(np.asarray(points[labels[-1]], dtype=float))
    distances.append(total)
    return np.asarray(coords), np.asarray(distances), tick_positions


def dos_histogram(energies: np.ndarray, bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Estimate density of states using a normalized histogram."""

    hist, edges = np.histogram(np.ravel(energies), bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist

