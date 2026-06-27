"""Reciprocal-space and simple diffraction helpers."""

from __future__ import annotations

from collections import defaultdict
from math import asin, degrees, sqrt

import numpy as np


def reciprocal_vectors(primitive_vectors: np.ndarray) -> np.ndarray:
    """Return reciprocal vectors for primitive vectors stored as rows."""

    primitive_vectors = np.asarray(primitive_vectors, dtype=float)
    if primitive_vectors.shape != (3, 3):
        raise ValueError("primitive_vectors must be a 3x3 array")
    return 2.0 * np.pi * np.linalg.inv(primitive_vectors).T


def cubic_d_spacing(a: float, h: int, k: int, l: int) -> float:
    """Interplanar spacing for a cubic crystal."""

    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller indices cannot all be zero")
    return a / sqrt(h * h + k * k + l * l)


def bragg_two_theta(d_spacing: float, wavelength: float, order: int = 1) -> float | None:
    """Return Bragg diffraction angle 2 theta in degrees.

    Returns None when the reflection is kinematically forbidden because
    order*wavelength > 2*d.
    """

    argument = order * wavelength / (2.0 * d_spacing)
    if argument > 1.0:
        return None
    return 2.0 * degrees(asin(argument))


def allowed_reflection(lattice: str, h: int, k: int, l: int) -> bool:
    """Basic cubic reflection selection rules."""

    lattice_key = lattice.lower().replace("_", "-").replace(" ", "-")
    if h == 0 and k == 0 and l == 0:
        return False
    if lattice_key in {"sc", "simple-cubic", "primitive-cubic"}:
        return True
    if lattice_key in {"bcc", "body-centered-cubic"}:
        return (h + k + l) % 2 == 0
    if lattice_key in {"fcc", "face-centered-cubic"}:
        parity = [abs(h) % 2, abs(k) % 2, abs(l) % 2]
        return parity == [0, 0, 0] or parity == [1, 1, 1]
    raise ValueError(f"Unknown cubic lattice type: {lattice}")


def _canonical_hkl(h: int, k: int, l: int) -> tuple[int, int, int]:
    return tuple(sorted((abs(h), abs(k), abs(l)), reverse=True))


def powder_peaks_cubic(
    a: float,
    wavelength: float,
    lattice: str = "fcc",
    max_index: int = 6,
) -> list[dict[str, float | int | tuple[int, int, int]]]:
    """Generate schematic powder diffraction peaks for cubic crystals.

    Reflections with the same absolute sorted hkl are grouped. The returned
    intensity is a simple multiplicity proxy, not a realistic structure-factor
    calculation.
    """

    groups: dict[tuple[int, int, int], int] = defaultdict(int)
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                if not allowed_reflection(lattice, h, k, l):
                    continue
                groups[_canonical_hkl(h, k, l)] += 1

    peaks: list[dict[str, float | int | tuple[int, int, int]]] = []
    for hkl, multiplicity in groups.items():
        h, k, l = hkl
        d_spacing = cubic_d_spacing(a, h, k, l)
        two_theta = bragg_two_theta(d_spacing, wavelength)
        if two_theta is None:
            continue
        peaks.append(
            {
                "hkl": hkl,
                "d_spacing": d_spacing,
                "two_theta": two_theta,
                "multiplicity": multiplicity,
                "relative_intensity": multiplicity,
            }
        )
    return sorted(peaks, key=lambda peak: float(peak["two_theta"]))

