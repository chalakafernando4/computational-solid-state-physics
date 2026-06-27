"""Crystal lattice utilities.

The primitive vectors are stored as rows of a 3x3 matrix. Reciprocal vectors use
the standard physics convention where a_i dot b_j = 2*pi delta_ij.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Lattice:
    """A Bravais lattice represented by primitive vectors."""

    name: str
    primitive_vectors: Array
    basis: Array | None = None

    def __post_init__(self) -> None:
        vectors = np.asarray(self.primitive_vectors, dtype=float)
        if vectors.shape != (3, 3):
            raise ValueError("primitive_vectors must be a 3x3 array")
        object.__setattr__(self, "primitive_vectors", vectors)
        if self.basis is not None:
            basis = np.asarray(self.basis, dtype=float)
            if basis.ndim != 2 or basis.shape[1] != 3:
                raise ValueError("basis must have shape (n_atoms, 3)")
            object.__setattr__(self, "basis", basis)

    @property
    def volume(self) -> float:
        """Primitive cell volume."""

        return float(abs(np.linalg.det(self.primitive_vectors)))

    @property
    def reciprocal_vectors(self) -> Array:
        """Reciprocal primitive vectors using the 2*pi convention."""

        return 2.0 * np.pi * np.linalg.inv(self.primitive_vectors).T

    def points(self, n_min: int = -1, n_max: int = 1, include_basis: bool = False) -> Array:
        """Return Cartesian lattice points from integer coefficients.

        Parameters
        ----------
        n_min, n_max:
            Inclusive integer range for each primitive vector coefficient.
        include_basis:
            If true and the lattice has a basis, include all basis atoms in each
            primitive cell.
        """

        coeffs = np.array(
            [[i, j, k] for i in range(n_min, n_max + 1) for j in range(n_min, n_max + 1) for k in range(n_min, n_max + 1)],
            dtype=float,
        )
        origins = coeffs @ self.primitive_vectors
        if not include_basis or self.basis is None:
            return origins
        return np.vstack([origin + self.basis @ self.primitive_vectors for origin in origins])

    def reciprocal_points(self, n_min: int = -2, n_max: int = 2) -> Array:
        """Return reciprocal lattice points from integer coefficients."""

        coeffs = np.array(
            [[h, k, l] for h in range(n_min, n_max + 1) for k in range(n_min, n_max + 1) for l in range(n_min, n_max + 1)],
            dtype=float,
        )
        return coeffs @ self.reciprocal_vectors

    def miller_normal(self, h: int, k: int, l: int) -> Array:
        """Return the Cartesian normal vector to the (hkl) plane."""

        normal = h * self.reciprocal_vectors[0] + k * self.reciprocal_vectors[1] + l * self.reciprocal_vectors[2]
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError("Miller indices cannot all be zero")
        return normal / norm


def simple_cubic(a: float = 1.0) -> Lattice:
    """Simple cubic primitive cell."""

    return Lattice(
        name="simple cubic",
        primitive_vectors=np.array(
            [
                [a, 0.0, 0.0],
                [0.0, a, 0.0],
                [0.0, 0.0, a],
            ]
        ),
    )


def body_centered_cubic(a: float = 1.0) -> Lattice:
    """Body-centered cubic primitive cell."""

    return Lattice(
        name="body-centered cubic",
        primitive_vectors=0.5
        * a
        * np.array(
            [
                [1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ]
        ),
    )


def face_centered_cubic(a: float = 1.0) -> Lattice:
    """Face-centered cubic primitive cell."""

    return Lattice(
        name="face-centered cubic",
        primitive_vectors=0.5
        * a
        * np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        ),
    )


def hexagonal(a: float = 1.0, c: float = 1.6) -> Lattice:
    """Hexagonal primitive cell."""

    return Lattice(
        name="hexagonal",
        primitive_vectors=np.array(
            [
                [a, 0.0, 0.0],
                [-0.5 * a, np.sqrt(3.0) * a / 2.0, 0.0],
                [0.0, 0.0, c],
            ]
        ),
    )


def tetragonal(a: float = 1.0, c: float = 1.5) -> Lattice:
    """Primitive tetragonal cell."""

    return Lattice(
        name="tetragonal",
        primitive_vectors=np.array(
            [
                [a, 0.0, 0.0],
                [0.0, a, 0.0],
                [0.0, 0.0, c],
            ]
        ),
    )


def orthorhombic(a: float = 1.0, b: float = 1.3, c: float = 1.7) -> Lattice:
    """Primitive orthorhombic cell."""

    return Lattice(
        name="orthorhombic",
        primitive_vectors=np.array(
            [
                [a, 0.0, 0.0],
                [0.0, b, 0.0],
                [0.0, 0.0, c],
            ]
        ),
    )


def diamond_cubic(a: float = 1.0) -> Lattice:
    """Diamond cubic structure using an FCC primitive lattice plus a basis."""

    fcc = face_centered_cubic(a)
    return Lattice(
        name="diamond cubic",
        primitive_vectors=fcc.primitive_vectors,
        basis=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
            ]
        ),
    )
