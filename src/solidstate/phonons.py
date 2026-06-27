"""Phonon dispersion models for simple one-dimensional chains."""

from __future__ import annotations

import numpy as np


def first_brillouin_zone(a: float = 1.0, n_points: int = 500) -> np.ndarray:
    """Return k values from -pi/a to pi/a."""

    return np.linspace(-np.pi / a, np.pi / a, n_points)


def monoatomic_chain_dispersion(k: np.ndarray, a: float = 1.0, spring_constant: float = 1.0, mass: float = 1.0) -> np.ndarray:
    """Angular frequency for a 1D monoatomic chain."""

    k = np.asarray(k, dtype=float)
    return 2.0 * np.sqrt(spring_constant / mass) * np.abs(np.sin(0.5 * k * a))


def diatomic_chain_dispersion(
    k: np.ndarray,
    a: float = 1.0,
    spring_constant: float = 1.0,
    mass_1: float = 1.0,
    mass_2: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Acoustic and optical branches for a 1D diatomic chain.

    The unit-cell spacing is `a`. The returned arrays are angular frequencies.
    """

    k = np.asarray(k, dtype=float)
    prefactor = spring_constant / mass_1 + spring_constant / mass_2
    discriminant = prefactor**2 - 4.0 * spring_constant**2 / (mass_1 * mass_2) * np.sin(0.5 * k * a) ** 2
    discriminant = np.clip(discriminant, 0.0, None)
    acoustic_sq = prefactor - np.sqrt(discriminant)
    optical_sq = prefactor + np.sqrt(discriminant)
    return np.sqrt(np.clip(acoustic_sq, 0.0, None)), np.sqrt(np.clip(optical_sq, 0.0, None))


def group_velocity(k: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Numerical group velocity d omega / d k."""

    return np.gradient(np.asarray(omega, dtype=float), np.asarray(k, dtype=float))

