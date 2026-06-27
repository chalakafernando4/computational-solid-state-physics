"""Small semiconductor-physics calculation helpers using SI units."""

from __future__ import annotations

import numpy as np

ELECTRON_CHARGE = 1.602176634e-19
BOLTZMANN = 1.380649e-23
HBAR = 1.054571817e-34
ELECTRON_MASS = 9.1093837015e-31


def effective_density_of_states(temperature: float, effective_mass_ratio: float, degeneracy: int = 2) -> float:
    """Effective density of states in m^-3.

    `effective_mass_ratio` is m*/m_e.
    """

    effective_mass = effective_mass_ratio * ELECTRON_MASS
    return float(degeneracy * (effective_mass * BOLTZMANN * temperature / (2.0 * np.pi * HBAR**2)) ** 1.5)


def intrinsic_carrier_concentration(
    band_gap_ev: float,
    temperature: float,
    nc: float,
    nv: float,
) -> float:
    """Intrinsic carrier concentration in m^-3."""

    band_gap_joule = band_gap_ev * ELECTRON_CHARGE
    exponent = -band_gap_joule / (2.0 * BOLTZMANN * temperature)
    return float(np.sqrt(nc * nv) * np.exp(exponent))


def intrinsic_fermi_level_offset(temperature: float, nc: float, nv: float) -> float:
    """Intrinsic Fermi-level shift from midgap in eV.

    Positive values move the intrinsic level upward toward the conduction band.
    """

    return float(0.5 * (BOLTZMANN * temperature / ELECTRON_CHARGE) * np.log(nv / nc))


def conductivity(n: float, p: float, electron_mobility: float, hole_mobility: float) -> float:
    """Electrical conductivity in S/m.

    Carrier densities are in m^-3 and mobilities are in m^2/(V s).
    """

    return float(ELECTRON_CHARGE * (n * electron_mobility + p * hole_mobility))

