"""Einstein and Debye heat-capacity models."""

from __future__ import annotations

import numpy as np

GAS_CONSTANT = 8.31446261815324


def _as_temperature_array(temperature: float | np.ndarray) -> np.ndarray:
    values = np.asarray(temperature, dtype=float)
    if np.any(values <= 0.0):
        raise ValueError("Temperature values must be positive")
    return values


def einstein_heat_capacity(temperature: float | np.ndarray, theta_e: float, atoms_per_formula_unit: float = 1.0) -> np.ndarray:
    """Molar Einstein heat capacity in J/(mol K)."""

    temperature = _as_temperature_array(temperature)
    x = theta_e / temperature
    exp_x = np.exp(np.clip(x, None, 700.0))
    denominator = np.expm1(np.clip(x, None, 700.0)) ** 2
    return 3.0 * atoms_per_formula_unit * GAS_CONSTANT * x**2 * exp_x / denominator


def debye_heat_capacity(
    temperature: float | np.ndarray,
    theta_d: float,
    atoms_per_formula_unit: float = 1.0,
    integration_points: int = 2000,
) -> np.ndarray:
    """Molar Debye heat capacity in J/(mol K)."""

    temperatures = np.atleast_1d(_as_temperature_array(temperature))
    heat_capacities = []
    for temp in temperatures:
        upper = theta_d / temp
        x = np.linspace(1.0e-8, upper, integration_points)
        exp_x = np.exp(np.clip(x, None, 700.0))
        integrand = x**4 * exp_x / np.expm1(np.clip(x, None, 700.0)) ** 2
        integral = np.trapz(integrand, x)
        heat_capacities.append(9.0 * atoms_per_formula_unit * GAS_CONSTANT * (temp / theta_d) ** 3 * integral)
    result = np.asarray(heat_capacities)
    if np.ndim(temperature) == 0:
        return result[0]
    return result
