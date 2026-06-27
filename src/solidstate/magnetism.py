"""A compact 2D Ising model implementation."""

from __future__ import annotations

import numpy as np


def initialize_spins(size: int, seed: int | None = None) -> np.ndarray:
    """Create a random square lattice of spins with values +/-1."""

    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(size, size))


def ising_energy(spins: np.ndarray, coupling: float = 1.0, field: float = 0.0) -> float:
    """Total energy of a periodic 2D Ising lattice."""

    spins = np.asarray(spins)
    neighbor_sum = np.roll(spins, 1, axis=0) + np.roll(spins, 1, axis=1)
    interaction = -coupling * np.sum(spins * neighbor_sum)
    zeeman = -field * np.sum(spins)
    return float(interaction + zeeman)


def magnetization(spins: np.ndarray) -> float:
    """Magnetization per spin."""

    return float(np.mean(spins))


def metropolis_sweep(
    spins: np.ndarray,
    temperature: float,
    coupling: float = 1.0,
    field: float = 0.0,
    rng: np.random.Generator | None = None,
) -> None:
    """Perform one Metropolis sweep in-place."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if rng is None:
        rng = np.random.default_rng()

    size = spins.shape[0]
    for _ in range(size * size):
        i = rng.integers(0, size)
        j = rng.integers(0, size)
        spin = spins[i, j]
        neighbors = spins[(i + 1) % size, j] + spins[(i - 1) % size, j] + spins[i, (j + 1) % size] + spins[i, (j - 1) % size]
        delta_e = 2.0 * spin * (coupling * neighbors + field)
        if delta_e <= 0.0 or rng.random() < np.exp(-delta_e / temperature):
            spins[i, j] *= -1


def simulate_ising(
    size: int,
    temperatures: np.ndarray,
    sweeps: int = 600,
    burn_in: int = 200,
    coupling: float = 1.0,
    field: float = 0.0,
    seed: int | None = 7,
) -> dict[str, np.ndarray]:
    """Simulate the 2D Ising model over a list of temperatures.

    Temperature is measured in units where k_B = 1.
    """

    rng = np.random.default_rng(seed)
    spins = initialize_spins(size, seed=seed)
    n_spins = size * size

    mean_energy = []
    mean_abs_magnetization = []
    heat_capacity = []
    susceptibility = []

    for temperature in np.asarray(temperatures, dtype=float):
        energies = []
        mags = []
        for sweep in range(sweeps):
            metropolis_sweep(spins, temperature, coupling=coupling, field=field, rng=rng)
            if sweep >= burn_in:
                energies.append(ising_energy(spins, coupling=coupling, field=field) / n_spins)
                mags.append(magnetization(spins))

        energies_array = np.asarray(energies)
        mags_array = np.asarray(mags)
        mean_energy.append(np.mean(energies_array))
        mean_abs_magnetization.append(np.mean(np.abs(mags_array)))
        heat_capacity.append(n_spins * np.var(energies_array) / temperature**2)
        susceptibility.append(n_spins * np.var(np.abs(mags_array)) / temperature)

    return {
        "temperature": np.asarray(temperatures, dtype=float),
        "energy": np.asarray(mean_energy),
        "abs_magnetization": np.asarray(mean_abs_magnetization),
        "heat_capacity": np.asarray(heat_capacity),
        "susceptibility": np.asarray(susceptibility),
    }

