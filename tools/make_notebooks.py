from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


def notebook(title: str, intro: str, code: str) -> dict:
    setup = """
from pathlib import Path
import sys

ROOT = Path.cwd()
if (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT / "src"))
elif (ROOT.parent / "src").exists():
    sys.path.insert(0, str(ROOT.parent / "src"))
"""
    return {
        "cells": [
            markdown_cell(f"# {title}\n\n{intro}"),
            code_cell(setup),
            code_cell(code),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOK_DEFINITIONS = {
    "01_lattices_and_reciprocal_space.ipynb": notebook(
        "Lattices And Reciprocal Space",
        "Build simple Bravais lattices, inspect primitive-cell volumes, and check reciprocal-space duality.",
        """
import numpy as np
from solidstate import simple_cubic, body_centered_cubic, face_centered_cubic, hexagonal

for lattice in [simple_cubic(), body_centered_cubic(), face_centered_cubic(), hexagonal()]:
    product = lattice.primitive_vectors @ lattice.reciprocal_vectors.T
    print(lattice.name, "volume =", lattice.volume)
    print(np.round(product, 3))
""",
    ),
    "02_phonons.ipynb": notebook(
        "Phonon Dispersion",
        "Compare monoatomic and diatomic one-dimensional chain dispersion relations.",
        """
import matplotlib.pyplot as plt
from solidstate.phonons import first_brillouin_zone, monoatomic_chain_dispersion, diatomic_chain_dispersion

k = first_brillouin_zone(n_points=600)
mono = monoatomic_chain_dispersion(k)
acoustic, optical = diatomic_chain_dispersion(k, mass_1=1.0, mass_2=2.5)

plt.figure(figsize=(7, 4))
plt.plot(k, mono, label="monoatomic")
plt.plot(k, acoustic, label="acoustic")
plt.plot(k, optical, label="optical")
plt.xlabel("k")
plt.ylabel("angular frequency")
plt.legend(frameon=False)
plt.show()
""",
    ),
    "03_tight_binding.ipynb": notebook(
        "Tight-Binding Bands",
        "Plot one-dimensional and square-lattice nearest-neighbor tight-binding bands.",
        """
import matplotlib.pyplot as plt
import numpy as np
from solidstate import tight_binding_1d, square_lattice_dispersion

k = np.linspace(-np.pi, np.pi, 600)
grid = np.linspace(-np.pi, np.pi, 200)
kx, ky = np.meshgrid(grid, grid)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(k, tight_binding_1d(k))
axes[0].set_xlabel("k")
axes[0].set_ylabel("energy / t")
contour = axes[1].contourf(kx, ky, square_lattice_dispersion(kx, ky), levels=40)
fig.colorbar(contour, ax=axes[1])
plt.show()
""",
    ),
    "04_semiconductors.ipynb": notebook(
        "Semiconductor Estimates",
        "Estimate effective density of states, intrinsic carrier concentration, and conductivity.",
        """
from solidstate import effective_density_of_states, intrinsic_carrier_concentration, conductivity

temperature = 300.0
nc = effective_density_of_states(temperature, effective_mass_ratio=1.08)
nv = effective_density_of_states(temperature, effective_mass_ratio=0.56)
ni = intrinsic_carrier_concentration(1.12, temperature, nc, nv)
sigma = conductivity(ni, ni, electron_mobility=0.135, hole_mobility=0.048)

print(f"Nc = {nc:.3e} m^-3")
print(f"Nv = {nv:.3e} m^-3")
print(f"ni = {ni:.3e} m^-3")
print(f"intrinsic conductivity = {sigma:.3e} S/m")
""",
    ),
    "05_heat_capacity.ipynb": notebook(
        "Heat Capacity Models",
        "Compare Debye, Einstein, and high-temperature Dulong-Petit behavior.",
        """
import matplotlib.pyplot as plt
import numpy as np
from solidstate import debye_heat_capacity, einstein_heat_capacity
from solidstate.heat_capacity import GAS_CONSTANT

temperature = np.linspace(2, 500, 260)
plt.figure(figsize=(7, 4))
plt.plot(temperature, debye_heat_capacity(temperature, theta_d=220), label="Debye")
plt.plot(temperature, einstein_heat_capacity(temperature, theta_e=220), label="Einstein")
plt.axhline(3 * GAS_CONSTANT, color="black", linestyle="--", label="Dulong-Petit")
plt.xlabel("temperature (K)")
plt.ylabel("heat capacity (J mol^-1 K^-1)")
plt.legend(frameon=False)
plt.show()
""",
    ),
    "06_ising_model.ipynb": notebook(
        "2D Ising Model",
        "Run a compact Monte Carlo scan and plot the transition region.",
        """
import matplotlib.pyplot as plt
import numpy as np
from solidstate import simulate_ising

temperatures = np.linspace(1.6, 3.4, 11)
result = simulate_ising(size=16, temperatures=temperatures, sweeps=180, burn_in=60, seed=11)

plt.figure(figsize=(7, 4))
plt.plot(result["temperature"], result["abs_magnetization"], marker="o", label="|M|")
plt.plot(result["temperature"], result["susceptibility"] / result["susceptibility"].max(), marker="o", label="scaled susceptibility")
plt.xlabel("temperature J/k_B")
plt.legend(frameon=False)
plt.show()
""",
    ),
    "07_brillouin_zones.ipynb": notebook(
        "Brillouin Zones",
        "Construct first Brillouin-zone polygons from reciprocal lattice bases.",
        """
import matplotlib.pyplot as plt
import numpy as np
from solidstate import brillouin_zone_2d, square_reciprocal_basis, triangular_reciprocal_basis

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, basis, title in [
    (axes[0], square_reciprocal_basis(), "square"),
    (axes[1], triangular_reciprocal_basis(), "triangular"),
]:
    zone = brillouin_zone_2d(basis)
    closed = np.vstack([zone, zone[0]])
    ax.plot(closed[:, 0], closed[:, 1], linewidth=2)
    ax.fill(zone[:, 0], zone[:, 1], alpha=0.2)
    ax.set_title(title)
    ax.set_aspect("equal")
plt.show()
""",
    ),
}


def main() -> None:
    for filename, content in NOTEBOOK_DEFINITIONS.items():
        path = NOTEBOOKS / filename
        path.write_text(json.dumps(content, indent=2), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

