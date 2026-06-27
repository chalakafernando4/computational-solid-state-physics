from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _paths import add_src_to_path

add_src_to_path()

from solidstate import debye_heat_capacity, einstein_heat_capacity
from solidstate.heat_capacity import GAS_CONSTANT


OUTDIR = Path(__file__).resolve().parents[1] / "outputs"
OUTDIR.mkdir(exist_ok=True)


def main() -> None:
    temperatures = np.linspace(2.0, 500.0, 300)
    theta = 220.0
    cv_debye = debye_heat_capacity(temperatures, theta_d=theta)
    cv_einstein = einstein_heat_capacity(temperatures, theta_e=theta)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(temperatures, cv_debye, label="Debye", color="tab:blue")
    ax.plot(temperatures, cv_einstein, label="Einstein", color="tab:orange")
    ax.axhline(3.0 * GAS_CONSTANT, color="black", linestyle="--", linewidth=1, label="Dulong-Petit")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("molar heat capacity (J mol^-1 K^-1)")
    ax.set_title("Lattice heat capacity models")
    ax.legend(frameon=False)
    fig.tight_layout()

    output = OUTDIR / "04_heat_capacity.png"
    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
