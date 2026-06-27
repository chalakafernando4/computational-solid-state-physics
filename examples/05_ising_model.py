from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _paths import add_src_to_path

add_src_to_path()

from solidstate import simulate_ising


OUTDIR = Path(__file__).resolve().parents[1] / "outputs"
OUTDIR.mkdir(exist_ok=True)


def main() -> None:
    temperatures = np.linspace(1.6, 3.4, 11)
    result = simulate_ising(size=16, temperatures=temperatures, sweeps=180, burn_in=60, seed=11)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(result["temperature"], result["abs_magnetization"], marker="o", color="tab:blue")
    axes[0].set_xlabel("temperature J/k_B")
    axes[0].set_ylabel("|M| per spin")
    axes[0].set_title("Magnetization")

    axes[1].plot(result["temperature"], result["heat_capacity"], marker="o", color="tab:red")
    axes[1].set_xlabel("temperature J/k_B")
    axes[1].set_ylabel("C")
    axes[1].set_title("Heat capacity")

    axes[2].plot(result["temperature"], result["susceptibility"], marker="o", color="tab:green")
    axes[2].set_xlabel("temperature J/k_B")
    axes[2].set_ylabel("susceptibility")
    axes[2].set_title("Susceptibility")

    fig.tight_layout()
    output = OUTDIR / "05_ising_model.png"
    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
