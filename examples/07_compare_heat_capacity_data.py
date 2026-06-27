from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from _paths import add_src_to_path

add_src_to_path()

from solidstate import debye_heat_capacity, einstein_heat_capacity


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)


def load_heat_capacity_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.asarray(data["temperature_K"], dtype=float), np.asarray(data["heat_capacity_J_mol_K"], dtype=float)


def debye_model(temperature: np.ndarray, theta_d: float, scale: float) -> np.ndarray:
    return scale * debye_heat_capacity(temperature, theta_d=theta_d)


def einstein_model(temperature: np.ndarray, theta_e: float, scale: float) -> np.ndarray:
    return scale * einstein_heat_capacity(temperature, theta_e=theta_e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Debye and Einstein models to heat-capacity CSV data.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic_debye_heat_capacity.csv")
    args = parser.parse_args()

    temperature, heat_capacity = load_heat_capacity_csv(args.data)
    debye_params, _ = curve_fit(debye_model, temperature, heat_capacity, p0=(250.0, 1.0), bounds=([1.0, 0.01], [2000.0, 10.0]))
    einstein_params, _ = curve_fit(einstein_model, temperature, heat_capacity, p0=(250.0, 1.0), bounds=([1.0, 0.01], [2000.0, 10.0]))

    grid = np.linspace(max(1.0, temperature.min()), temperature.max(), 400)
    debye_fit = debye_model(grid, *debye_params)
    einstein_fit = einstein_model(grid, *einstein_params)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(temperature, heat_capacity, label="CSV data", color="black", zorder=3)
    ax.plot(grid, debye_fit, label=f"Debye fit: theta_D={debye_params[0]:.1f} K", linewidth=2)
    ax.plot(grid, einstein_fit, label=f"Einstein fit: theta_E={einstein_params[0]:.1f} K", linewidth=2)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("heat capacity (J mol^-1 K^-1)")
    ax.set_title("Heat-capacity model comparison")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output = OUTDIR / "07_compare_heat_capacity_data.png"
    fig.savefig(output, dpi=180)
    print(f"saved {output}")
    print(f"Debye theta_D = {debye_params[0]:.2f} K, scale = {debye_params[1]:.3f}")
    print(f"Einstein theta_E = {einstein_params[0]:.2f} K, scale = {einstein_params[1]:.3f}")


if __name__ == "__main__":
    main()

