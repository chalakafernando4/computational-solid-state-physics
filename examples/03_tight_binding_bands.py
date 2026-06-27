from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _paths import add_src_to_path

add_src_to_path()

from solidstate import dos_histogram, high_symmetry_path, square_lattice_dispersion, tight_binding_1d


OUTDIR = Path(__file__).resolve().parents[1] / "outputs"
OUTDIR.mkdir(exist_ok=True)


def main() -> None:
    k_1d = np.linspace(-np.pi, np.pi, 600)
    energy_1d = tight_binding_1d(k_1d, hopping=1.0)

    grid = np.linspace(-np.pi, np.pi, 250)
    kx, ky = np.meshgrid(grid, grid)
    energy_2d = square_lattice_dispersion(kx, ky, hopping=1.0)
    dos_x, dos_y = dos_histogram(energy_2d, bins=120)

    points = {
        "G": (0.0, 0.0),
        "X": (np.pi, 0.0),
        "M": (np.pi, np.pi),
    }
    path, distance, ticks = high_symmetry_path(points, ["G", "X", "M", "G"], points_per_segment=120)
    path_energy = square_lattice_dispersion(path[:, 0], path[:, 1], hopping=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(k_1d, energy_1d, color="tab:blue")
    axes[0].set_title("1D nearest-neighbor band")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("energy / t")

    contour = axes[1].contourf(kx, ky, energy_2d, levels=40, cmap="viridis")
    fig.colorbar(contour, ax=axes[1], label="energy / t")
    axes[1].set_title("Square-lattice band")
    axes[1].set_xlabel("kx")
    axes[1].set_ylabel("ky")

    axes[2].plot(distance, path_energy, color="tab:red", label="band")
    twin = axes[2].twiny()
    twin.plot(dos_y / dos_y.max() * distance.max(), dos_x, color="0.55", alpha=0.8, label="DOS")
    twin.set_xticks([])
    axes[2].set_title("G-X-M-G path")
    axes[2].set_xlabel("path distance")
    axes[2].set_ylabel("energy / t")
    axes[2].set_xticks(ticks)
    axes[2].set_xticklabels(["G", "X", "M", "G"])

    fig.tight_layout()
    output = OUTDIR / "03_tight_binding_bands.png"
    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
