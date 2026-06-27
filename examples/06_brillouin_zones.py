from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _paths import add_src_to_path

add_src_to_path()

from solidstate import brillouin_zone_2d, graphene_reciprocal_basis, square_reciprocal_basis, triangular_reciprocal_basis


OUTDIR = Path(__file__).resolve().parents[1] / "outputs"
OUTDIR.mkdir(exist_ok=True)


def draw_zone(ax: plt.Axes, basis: np.ndarray, title: str) -> None:
    zone = brillouin_zone_2d(basis)
    closed = np.vstack([zone, zone[0]])
    points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            points.append(i * basis[0] + j * basis[1])
    points = np.asarray(points)
    ax.scatter(points[:, 0], points[:, 1], s=12, color="0.55")
    ax.plot(closed[:, 0], closed[:, 1], color="tab:red", linewidth=2)
    ax.fill(zone[:, 0], zone[:, 1], color="tab:red", alpha=0.14)
    ax.scatter([0.0], [0.0], color="black", s=30)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    draw_zone(axes[0], square_reciprocal_basis(a=1.0), "Square lattice")
    draw_zone(axes[1], triangular_reciprocal_basis(a=1.0), "Triangular lattice")
    draw_zone(axes[2], graphene_reciprocal_basis(a=1.0), "Graphene reciprocal lattice")
    fig.tight_layout()

    output = OUTDIR / "06_brillouin_zones.png"
    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()

