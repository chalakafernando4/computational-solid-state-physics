from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from solidstate.magnetism import initialize_spins, metropolis_sweep


def main() -> None:
    rng = np.random.default_rng(5)
    spins = initialize_spins(size=48, seed=5)
    temperature = 2.25

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    image = ax.imshow(spins, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("2D Ising spin flips near Tc")

    def update(_: int):
        for _ in range(4):
            metropolis_sweep(spins, temperature=temperature, rng=rng)
        image.set_data(spins)
        return (image,)

    ani = animation.FuncAnimation(fig, update, frames=80, interval=55, blit=True)
    output = OUTDIR / "ising_spin_flips.gif"
    ani.save(output, writer="pillow", fps=18)
    print(f"saved {output}")


if __name__ == "__main__":
    main()

