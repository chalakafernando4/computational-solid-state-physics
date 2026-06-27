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

from solidstate.phonons import monoatomic_chain_dispersion


def main() -> None:
    n_atoms = 36
    a = 1.0
    k = np.pi / 3.0
    amplitude = 0.18
    positions = np.arange(n_atoms) * a
    omega = monoatomic_chain_dispersion(np.array([k]), a=a)[0]

    fig, ax = plt.subplots(figsize=(8, 2.6))
    scatter = ax.scatter(positions, np.zeros_like(positions), s=80, color="tab:blue")
    line, = ax.plot(positions, np.zeros_like(positions), color="tab:blue", alpha=0.35)
    ax.set_xlim(-1, n_atoms)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("atom index")
    ax.set_title("Longitudinal phonon normal mode")

    def update(frame: int):
        time = frame / 18.0
        displacement = amplitude * np.cos(k * positions - omega * time)
        xy = np.column_stack([positions + displacement, np.zeros_like(positions)])
        scatter.set_offsets(xy)
        line.set_data(positions + displacement, np.zeros_like(positions))
        return scatter, line

    ani = animation.FuncAnimation(fig, update, frames=90, interval=40, blit=True)
    output = OUTDIR / "phonon_mode.gif"
    ani.save(output, writer="pillow", fps=24)
    print(f"saved {output}")


if __name__ == "__main__":
    main()

