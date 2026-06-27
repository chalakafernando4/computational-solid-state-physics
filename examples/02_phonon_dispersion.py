from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from _paths import add_src_to_path

add_src_to_path()

from solidstate.phonons import diatomic_chain_dispersion, first_brillouin_zone, group_velocity, monoatomic_chain_dispersion


OUTDIR = Path(__file__).resolve().parents[1] / "outputs"
OUTDIR.mkdir(exist_ok=True)


def main() -> None:
    a = 1.0
    k = first_brillouin_zone(a=a, n_points=700)
    mono = monoatomic_chain_dispersion(k, a=a, spring_constant=1.0, mass=1.0)
    acoustic, optical = diatomic_chain_dispersion(k, a=a, spring_constant=1.0, mass_1=1.0, mass_2=2.5)
    velocity = group_velocity(k, acoustic)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

    axes[0].plot(k, mono, label="monoatomic", color="tab:blue")
    axes[0].plot(k, acoustic, label="diatomic acoustic", color="tab:green")
    axes[0].plot(k, optical, label="diatomic optical", color="tab:orange")
    axes[0].set_xlabel("wave vector k")
    axes[0].set_ylabel("angular frequency")
    axes[0].set_title("1D phonon dispersion")
    axes[0].legend(frameon=False)

    axes[1].plot(k, velocity, color="tab:purple")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("wave vector k")
    axes[1].set_ylabel("d omega / dk")
    axes[1].set_title("Acoustic branch group velocity")

    fig.tight_layout()
    output = OUTDIR / "02_phonon_dispersion.png"
    fig.savefig(output, dpi=180)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
