from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from solidstate import simulate_ising, square_lattice_dispersion, tight_binding_1d
from solidstate.phonons import diatomic_chain_dispersion, first_brillouin_zone, monoatomic_chain_dispersion


st.set_page_config(page_title="Computational Solid State Physics", layout="wide")
st.title("Computational Solid State Physics")

page = st.sidebar.selectbox(
    "Demo",
    [
        "Phonon dispersion",
        "Tight-binding bands",
        "Ising model",
    ],
)


def show_phonons() -> None:
    st.header("Phonon Dispersion")
    a = st.sidebar.slider("Lattice spacing a", 0.5, 3.0, 1.0, 0.1)
    spring = st.sidebar.slider("Spring constant K", 0.2, 5.0, 1.0, 0.1)
    mass_1 = st.sidebar.slider("Mass 1", 0.2, 5.0, 1.0, 0.1)
    mass_2 = st.sidebar.slider("Mass 2", 0.2, 8.0, 2.0, 0.1)

    k = first_brillouin_zone(a=a, n_points=600)
    mono = monoatomic_chain_dispersion(k, a=a, spring_constant=spring, mass=mass_1)
    acoustic, optical = diatomic_chain_dispersion(k, a=a, spring_constant=spring, mass_1=mass_1, mass_2=mass_2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k, mono, label="monoatomic", linewidth=2)
    ax.plot(k, acoustic, label="diatomic acoustic", linewidth=2)
    ax.plot(k, optical, label="diatomic optical", linewidth=2)
    ax.set_xlabel("k")
    ax.set_ylabel("angular frequency")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    st.pyplot(fig)


def show_bands() -> None:
    st.header("Tight-Binding Bands")
    hopping = st.sidebar.slider("Hopping t", 0.1, 4.0, 1.0, 0.1)
    onsite = st.sidebar.slider("Onsite energy", -3.0, 3.0, 0.0, 0.1)
    grid_points = st.sidebar.slider("Grid points", 50, 350, 180, 10)

    k = np.linspace(-np.pi, np.pi, 600)
    energy_1d = tight_binding_1d(k, hopping=hopping, onsite=onsite)

    grid = np.linspace(-np.pi, np.pi, grid_points)
    kx, ky = np.meshgrid(grid, grid)
    energy_2d = square_lattice_dispersion(kx, ky, hopping=hopping, onsite=onsite)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(k, energy_1d, linewidth=2)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("energy")
    axes[0].set_title("1D chain")
    contour = axes[1].contourf(kx, ky, energy_2d, levels=45, cmap="viridis")
    axes[1].set_xlabel("kx")
    axes[1].set_ylabel("ky")
    axes[1].set_title("2D square lattice")
    fig.colorbar(contour, ax=axes[1], label="energy")
    fig.tight_layout()
    st.pyplot(fig)


@st.cache_data(show_spinner=False)
def cached_ising(size: int, t_min: float, t_max: float, n_t: int, sweeps: int, burn_in: int) -> dict[str, np.ndarray]:
    temperatures = np.linspace(t_min, t_max, n_t)
    return simulate_ising(size=size, temperatures=temperatures, sweeps=sweeps, burn_in=burn_in, seed=13)


def show_ising() -> None:
    st.header("2D Ising Model")
    size = st.sidebar.slider("Lattice size", 8, 32, 16, 2)
    sweeps = st.sidebar.slider("Sweeps", 80, 800, 180, 20)
    burn_in = st.sidebar.slider("Burn-in", 20, 300, 60, 20)
    n_t = st.sidebar.slider("Temperature points", 7, 31, 11, 2)

    result = cached_ising(size, 1.6, 3.4, n_t, sweeps, burn_in)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(result["temperature"], result["abs_magnetization"], marker="o")
    axes[0].set_title("Magnetization")
    axes[0].set_xlabel("T")
    axes[0].set_ylabel("|M|")
    axes[1].plot(result["temperature"], result["heat_capacity"], marker="o", color="tab:red")
    axes[1].set_title("Heat capacity")
    axes[1].set_xlabel("T")
    axes[2].plot(result["temperature"], result["susceptibility"], marker="o", color="tab:green")
    axes[2].set_title("Susceptibility")
    axes[2].set_xlabel("T")
    fig.tight_layout()
    st.pyplot(fig)


if page == "Phonon dispersion":
    show_phonons()
elif page == "Tight-binding bands":
    show_bands()
else:
    show_ising()

