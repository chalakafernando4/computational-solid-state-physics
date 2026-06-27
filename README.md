# Computational Solid State Physics

Python simulations and visualizations for core solid state physics ideas: crystal
lattices, reciprocal space, diffraction selection rules, phonon dispersion,
tight-binding bands, semiconductor carrier estimates, heat capacity models, and
the 2D Ising model. The repository now includes scripts, notebooks, a Streamlit
app, and animation examples.

This project is inspired by standard undergraduate solid state physics topics,
including the style of material covered in Kittel. It does not include copied
book text, figures, or problem statements.

## Why this is a good GitHub project

- It connects physics theory to runnable numerical experiments.
- It produces clear plots for a portfolio or research notebook.
- It is modular enough to grow into a larger educational package.
- It demonstrates Python, NumPy, Matplotlib, testing, and scientific reasoning.

## Project structure

```text
computational-solid-state-physics/
  src/solidstate/
    lattices.py          # crystal lattice vectors and lattice utilities
    brillouin.py         # 2D Brillouin-zone construction
    reciprocal.py        # reciprocal lattice, d-spacing, XRD selection rules
    phonons.py           # monoatomic and diatomic chain dispersion
    bands.py             # tight-binding band models and paths
    semiconductors.py    # carrier concentration and conductivity helpers
    heat_capacity.py     # Debye and Einstein heat capacity models
    magnetism.py         # 2D Ising model Monte Carlo simulation
  examples/
    01_crystal_and_xrd.py
    02_phonon_dispersion.py
    03_tight_binding_bands.py
    04_heat_capacity.py
    05_ising_model.py
    06_brillouin_zones.py
    07_compare_heat_capacity_data.py
  notebooks/
    01_lattices_and_reciprocal_space.ipynb
    02_phonons.ipynb
    03_tight_binding.ipynb
    04_semiconductors.ipynb
    05_heat_capacity.ipynb
    06_ising_model.ipynb
    07_brillouin_zones.ipynb
  apps/
    streamlit_app.py
  animations/
    01_phonon_animation.py
    02_ising_animation.py
  data/
    synthetic_debye_heat_capacity.csv
  tests/
    test_solidstate.py
  outputs/
```

## Setup

From this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

If you prefer a simple install:

```bash
python -m pip install -r requirements.txt
```

## Run the examples

```bash
python examples/01_crystal_and_xrd.py
python examples/02_phonon_dispersion.py
python examples/03_tight_binding_bands.py
python examples/04_heat_capacity.py
python examples/05_ising_model.py
python examples/06_brillouin_zones.py
python examples/07_compare_heat_capacity_data.py
```

The scripts save figures into `outputs/`.

The Ising example uses quick Monte Carlo settings so it runs comfortably on a
laptop. Increase `size`, `sweeps`, and `burn_in` in the script when you want
smoother curves.

## Run the notebooks

Start Jupyter from this folder and open the notebooks:

```bash
jupyter notebook notebooks
```

The notebooks are generated from:

```bash
python tools/make_notebooks.py
```

## Run the Streamlit app

```bash
streamlit run apps/streamlit_app.py
```

The app includes interactive sliders for:

- Phonon dispersion
- Tight-binding bands
- The 2D Ising model

## Create animations

```bash
python animations/01_phonon_animation.py
python animations/02_ising_animation.py
```

The GIFs are saved into `outputs/`.

## Compare theory to data

The heat-capacity comparison script fits Debye and Einstein models to a CSV
with this schema:

```text
temperature_K,heat_capacity_J_mol_K
```

The included `data/synthetic_debye_heat_capacity.csv` is a teaching dataset,
not an experimental-data claim. Replace it with a real open dataset using the
same column names:

```bash
python examples/07_compare_heat_capacity_data.py --data data/your_dataset.csv
```

## Run tests

```bash
pytest
```

## Implemented roadmap

- Jupyter notebooks for the main modules.
- Streamlit sliders for phonons, band structure, and the Ising model.
- Additional lattices plus 2D Brillouin-zone construction.
- A reusable heat-capacity CSV fitting workflow.
- GIF animations for a phonon normal mode and Ising spin flips.

## Next possible upgrades

1. Add real open datasets with citations in `data/README.md`.
2. Add 3D Brillouin-zone construction for cubic lattices.
3. Add phonon animations for acoustic and optical diatomic modes.
4. Add a small documentation site using MkDocs.
5. Add GitHub Actions to run tests on every push.

## Notes on physics conventions

- Reciprocal lattice vectors use the `2*pi` convention.
- Energies in the tight-binding module are in arbitrary units unless stated.
- Semiconductor helper functions use SI units internally.
- Diffraction intensities are only schematic in the current examples. The
  selection rules are useful, but realistic intensities require form factors,
  Debye-Waller factors, and instrument broadening.
