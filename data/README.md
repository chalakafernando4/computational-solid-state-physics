# Data Folder

`synthetic_debye_heat_capacity.csv` is a small teaching dataset with the same
columns expected by the heat-capacity comparison script. It is not claimed as
experimental data.

To compare with an open experimental dataset, replace it with a CSV containing:

```text
temperature_K,heat_capacity_J_mol_K
```

Then run:

```bash
python examples/07_compare_heat_capacity_data.py --data data/your_dataset.csv
```

