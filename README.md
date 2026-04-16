# DispersionLab

Experiments on model-glass dispersion representations — comparing the **Buchdahl chromatic coordinate** against a Chebyshev intermediate representation.

## Contents

- [model_glass_buchdahl.ipynb](model_glass_buchdahl.ipynb) — Buchdahl model (`n(λ) = n_d + ν₁ω + ν₂ω² + ν₃ω³ + ν₄ω⁴`) fitted across Schott / CDGM / Ohara / etc. catalogs, with `n_d`, `V_d`, `ΔP_{g,F}` recomputed from Sellmeier coefficients using the Schott normal line for cross-catalog consistency.

## Setup

Create the conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate dispersionlab
```

Then launch Jupyter (`jupyter notebook`) or open the notebook in VSCode and select the `dispersionlab` kernel.

Update after `environment.yml` changes: `conda env update -f environment.yml --prune`.

### Requirements

- Python 3.11, Jupyter
- `numpy`, `scipy`, `matplotlib`, `pyyaml`
- [`optiland`](https://pypi.org/project/optiland/) (pip-only; provides the bundled glass database)

The notebook reads glass data from `optiland`'s `database/data-nk/glass/`. The path is auto-detected from the installed `optiland` package; set the `OPTILAND_DB_ROOT` environment variable to override if you keep the database in a non-standard location.
