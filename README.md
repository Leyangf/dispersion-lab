# DispersionLab

Experiments on model-glass dispersion representations — comparing the **Buchdahl chromatic coordinate** against a Chebyshev intermediate representation.

## Contents

- [model_glass_buchdahl.ipynb](model_glass_buchdahl.ipynb) — Buchdahl model (`n(λ) = n_d + ν₁ω + ν₂ω² + ν₃ω³ + ν₄ω⁴`) fitted across Schott / CDGM / Ohara / etc. catalogs, with `n_d`, `V_d`, `ΔP_{g,F}` recomputed from Sellmeier coefficients using the Schott normal line for cross-catalog consistency.

## Requirements

- Python 3, Jupyter
- `numpy`, `pyyaml`, [`optiland`](https://pypi.org/project/optiland/) (for the bundled glass database)

The notebook reads glass data from `optiland`'s `database/data-nk/glass/` — update `DB_ROOT` in the first code cell if your install path differs.
