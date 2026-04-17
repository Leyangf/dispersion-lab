# DispersionLab

Experiments on model-glass dispersion representations for optical design, centered on the **Buchdahl chromatic coordinate**. The notebooks test whether a compact physical parameterization `(nd, Vd, dPgF)` can predict spectral dispersion accurately enough to guide:

- glass substitution
- differentiable design
- tolerancing
- broadband refinement

---

## Contents

Read in numerical order. The project is organized in two parts:

- **Part I** (notebooks 1–3) — establishes the original Buchdahl workflow.
- **Part II** (notebooks 4–5) — audits and refines that workflow under stricter physical constraints.

### Part I — Original Buchdahl workflow

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | [01_model_glass_buchdahl.ipynb](notebooks/01_model_glass_buchdahl.ipynb) | Fits the 4-term Buchdahl model `n(λ) = nd + ν₁·ω + ν₂·ω² + ν₃·ω³ + ν₄·ω⁴` across catalog glasses. |
| 2 | [02_glass_substitution_workflow.ipynb](notebooks/02_glass_substitution_workflow.ipynb) | Validates the model as a glass-selection tool via an apochromatic doublet workflow. |
| 3 | [03_differentiable_design.ipynb](notebooks/03_differentiable_design.ipynb) | Ports the workflow to `torch` / autograd for gradient design and fixed-prescription tolerancing. |

### Part II — Audit and constrained refinement

| # | Notebook | Purpose |
|---|----------|---------|
| 4 | [04_neural_buchdahl_phase1.ipynb](notebooks/04_neural_buchdahl_phase1.ipynb) | Audits the original coefficient construction, repairs spectral-line anchor consistency, and tests neural coefficient predictors. |
| 5 | [05_neural_buchdahl_phase2.ipynb](notebooks/05_neural_buchdahl_phase2.ipynb) | Adds a bounded residual correction on top of the repaired anchor-preserving baseline to improve off-anchor broadband accuracy. |

> **Note:** Future runs should use the anchor-preserving baseline introduced in notebook 4. Notebooks 1–3 are retained as the original workflow baseline.

### Current refinement findings

- **Notebook 4** — the main repair is *physical*, not neural: re-solving `ν₁, ν₂` after predicting `ν₃, ν₄` preserves the d / F / C / g anchor definitions exactly. The MLP coefficient predictor adds little over the repaired linear baseline.
- **Notebook 5** — keeps those anchors fixed by construction with a residual envelope `q(λ)` that vanishes at d / F / C / g. It improves full-support broadband error off the anchors, while anchor-only downstream metrics (e.g. the apo secondary-spectrum check) remain unchanged.
- **Physical range** — for `eps_scale ≤ 1e-2`, the Phase 2 residual reduces H3 test RMS from ≈ `1.41e-3` to `9.18e-4`. Larger `eps_scale` rows are best read as stress tests, not small physical corrections.

---

## Repository layout

```
DispersionLab/
├── notebooks/      authored + generated .ipynb files (01–05)
├── scripts/        build_0N_*.py — regenerate notebooks 02–05 from source
├── data/           pre-trained regression matrices (produced by notebook 01)
├── pyproject.toml  uv-managed dependencies
└── uv.lock         locked versions
```

**Supporting files:**

- [data/regression_buchdahl_nu34_20dim_opt.npy](data/regression_buchdahl_nu34_20dim_opt.npy) — pretrained linear map for the 3rd / 4th Buchdahl coefficients (used by notebooks 02–04).
- [scripts/](scripts/) — `build_0N_*.py` regenerates notebooks 02–05. Notebook 01 is authored directly and produces the `.npy` regression files.

### Regenerating a notebook

Run the build script from the project root, then execute with `nbconvert`:

```bash
uv run python scripts/build_02_workflow.py
uv run jupyter nbconvert --to notebook --execute \
  notebooks/02_glass_substitution_workflow.ipynb \
  --output notebooks/02_glass_substitution_workflow.ipynb
```

> Notebooks use relative paths (`../data/...`), so they must run with `notebooks/` as the working directory — the default when launching Jupyter from that folder or when using `nbconvert --execute`.

---

## Setup

`uv` installs the locked environment from `pyproject.toml` and `uv.lock`:

```bash
uv sync

# Activate the virtual environment:
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
```

Then launch Jupyter or VSCode and select the project `.venv` kernel.

> The checked-in notebooks were run with the `uv` environment. Using a global Python or Conda kernel may change the bundled Optiland database and reproduce slightly different catalog counts.

---

## Requirements

Managed by `uv`:

- Python **3.11+**
- Jupyter, NumPy, SciPy, Matplotlib, pandas
- PyYAML
- Optiland
- PyTorch
- scikit-learn

Glass data is read from Optiland's bundled `database/data-nk/glass/`. Set `OPTILAND_DB_ROOT` only if the database lives elsewhere.

---

## Notes

- Wavelengths are in **micrometers**; optical distances in **millimeters**.
- Glass parameters are recomputed from Sellmeier coefficients for cross-catalog consistency.
- The differentiable notebooks use **CPU Torch** by default — no GPU required for current workloads.
