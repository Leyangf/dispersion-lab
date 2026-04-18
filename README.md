# DispersionLab

Experiments on model-glass dispersion representations for optical design, centered on the **Buchdahl chromatic coordinate**. The central question is whether a compact **3-parameter physical glass coordinate** `(nd, Vd, dPgF)` — the same three descriptors every lens designer already has for any catalog glass — can predict the full refractive index curve `n(λ)` accurately enough to guide:

- glass substitution
- differentiable design
- tolerancing
- broadband refinement

The 3-parameter identity is the load-bearing thesis claim: all model extensions (neural residuals, anchor repair, regression variants) are evaluated under the constraint that the input contract stays at `(nd, Vd, dPgF)`.

---

## Contents

Read in numerical order. The project is organized in two parts:

- **Part I** (notebooks 1–3) — establishes the original Buchdahl workflow.
- **Part II** (notebooks 4–5) — audits and refines that workflow under stricter physical constraints.

### Part I — Original Buchdahl workflow

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | [01_model_glass_buchdahl.ipynb](notebooks/01_model_glass_buchdahl.ipynb) | Fits the 4-term Buchdahl model `n(λ) = nd + ν₁·ω + ν₂·ω² + ν₃·ω³ + ν₄·ω⁴` across catalog glasses. Establishes α = 1.818, order 4 as the sweet spot; Test A shows order 5+ overfits at the current 17-wavelength grid. 99 % of 543 glasses reach end-to-end error < 5·10⁻³. |
| 2 | [02_glass_substitution_workflow.ipynb](notebooks/02_glass_substitution_workflow.ipynb) | Validates the model as a glass-selection tool via an apochromatic doublet workflow. Five claims (A–E); Claim C finds all 10 top-ranked CDGM crowns are fluor-crown glasses (industry consensus), with Spearman 0.96 against Sellmeier ground truth. |
| 3 | [03_differentiable_design.ipynb](notebooks/03_differentiable_design.ipynb) | Ports the workflow to `torch` / autograd for gradient design and fixed-prescription tolerancing. Analytical Jacobian + covariance propagation matches 2 000-sample Monte Carlo to 1–2 % at ~1000× speed-up. |

### Part II — Audit and constrained refinement

| # | Notebook | Purpose |
|---|----------|---------|
| 4 | [04_neural_buchdahl_phase1.ipynb](notebooks/04_neural_buchdahl_phase1.ipynb) | Audits the original coefficient construction, repairs spectral-line anchor consistency, and runs a 4-variant ablation (A / C / D / Oracle) over linear vs neural coefficient predictors. The repair drops Claim G median ‖ΔS‖ from ~12 µm to ~1.5 µm; the MLP variant matches the linear one, giving a clean negative result. |
| 5 | [05_neural_buchdahl_phase2.ipynb](notebooks/05_neural_buchdahl_phase2.ipynb) | Adds a bounded, anchor-preserving residual correction `ε·q(λ)·tanh(NN)` on top of the repaired baseline. The `q(λ)` envelope vanishes at d / F / C / g, so anchor-only downstream quantities are untouched by construction; off-anchor H3 test RMS drops by ~35 % at `ε_scale = 1e-2`. |

> **Note:** Future runs should use the anchor-preserving baseline introduced in notebook 4. Notebooks 1–3 are retained as the pre-repair baseline that lets the repair's 8× improvement be quantified.

### Current refinement findings

- **Notebook 4** — the main repair is *physical*, not neural: re-solving `ν₁, ν₂` after predicting `ν₃, ν₄` preserves the d / F / C / g anchor definitions exactly (`G_anchor_delta ≈ 10⁻¹⁴`). The MLP coefficient predictor adds no measurable lift — at ~544 glasses with 3-d input, the 20-D linear polynomial already saturates the Level-2 map, so the 1.5 µm residual is the structural floor of the 4-term anchor-preserving Buchdahl family, not a regression error.
- **Notebook 5** — keeps those anchors fixed by construction with a residual envelope `q(λ)` that vanishes at d / F / C / g. It improves full-support broadband error off the anchors, while anchor-only downstream metrics (e.g. the apo secondary-spectrum check) remain unchanged.
- **Physical range** — for `eps_scale ≤ 1e-2`, the Phase 2 residual reduces H3 test RMS from ≈ `1.41e-3` to `9.18e-4` with `H3_gap` ≤ 1.15 (no overfitting) and residual field curvature ≲ `4·10⁻⁵` (smooth). Larger `eps_scale` rows are diagnostic stress tests, not small physical corrections.

### Future directions (notebooks 4 & 5)

The primary continuation is to keep the input fixed at `(nd, Vd, dPgF)` and improve the residual model around that constraint — this preserves the 3-parameter thesis claim. Secondary directions are explicitly labelled conditional or constrained:

- **Stay in 3 parameters (recommended).** Alternative feature maps for the `(nd, Vd, dPgF) → (ν₃, ν₄)` predictor, cross-catalog augmentation / equivalent-glass grouping, and residual-model regularisation inside Phase 2 all preserve the input contract.
- **A fourth physical descriptor is conditional.** Adding a second partial-dispersion coordinate may help when the application *requires* sub-`10⁻³` off-anchor error, but it is an accuracy-for-complexity tradeoff: the model is no longer a pure `(nd, Vd, dPgF)` glass representation, and catalog coverage must be re-validated.
- **Higher-order Buchdahl is a constrained experiment, not a drop-in extension.** Notebook 01's Test A shows order-5+ holdout ratios jumping to ~3.8× — a principled higher-order scheme requires a denser wavelength grid, explicit regularisation, or structural priors on `ν_{k≥3}`.
- **Sellmeier-aware residuals are a separate model class**, not the next step of the 3-parameter pipeline.

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

- [data/regression_buchdahl_nu34_20dim_opt.npy](data/regression_buchdahl_nu34_20dim_opt.npy) — pretrained linear map for the 3rd / 4th Buchdahl coefficients at α = 1.818 (the default used by notebooks 02–04).
- [data/regression_buchdahl_nu34_20dim.npy](data/regression_buchdahl_nu34_20dim.npy) — same map trained at the legacy α = 2.5 (kept for pre/post-optimization comparisons).
- [data/regression_buchdahl_nu34_10dim.npy](data/regression_buchdahl_nu34_10dim.npy) — a 10-D feature-vector variant used inside notebook 01 as an ablation.
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
