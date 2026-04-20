# DispersionLab

Experiments on model-glass dispersion representations for optical design, centered on the **Buchdahl chromatic coordinate**. The central question is whether a compact **3-parameter physical glass coordinate** `(nd, Vd, dPgF)` — the same three descriptors every lens designer already has for any catalog glass — can predict the full refractive index curve `n(λ)` accurately enough to guide:

- glass substitution
- differentiable design
- tolerancing
- broadband refinement

The 3-parameter identity is the load-bearing thesis claim: all model extensions (neural residuals, anchor-preserving construction, regression variants) are evaluated under the constraint that the input contract stays at `(nd, Vd, dPgF)`.

The forward model used across the downstream notebooks is **anchor-preserving by construction**: given `(nd, Vd, dPgF)`, the cross-glass regressor predicts `(ν₃, ν₄)`; the low-order `(ν₁, ν₂)` are back-solved from the F−C and g−F anchor differences so that

```
n(λ_d)              = nd
n(λ_F) - n(λ_C)     = (nd - 1) / Vd        (defines Vd)
n(λ_g) - n(λ_F)     = P_{g,F} · (nd - 1) / Vd  (defines P_{g,F})
```

hold at machine precision for any predicted `(ν₃, ν₄)`. The shared math lives in `src/dispersionlab/` and is covered by [tests/test_anchor_delta.py](tests/test_anchor_delta.py).

---

## Contents

Read in numerical order.

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | [01_model_glass_buchdahl.ipynb](notebooks/01_model_glass_buchdahl.ipynb) | Defines the anchor-preserving Buchdahl family for arbitrary K, runs the full (K × α) sweep inside it, and selects the production hyperparameters by principled criteria. Part 1 closes the legacy vs anchor-preserving construction choice with one decisive table (legacy V_d slip ≈ 1.14 units vs anchor ≈ 5·10⁻¹² on 544 glasses). Parts 3–5 derive **K = 4, α = 1.9547** from five principled α candidates, selecting the band-symmetric geometric derivation `α = (b−a)/(2ab)` as the only candidate both principled and inside the Test B max plateau. The legacy α = 1.818 is shown to be a retrospective coincidence, not a derivation. The full heritage notebook is archived at [01_legacy_reference.ipynb](notebooks/01_legacy_reference.ipynb). |
| 2 | [02_glass_substitution_workflow.ipynb](notebooks/02_glass_substitution_workflow.ipynb) | Validates the **anchor-preserving** model as a glass-selection tool via an apochromatic doublet workflow. Claim C now reports Spearman **+1.000** against Sellmeier ground truth with all 10 top-ranked CDGM crowns from the FK fluor-crown family (industry consensus); median `\|ΔS\|` against Sellmeier truth is ≈ 0.5 µm. Claim E flags that the nominal doublet is already out of the 15 µm secondary-spectrum spec (24.4 µm) — a diagnostic that the anchor-preserving repair unmasks. |
| 3 | [03_differentiable_design.ipynb](notebooks/03_differentiable_design.ipynb) | Ports the anchor-preserving workflow to `torch` / autograd for gradient design and fixed-prescription tolerancing. `d(BFL)/d(nd) = -19.9` matches the analytic biconvex thin-lens expectation. Analytical Jacobian + covariance propagation (J·Σ·Jᵀ) matches a 2000-sample Monte Carlo to **1.5 %** at ~1000× speed-up. |
| 4 | [04_neural_buchdahl_phase1.ipynb](notebooks/04_neural_buchdahl_phase1.ipynb) | Introduces the anchor-preserving construction (`ν₁, ν₂` back-solved after `ν₃, ν₄` are regressed) and runs a 4-variant ablation (A / C / D / Oracle) over linear vs neural coefficient predictors. On the cluster-holdout split the MLP matches the linear predictor — a clean negative result: at ~544 glasses with 3-d input, the 20-D linear map already saturates the cross-glass (nd, Vd, dPgF) → (ν₃, ν₄) target. |
| 5 | [05_neural_buchdahl_phase2.ipynb](notebooks/05_neural_buchdahl_phase2.ipynb) | Adds a bounded, anchor-preserving residual correction `ε·q(λ)·tanh(NN)` on top of the anchor-preserving baseline. The `q(λ)` envelope vanishes at d / F / C / g, so anchor-only downstream quantities are untouched by construction; off-anchor H3 test RMS drops by ~35 % at `ε_scale = 1e-2`. |

### Model selection — production vs accuracy frontier

The project distinguishes **production baseline** from **accuracy frontier**. Both live inside the anchor-preserving family; they differ only in hyperparameters.

| | K | α | p95 Test A hold err | Role |
|---|---|---|---|---|
| **Production** | 4 | **1.9547** | ~7.7·10⁻³ (Test B max) | Forward model used by NB02 / NB03 / NB04 / NB05; artifacts in `data/` |
| **Accuracy frontier** | 8 | 1.26 | ~4.6·10⁻⁴ | Diagnostic ablation only; artifacts in `data/sweep_tmp/` |

**Why K = 4, α = 1.9547 as production.** K = 4 is the unique Pareto-optimal K on (Test B max, cross-glass ratio) — other K values either tie on max with worse ratio or vice versa. α = 1.9547 is derived from the closed-form band-symmetric criterion `α = (b − a) / (2ab)` with `a = λ_d − λ_min`, `b = λ_max − λ_d` on the training grid — it makes the Buchdahl coordinate ω symmetric around zero on `[0.365, 2.3] μm`. Among five principled α candidates evaluated in NB01 Part 5 (band-symmetric, anchor-line symmetric, min cond(M), IR/UV asymptotic, Mercado-Robb Sellmeier-curvature), only α = 1.9547 sits inside the Test B max plateau while minimising Test B p95 — principled derivation + Pareto-acceptable data behaviour. The legacy α = 1.818 (from a sweep run under the old legacy construction, which had the anchor-slip bug) is retained only in `01_legacy_reference.ipynb` as heritage.

**Why K = 8, α = 1.26 is the frontier.** The anchor-preserving construction's implicit regularisation (ν₁, ν₂ are pinned by anchor equations, not trained on the 17-wavelength grid) pushes the wavelength-holdout overfitting threshold much higher than legacy constructions would. A full (K × α) sweep on 544 glasses under Test A leave-one-wavelength-out shows p95 absolute hold error monotonically decreases from K = 2 to K = 8 with no overfitting inflection, reaching ~4.6·10⁻⁴ at K = 8, α = 1.26 (Test B cross-glass CV ratios stay ≤ 2× at every percentile). This is an **8× accuracy improvement** over production — but on a metric defined against Sellmeier-derived ground truth (itself a 6-parameter functional fit), so the improvement is partly "matching Sellmeier's form more closely" rather than "capturing physical truth more accurately". It is flagged as a future NIR / broadband extension, not as a production change.

The sweep lives in [scripts/build_01_anchor_model.py](scripts/build_01_anchor_model.py) and writes only to `data/sweep_tmp/` so it can never be accidentally consumed by the production downstream.

### Current findings

- **Anchor preservation as a structural property.** The Level-1 target (per-glass `(ν₃, ν₄)` constrained LSQ) and the Level-2 predictor (20-D linear on `(nd, Vd, dPgF)`) are composed so that the output curve's implied `(nd, Vd, P_{g,F})` equal the inputs at floating-point precision. Verified in CI by [tests/test_anchor_delta.py](tests/test_anchor_delta.py): F1 = 0, F2_rel < 1e-13, F3 < 1e-13 across the full 544-glass catalog, and under arbitrary `(ν₃, ν₄)` — so the property is a consequence of the math, not of the LSQ target. The sweep-parametrised variant at arbitrary (K, α) is covered by [tests/test_sweep_anchor_delta.py](tests/test_sweep_anchor_delta.py).
- **Downstream lift from the repair.** In the NB02 apo-doublet workflow, Spearman between model `S` and full-Sellmeier truth is **+1.000** with median `\|ΔS\|` ≈ 0.5 µm — down from the O(10 µm) bias the legacy construction produced on the same test bench. Top-10 crown candidates are 10/10 FK fluor-crown glasses, matching the physics consensus for apochromats.
- **MLP adds nothing over the linear map.** NB04's ablation confirms that at the current catalog size the 20-D linear predictor saturates the cross-glass map. The residual floor is a property of the 4-term Buchdahl family under the anchor-preserving Level-1 target, not of regression capacity.
- **Anchor-preserving residual (Phase 2).** The `q(λ)·tanh(NN)` envelope keeps all anchor-only downstream metrics identical to Phase 1 by construction while reducing H3 RMS from ≈ 1.41·10⁻³ to 9.18·10⁻⁴ at `ε_scale ≤ 1e-2` (`H3_gap` ≤ 1.15, no overfitting; residual field curvature ≲ 4·10⁻⁵).

### Future directions

- **Stay in 3 parameters (recommended).** Alternative feature maps for the `(nd, Vd, dPgF) → (ν₃, ν₄)` predictor, cross-catalog augmentation / equivalent-glass grouping, and residual-model regularisation inside Phase 2 all preserve the input contract.
- **Accuracy-frontier extension (K = 8, α = 1.26).** Applications dominated by wavelength-grid fidelity (NIR / broadband / multi-spectral imaging) may justify the K = 8, α = 1.26 frontier — implemented as an opt-in alternate forward model (`solve_nu12_from_nu_high` in `dispersionlab.sweep`) rather than a production swap. See §8 of the report.
- **A fourth physical descriptor is conditional.** Adding a second partial-dispersion coordinate may help when the application *requires* sub-10⁻³ off-anchor error, but is an accuracy-for-complexity tradeoff.
- **Cross-catalog augmentation.** Merge Schott / Ohara / Sumita + CDGM with sub-cluster deduplication to flatten the training distribution, targeting the high-Vd / positive-ΔP_{g,F} hull corner that NB01's Test B tail and NB04's FK extrapolation both flag.

---

## Repository layout

```
DispersionLab/
├── notebooks/          authored + generated .ipynb files (01–05)
├── scripts/            build scripts for notebooks 02–05 and the shared dataset
├── src/dispersionlab/  shared math: feature_vec_20, solve_nu12_from_nu34,
│                       fit_anchor_preserving_coefficients, catalog loader
├── tests/              machine-precision anchor-preservation regression tests
├── data/               pretrained regression matrices + per-glass LSQ targets
├── pyproject.toml      uv-managed dependencies; declares src/ package
└── uv.lock             locked versions
```

**Shared package.** `src/dispersionlab/` is installed as an editable package by `uv sync`; every build script and notebook imports the same `buchdahl_omega`, `feature_vec_20`, `solve_nu12_from_nu34`, `fit_anchor_preserving_coefficients`, and `load_glass_catalog`. No duplicate copies across notebooks. A sweep-capable variant (`dispersionlab.sweep`) exposes the same primitives at arbitrary `(K, α)` for model-selection diagnostics.

**Supporting data files:**

- [data/regression_buchdahl_anchor_nu34_20dim_opt.npy](data/regression_buchdahl_anchor_nu34_20dim_opt.npy) — **production** 20-D linear map `(nd, Vd, dPgF) → (ν₃, ν₄)` at α = 1.9547, trained on the full 544-glass catalog under the anchor-preserving construction. Consumed by NB02 and NB03.
- [data/buchdahl_coeffs_per_glass.npz](data/buchdahl_coeffs_per_glass.npz) — per-glass LSQ targets for both constructions (`nu14_anchor`, `nu14_legacy`) plus catalog scalars, Sellmeier coefficients, and the 17-wavelength truth grid. Used by the anchor-preservation tests and by NB01 §4 / NB04 downstream.
- [data/regression_buchdahl_nu34_20dim_opt.npy](data/regression_buchdahl_nu34_20dim_opt.npy) — legacy 20-D map (pre-repair construction). Retained for the NB01 §4 contrast only.
- [data/regression_buchdahl_nu34_20dim.npy](data/regression_buchdahl_nu34_20dim.npy) — same legacy map trained at the earlier α = 2.5.
- [data/regression_buchdahl_nu34_10dim.npy](data/regression_buchdahl_nu34_10dim.npy) — 10-D feature-vector variant, used inside NB01 as an ablation.
- `data/sweep_tmp/` — accuracy-frontier sweep artifacts at non-production `(K, α)`. Produced by `scripts/build_01_anchor_model.py`; never consumed by downstream notebooks.
- [scripts/build_anchor_dataset.py](scripts/build_anchor_dataset.py) regenerates the production anchor-preserving data artifacts (K = 4, α = 1.9547).
- [scripts/build_01_anchor_model.py](scripts/build_01_anchor_model.py) runs the (K × α) diagnostic sweep inside the anchor-preserving family — writes to `data/sweep_tmp/`, never to `data/`.
- [scripts/build_01_model_glass_buchdahl.py](scripts/build_01_model_glass_buchdahl.py) regenerates the new anchor-preserving-first NB01.
- [scripts/](scripts/) `build_0N_*.py` regenerate notebooks 02–05.

### Regenerating

Rebuild the shared dataset (per-glass targets + cross-glass regressor):

```bash
uv run python scripts/build_anchor_dataset.py
```

Rebuild a downstream notebook and execute it:

```bash
uv run python scripts/build_02_workflow.py
uv run jupyter nbconvert --to notebook --execute \
  notebooks/02_glass_substitution_workflow.ipynb \
  --output notebooks/02_glass_substitution_workflow.ipynb
```

> Notebooks use relative paths (`../data/...`), so they must run with `notebooks/` as the working directory — the default when launching Jupyter from that folder or when using `nbconvert --execute`.

---

## Setup

`uv` installs the locked environment and the `dispersionlab` package from `pyproject.toml` + `uv.lock`:

```bash
uv sync

# Activate the virtual environment:
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
```

Then launch Jupyter or VSCode and select the project `.venv` kernel. `import dispersionlab` works directly in any notebook or script.

Run the anchor-preservation tests to verify the forward model before rerunning a notebook:

```bash
uv run python -m pytest tests/ -q
```

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
- pytest (dev group)

Glass data is read from Optiland's bundled `database/data-nk/glass/`. Set `OPTILAND_DB_ROOT` only if the database lives elsewhere.

---

## Notes

- Wavelengths are in **micrometers**; optical distances in **millimeters**.
- Glass parameters are recomputed from Sellmeier coefficients for cross-catalog consistency.
- The differentiable notebooks use **CPU Torch** by default — no GPU required for current workloads.
- The `dispersionlab.buchdahl` module's coupling matrices are frozen at `ALPHA = 1.9547` at import time; alpha is not exposed as a per-call parameter on the fit helpers. Mixing a different ω with those frozen matrices would silently violate anchors, so a future α sweep must build a matched `(omega, matrices)` pair via `dispersionlab.sweep`.
