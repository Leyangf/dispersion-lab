# DispersionLab

Experiments on model-glass dispersion representations for optical design, centered on the Buchdahl chromatic coordinate. The notebooks test whether a compact physical parameterization `(nd, Vd, dPgF)` can predict spectral dispersion accurately enough to guide glass substitution, differentiable design, tolerancing, and broadband refinement.

## Contents

Read in this order. The project is organized in two parts: notebooks 1-3
establish the original Buchdahl workflow; notebooks 4-5 audit and refine
that workflow under stricter physical constraints.

### Part I - Original Buchdahl workflow

1. [model_glass_buchdahl.ipynb](model_glass_buchdahl.ipynb) - fits the 4-term Buchdahl model `n(lambda) = nd + nu1*w + nu2*w^2 + nu3*w^3 + nu4*w^4` across catalog glasses.
2. [glass_substitution_workflow.ipynb](glass_substitution_workflow.ipynb) - validates the model as a glass-selection tool using an apochromatic doublet workflow.
3. [differentiable_design.ipynb](differentiable_design.ipynb) - ports the workflow to torch/autograd for gradient design and fixed-prescription tolerancing.

### Part II - Audit and constrained refinement

4. [neural_buchdahl.ipynb](neural_buchdahl.ipynb) - audits the original coefficient construction, repairs spectral-line anchor consistency, and tests neural coefficient predictors.
5. [neural_buchdahl_phase2.ipynb](neural_buchdahl_phase2.ipynb) - adds a bounded residual correction on top of the repaired anchor-preserving baseline to improve off-anchor broadband accuracy.

Future workflow runs should use the anchor-preserving baseline introduced
in notebook 4; notebooks 1-3 are retained as the original workflow
baseline.

Current refinement findings:

- Notebook 4 shows that the main repair is physical, not neural: re-solving
  `nu1, nu2` after predicting `nu3, nu4` preserves the d/F/C/g anchor
  definitions exactly. The MLP coefficient predictor adds little over the
  repaired linear baseline.
- Notebook 5 keeps those anchors fixed by construction with a residual
  envelope `q(lambda)` that vanishes at d/F/C/g. It improves full-support
  broadband error off the anchors, while anchor-only downstream metrics
  such as the apo secondary-spectrum check remain unchanged.
- In the main physical range (`eps_scale <= 1e-2`), the Phase 2 residual
  reduces H3 test RMS from about `1.41e-3` to `9.18e-4`. Larger
  `eps_scale` rows are best read as stress tests, not as small physical
  corrections.

Supporting files:

- `regression_buchdahl_nu34_20dim_opt.npy` - pretrained linear map for the 3rd/4th Buchdahl coefficients.
- `_build_*.py` - scripts that regenerate the authored notebooks.

## Setup

`uv` installs the locked environment from `pyproject.toml` and `uv.lock`:

```bash
uv sync
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

Then launch Jupyter or VSCode and select the project `.venv` kernel. The checked notebooks were run with the uv environment; using a global Python or Conda kernel may change the bundled Optiland database and reproduce slightly different catalog counts.

## Requirements

Managed by `uv`: Python 3.11+, Jupyter, NumPy, SciPy, Matplotlib, PyYAML, Optiland, Torch, scikit-learn, and pandas.

Glass data is read from Optiland's bundled `database/data-nk/glass/`. Set `OPTILAND_DB_ROOT` only if you keep the database elsewhere.

## Notes

- Wavelengths are in micrometers; optical distances are in millimeters.
- Glass parameters are recomputed from Sellmeier coefficients for cross-catalog consistency.
- The differentiable notebooks use CPU Torch by default; no GPU is required for the current workloads.
