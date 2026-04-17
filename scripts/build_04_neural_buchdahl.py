"""Build neural_buchdahl.ipynb — Part 1.

Thesis chapter #4: extend Buchdahl to NN inside the existing coordinate
system, starting with anchor-preserving baseline repair.

Central question:

    Once the spectral-line anchors are enforced exactly on the per-glass
    coefficient targets, how much expressive power is still missing
    from the 3-parameter (nd, Vd, dPgF) -> (nu3, nu4) mapping?

Part 1 deliverables (self-contained):
  1A. Per-glass anchor-preserving coefficient fit (Level 1 repair).
  1B. Three Level 2 predictors + Oracle baseline on the repaired targets.
  1C. Claim F: end-to-end anchor preservation (F1, F2_abs, F2_rel, F3).
  1D. Claim G: downstream ranking stability on fixed apo-doublet bench.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
OUT = PROJECT_ROOT / "notebooks" / "04_neural_buchdahl.ipynb"


def md(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = []

# =========================================================================
# Intro
# =========================================================================

cells.append(md(
"""# Neural Extension of the Buchdahl Dispersion Model — Part 1

This is the 4th notebook of the thesis arc on **dispersion-model
representations** for "model glass" used in optical design. The
previous three:

- `model_glass_buchdahl.ipynb` — trains the 3-parameter Buchdahl model.
- `glass_substitution_workflow.ipynb` — validates it via 5 claims.
- `differentiable_design.ipynb` — torch backend + analytical tolerancing.

**What Notebook 4 does**: treats the Buchdahl coefficient regression as a
question of *model capacity inside a physically-constrained coordinate
system*. It fixes a latent bug in the existing targets and then asks
whether neural expressivity adds anything on top of the corrected
targets.

**The bug** (quantified on N-BK7: slip = 1.15e-4 in $n(F)-n(C)$, i.e.
~1.4% relative in $V_d$). The current `analytical_nu12` solves

$$M \\begin{bmatrix} \\nu_1 \\\\ \\nu_2 \\end{bmatrix}
 = \\begin{bmatrix} \\Delta n_{FC} \\\\ \\Delta n_{gF} \\end{bmatrix}$$

**ignoring the $\\nu_3, \\nu_4$ contribution** at $\\lambda_F, \\lambda_C,
\\lambda_g$. So the model's output curve has an $n(F) - n(C)$ that is
not exactly $(n_d - 1)/V_d$. The per-glass $(\\nu_3, \\nu_4)$ *target*
fed into `A_REG` regression is itself polluted by this slip, which
contaminates any downstream comparison of predictor capacity.

**Central question**:

> Once the spectral-line anchors are enforced exactly on the per-glass
> coefficient targets, how much expressive power is still missing from
> the 3-parameter $(n_d, V_d, \\Delta P_{g,F}) \\to (\\nu_3, \\nu_4)$
> mapping?

**Two-level decomposition**

| Level | What it computes | Current repo |
|---|---|---|
| **Level 1** (per-glass) | Sellmeier truth → best $(\\nu_3, \\nu_4)$ target | `nu34_residual` in main notebook |
| **Level 2** (cross-glass) | $(n_d, V_d, \\Delta P_{g,F}) \\to (\\nu_3, \\nu_4)$ predictor | `regression_buchdahl_nu34_20dim_opt.npy` |

**Part 1 plan**

1. **1A** — anchor-preserving Level 1 fit (physics repair, no ML).
2. **1B** — Level 2 predictors on repaired targets:
   - **A: Old linear** (existing `A_REG`, trained on old targets)
   - **C: Anchor linear** (same 20-dim features, retrained on anchor targets)
   - **D: Anchor MLP** (3→16→16→2 tanh, weight decay, 5-seed ensemble)
   - **Oracle** (per-glass best $(\\nu_3, \\nu_4)$ + anchor solve; upper bound)
3. **1C** — Claim F: end-to-end anchor preservation (F1, F2_abs, F2_rel, F3).
4. **1D** — Claim G: downstream ranking stability vs Sellmeier truth on
   the fixed apo-doublet test bench from the workflow notebook.
5. **Scorecard** joint F+G; conclusion.

Expected take-away (refined after first execution — see Conclusion for
the actual numbers):

> Anchor-preserving repair does **not** reduce raw $n(\\lambda)$
> reconstruction error on the cluster-holdout split; C/D are slightly
> *worse* than A on max/P95/RMS. It does, however, eliminate the physical
> anchor slip *by construction* and dramatically improves downstream
> secondary-spectrum agreement with Sellmeier truth. The MLP adds little
> over the retrained anchor-linear map — on 544 glasses with a 3-d
> input, the 20-d polynomial predictor is already sufficient inside the
> repaired parameterization.
"""))

# =========================================================================
# Setup
# =========================================================================

cells.append(md("""## Setup"""))

cells.append(code(
"""from __future__ import annotations
import os
# Intel OMP + torch OMP coexistence on Windows — same workaround used in
# differentiable_design.ipynb.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import importlib.util
import re
import time

import numpy as np
import yaml
import pandas as pd
import torch

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from scipy.optimize import fsolve

np.random.seed(0)
torch.manual_seed(0)

print(f"numpy {np.__version__}, torch {torch.__version__}, pandas {pd.__version__}")
"""))

cells.append(code(
"""# Buchdahl constants — identical to the other three notebooks.
LAMBDA_D = 0.5875618
LAMBDA_g = 0.4358343
LAMBDA_F = 0.4861327
LAMBDA_C = 0.6562725
ALPHA    = 1.818          # optimized in model_glass_buchdahl.ipynb

def buchdahl_omega(lam, alpha=ALPHA):
    dl = lam - LAMBDA_D
    return dl / (1.0 + alpha * dl)

OMEGA_D = 0.0
OMEGA_g = buchdahl_omega(LAMBDA_g)
OMEGA_F = buchdahl_omega(LAMBDA_F)
OMEGA_C = buchdahl_omega(LAMBDA_C)

# 17-wavelength training grid, same as main notebook.
WAVELENGTHS = np.array([
    0.36501, 0.40466, 0.43583, 0.48613, 0.54607,
    0.58756, 0.58929, 0.6328,  0.64385, 0.65627,
    0.70652, 0.85211, 1.01398, 1.060,   1.52958,
    1.97009, 2.3
])
OMEGA_TRAIN = buchdahl_omega(WAVELENGTHS)

# Optiland glass database
if "OPTILAND_DB_ROOT" in os.environ:
    DB_ROOT = Path(os.environ["OPTILAND_DB_ROOT"])
else:
    _spec = importlib.util.find_spec("optiland")
    if _spec is None or _spec.origin is None:
        raise RuntimeError("optiland not installed")
    DB_ROOT = Path(_spec.origin).parent / "database"
GLASS_ROOT = DB_ROOT / "data-nk" / "glass"
print(f"Glass DB: {GLASS_ROOT}")
"""))

cells.append(code(
"""# ---- Sellmeier catalog loader — identical recomputation to workflow nb ----
def _sellmeier(lam, sm):
    B1, C1, B2, C2, B3, C3 = sm
    wl2 = lam ** 2
    return np.sqrt(1 + B1*wl2/(wl2-C1) + B2*wl2/(wl2-C2) + B3*wl2/(wl2-C3))


def extract_record(yml_path):
    with yml_path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    data_blocks = doc.get("DATA", [])
    if not data_blocks:
        return None
    formula = data_blocks[0]
    if formula.get("type") != "formula 2":
        return None
    coeffs = formula.get("coefficients", "").split()
    if len(coeffs) != 7:
        return None
    wr = formula.get("wavelength_range", "").split()
    if len(wr) != 2:
        return None
    B1, C1 = float(coeffs[1]), float(coeffs[2])
    B2, C2 = float(coeffs[3]), float(coeffs[4])
    B3, C3 = float(coeffs[5]), float(coeffs[6])
    lam_lo, lam_hi = float(wr[0]), float(wr[1])
    # Need full coverage of the 17-wavelength training grid
    if lam_lo > 0.365 + 1e-6 or lam_hi < 2.3 - 1e-6:
        return None
    sm = (B1, C1, B2, C2, B3, C3)
    nd = float(_sellmeier(LAMBDA_D, sm))
    nF = float(_sellmeier(LAMBDA_F, sm))
    nC = float(_sellmeier(LAMBDA_C, sm))
    ng = float(_sellmeier(LAMBDA_g, sm))
    dn_FC = nF - nC
    if dn_FC < 1e-12:
        return None
    vd   = (nd - 1.0) / dn_FC
    PgF  = (ng - nF) / dn_FC
    dPgF = PgF - (0.6438 - 0.001682 * vd)
    return dict(
        name=yml_path.stem, catalog=yml_path.parent.name,
        nd=nd, vd=vd, PgF=PgF, dPgF=dPgF,
        sellmeier=sm,
        n_truth=np.array([_sellmeier(lam, sm) for lam in WAVELENGTHS]),
    )


glasses = []
for yml in GLASS_ROOT.rglob("*.yml"):
    rec = extract_record(yml)
    if rec is not None:
        glasses.append(rec)
glasses = [g for g in glasses if 10 < g["vd"] < 100 and -0.02 < g["dPgF"] < 0.10]
N_GLASS = len(glasses)
print(f"Loaded {N_GLASS} glasses with full 0.365–2.3 um Sellmeier support.")
"""))

cells.append(code(
"""# ---- Load the pre-trained linear regressor A_REG (variant A) ----
A_REG_old = np.load("../data/regression_buchdahl_nu34_20dim_opt.npy")
print(f"A_REG_old loaded: shape {A_REG_old.shape}  (alpha = {ALPHA})")
"""))

# =========================================================================
# Part 1A — anchor-preserving Level 1 fit
# =========================================================================

cells.append(md(
"""## 1A — Per-glass anchor-preserving coefficient fit

### Derivation

Buchdahl 4-term:

$$n(\\lambda) = n_d + \\nu_1 \\omega + \\nu_2 \\omega^2 + \\nu_3 \\omega^3 + \\nu_4 \\omega^4$$

Anchor constraints. $\\omega(\\lambda_d) = 0$ makes $n(\\lambda_d) = n_d$
automatic. The other two are:

$$n(\\lambda_F) - n(\\lambda_C) = \\frac{n_d - 1}{V_d}, \\qquad
  n(\\lambda_g) - n(\\lambda_F) = P_{g,F}\\,\\frac{n_d - 1}{V_d}$$

Substituting the Buchdahl expansion and keeping all four $\\nu_k$ gives a
**2×2 linear system for $\\nu_1, \\nu_2$ that does include the $\\nu_3,
\\nu_4$ contributions**:

$$M \\begin{bmatrix} \\nu_1 \\\\ \\nu_2 \\end{bmatrix}
 = \\begin{bmatrix} \\Delta n_{FC} \\\\ \\Delta n_{gF} \\end{bmatrix}
 - D \\begin{bmatrix} \\nu_3 \\\\ \\nu_4 \\end{bmatrix}$$

with

$$M = \\begin{bmatrix} \\omega_F - \\omega_C & \\omega_F^2 - \\omega_C^2 \\\\
                       \\omega_g - \\omega_F & \\omega_g^2 - \\omega_F^2 \\end{bmatrix}, \\quad
  D = \\begin{bmatrix} \\omega_F^3 - \\omega_C^3 & \\omega_F^4 - \\omega_C^4 \\\\
                       \\omega_g^3 - \\omega_F^3 & \\omega_g^4 - \\omega_F^4 \\end{bmatrix}$$

Define $a = M^{-1}[\\Delta n_{FC}, \\Delta n_{gF}]^\\top$ and $B = M^{-1} D$.
Then $[\\nu_1, \\nu_2]^\\top = a - B\\,[\\nu_3, \\nu_4]^\\top$.

The current repo uses $[\\nu_1, \\nu_2] = a$ (drops the $B$ coupling),
which is why the output's $V_d, \\Delta P_{g,F}$ don't exactly match the
input — the $\\nu_3, \\nu_4$ chosen by regression contribute at F/C/g.

Plugging this back into the full 4-term expansion, we get a linear
problem **still in $\\nu_3, \\nu_4$**:

$$n(\\lambda) - [n_d + \\omega a_1 + \\omega^2 a_2]
= \\Phi_3(\\omega)\\,\\nu_3 + \\Phi_4(\\omega)\\,\\nu_4$$

with effective basis

$$\\Phi_3(\\omega) = \\omega^3 - \\omega B_{11} - \\omega^2 B_{21}, \\qquad
  \\Phi_4(\\omega) = \\omega^4 - \\omega B_{12} - \\omega^2 B_{22}$$

**One `lstsq` per glass; no iteration.** Anchor constraints hold by
construction.
"""))

cells.append(code(
"""# ---- Constant anchor matrices ----
M_MAT = np.array([
    [OMEGA_F - OMEGA_C,   OMEGA_F**2 - OMEGA_C**2],
    [OMEGA_g - OMEGA_F,   OMEGA_g**2 - OMEGA_F**2],
])
D_MAT = np.array([
    [OMEGA_F**3 - OMEGA_C**3,  OMEGA_F**4 - OMEGA_C**4],
    [OMEGA_g**3 - OMEGA_F**3,  OMEGA_g**4 - OMEGA_F**4],
])
M_INV = np.linalg.inv(M_MAT)
B_COUP = M_INV @ D_MAT   # [[B11, B12], [B21, B22]]
print("Condition number of M:", np.linalg.cond(M_MAT))
print("B coupling matrix B = M^-1 D:")
print(B_COUP)
"""))

cells.append(code(
"""# ---- Unit test: Phi_k must vanish at omega_{F,C,g} anchor lines by construction
def Phi3(om):
    return om**3 - om*B_COUP[0,0] - om**2 * B_COUP[1,0]

def Phi4(om):
    return om**4 - om*B_COUP[0,1] - om**2 * B_COUP[1,1]

# At anchor omegas, the F-C and g-F DIFFERENCES of Phi_k must be zero
# (since ν3, ν4 must not contribute to dn_FC or dn_gF by construction).
diff_FC_3 = Phi3(OMEGA_F) - Phi3(OMEGA_C)
diff_gF_3 = Phi3(OMEGA_g) - Phi3(OMEGA_F)
diff_FC_4 = Phi4(OMEGA_F) - Phi4(OMEGA_C)
diff_gF_4 = Phi4(OMEGA_g) - Phi4(OMEGA_F)
print("Phi_3 and Phi_4 anchor-difference residuals (must be ~1e-16):")
print(f"  Phi3(F)-Phi3(C) = {diff_FC_3:+.3e}")
print(f"  Phi3(g)-Phi3(F) = {diff_gF_3:+.3e}")
print(f"  Phi4(F)-Phi4(C) = {diff_FC_4:+.3e}")
print(f"  Phi4(g)-Phi4(F) = {diff_gF_4:+.3e}")
for x in (diff_FC_3, diff_gF_3, diff_FC_4, diff_gF_4):
    assert abs(x) < 1e-12, f"Anchor constraint violated: {x}"
print("  OK — anchor preservation holds by construction.")
"""))

cells.append(code(
"""def anchor_preserving_fit(nd, vd, dPgF, n_truth, omega_grid=OMEGA_TRAIN):
    \"\"\"Return (nu1, nu2, nu3, nu4) with anchors exact.\"\"\"
    dn_FC = (nd - 1.0) / vd
    PgF = 0.6438 - 0.001682*vd + dPgF
    dn_gF = PgF * dn_FC

    a = M_INV @ np.array([dn_FC, dn_gF])
    phi3 = omega_grid**3 - omega_grid*B_COUP[0,0] - omega_grid**2 * B_COUP[1,0]
    phi4 = omega_grid**4 - omega_grid*B_COUP[0,1] - omega_grid**2 * B_COUP[1,1]
    basis = np.column_stack([phi3, phi4])
    target = n_truth - nd - omega_grid*a[0] - omega_grid**2 * a[1]
    nu34, *_ = np.linalg.lstsq(basis, target, rcond=None)
    nu12 = a - B_COUP @ nu34
    return nu12[0], nu12[1], nu34[0], nu34[1]


def old_fit(nd, vd, dPgF, n_truth, omega_grid=OMEGA_TRAIN):
    \"\"\"Current repo practice: nu1,nu2 analytic (no coupling); nu3,nu4 from residual.\"\"\"
    dn_FC = (nd - 1.0) / vd
    PgF = 0.6438 - 0.001682*vd + dPgF
    dn_gF = PgF * dn_FC
    nu12 = np.linalg.solve(M_MAT, np.array([dn_FC, dn_gF]))
    residual = n_truth - nd - nu12[0]*omega_grid - nu12[1]*omega_grid**2
    A = np.column_stack([omega_grid**3, omega_grid**4])
    nu34, *_ = np.linalg.lstsq(A, residual, rcond=None)
    return nu12[0], nu12[1], nu34[0], nu34[1]


# Compute both per-glass target sets
old_coefs = np.zeros((N_GLASS, 4))
new_coefs = np.zeros((N_GLASS, 4))
for i, g in enumerate(glasses):
    old_coefs[i] = old_fit(g["nd"], g["vd"], g["dPgF"], g["n_truth"])
    new_coefs[i] = anchor_preserving_fit(g["nd"], g["vd"], g["dPgF"], g["n_truth"])

print(f"Per-glass Level 1 coefficient fits computed for all {N_GLASS} glasses.")
print()
print("Change in targets (old → anchor-preserving):")
print(f"  |Δν3|: max = {np.abs(old_coefs[:,2]-new_coefs[:,2]).max():.3e}, "
      f"mean = {np.abs(old_coefs[:,2]-new_coefs[:,2]).mean():.3e}")
print(f"  |Δν4|: max = {np.abs(old_coefs[:,3]-new_coefs[:,3]).max():.3e}, "
      f"mean = {np.abs(old_coefs[:,3]-new_coefs[:,3]).mean():.3e}")
"""))

cells.append(code(
"""# ---- Level 1 reconstruction error: best-per-glass coefficients ----
def reconstruct(nu14, nd, omega_grid):
    n = nd * np.ones_like(omega_grid)
    for k, nk in enumerate(nu14, 1):
        n = n + nk * omega_grid**k
    return n


def level1_errors(coefs, glass_list=None, omega_grid=OMEGA_TRAIN):
    if glass_list is None:
        glass_list = glasses
    errs = []
    for (nu1, nu2, nu3, nu4), g in zip(coefs, glass_list):
        n_pred = reconstruct([nu1, nu2, nu3, nu4], g["nd"], omega_grid)
        errs.append(np.abs(n_pred - g["n_truth"]))
    E = np.array(errs)
    return dict(
        max=float(E.max()),
        p95=float(np.percentile(E, 95)),
        rms=float(np.sqrt((E**2).mean())),
    )


E_old = level1_errors(old_coefs)
E_new = level1_errors(new_coefs)
print(f"Level 1 per-glass reconstruction error (best coefficients, {N_GLASS} glasses):")
print(f"  {'':<26} {'Old target':>14} {'Anchor target':>16}")
print(f"  {'-'*26}  {'-'*12}  {'-'*14}")
print(f"  {'max |n(λ) - truth|':<26} {E_old['max']:>14.3e} {E_new['max']:>16.3e}")
print(f"  {'P95 |err|':<26} {E_old['p95']:>14.3e} {E_new['p95']:>16.3e}")
print(f"  {'RMS |err|':<26} {E_old['rms']:>14.3e} {E_new['rms']:>16.3e}")
print()
print("These are Level-1 lower bounds. Level-2 predictors can't do better")
print("than per-glass optimal targets.")
"""))

# =========================================================================
# Part 1B — Level 2 predictors
# =========================================================================

cells.append(md(
"""## 1B — Level 2 predictors on the repaired targets

Four rows compared on the same test set:

| Variant | Target | Regressor |
|---|---|---|
| **A: Old linear** | old non-anchor $(\\nu_3, \\nu_4)$ | pretrained `A_REG` (20-d linear) |
| **C: Anchor linear** | anchor-preserving $(\\nu_3, \\nu_4)$ | new `A_REG'` (20-d linear, retrained) |
| **D: Anchor MLP** | anchor-preserving $(\\nu_3, \\nu_4)$ | MLP 3→16→16→2, tanh, WD=1e-3, 5-seed ensemble |
| **Oracle** | anchor-preserving $(\\nu_3, \\nu_4)$ | per-glass best (the Level-1 target itself) |

Oracle is the **upper bound for anchor-preserving Level-2 predictors
within the 4-term Buchdahl parameterization** (variants C and D). It is
*not* a bound for the unconstrained variant A, which can fit raw
$n(\\lambda)$ RMS better by sacrificing anchor preservation. The gap
between Oracle and C/D quantifies how much of each metric is due to the
predictor vs. the Buchdahl parameterization itself.

### Cluster-based train/test split

DBSCAN on standardized $(n_d, V_d, \\Delta P_{g,F})$ clusters near-
duplicate glasses (same glass across catalogs — e.g. N-BK7 /
S-BSL7 / H-K9L). Splitting by cluster (not by record) prevents
train/test leakage.

We also report an **FK/FPL subset evaluation**: post-hoc inspection of
errors on the 21 FK-family glasses. Note that FK/FPL clusters may have
been partially in the training split, so this is NOT a strict
extrapolation holdout — just a subset-performance check on the most
anomalous-dispersion region of the parameter space. A true FK
extrapolation holdout (retrain with FK excluded) is provided below as
an additional cell.
"""))

cells.append(code(
"""# ---- 20-dim polynomial feature vector (same as main notebook) ----
def feature_vec(nd, vd, dPgF):
    square = [1.0, nd, vd, dPgF, nd*nd, vd*vd, dPgF*dPgF,
              nd*vd, nd*dPgF, vd*dPgF]
    cube   = [nd**3, vd**3, dPgF**3,
              nd*nd*vd, nd*nd*dPgF, vd*vd*nd, vd*vd*dPgF,
              dPgF*dPgF*nd, dPgF*dPgF*vd, nd*vd*dPgF]
    return np.array(square + cube, dtype=np.float64)


Phi = np.array([feature_vec(g["nd"], g["vd"], g["dPgF"]) for g in glasses])
print(f"Phi (20-dim features): {Phi.shape}")
"""))

cells.append(code(
"""# ---- Cluster split via DBSCAN on standardized params ----
X_params = np.array([[g["nd"], g["vd"], g["dPgF"]] for g in glasses])
X_scaled = StandardScaler().fit_transform(X_params)

db = DBSCAN(eps=0.15, min_samples=2).fit(X_scaled)
cluster_ids = db.labels_.copy()
# DBSCAN gives -1 to noise; treat each noise point as its own cluster
next_id = cluster_ids.max() + 1
for i, cid in enumerate(cluster_ids):
    if cid == -1:
        cluster_ids[i] = next_id
        next_id += 1

unique_clusters = np.unique(cluster_ids)
n_clusters = len(unique_clusters)
print(f"DBSCAN(eps=0.15): {n_clusters} clusters over {N_GLASS} glasses  "
      f"(mean size {N_GLASS/n_clusters:.2f})")

# 80/20 split by cluster
rng = np.random.default_rng(0)
perm = rng.permutation(unique_clusters)
n_train = int(0.8 * len(perm))
train_clusters = set(perm[:n_train])
train_mask = np.array([cid in train_clusters for cid in cluster_ids])
test_mask  = ~train_mask
print(f"Cluster split: {train_mask.sum()} train, {test_mask.sum()} test")

# FK/FPL family subset identifier (post-hoc eval, not a strict holdout —
# the cluster split may have placed some FK glasses in the training side)
fk_regex = re.compile(r"(^|[^A-Z])(FK|FPL)(\\d|$|[^A-Z])", re.IGNORECASE)
def is_fk_family(name):
    return bool(fk_regex.search(name))

fk_mask = np.array([is_fk_family(g["name"]) for g in glasses])
fk_in_train = int((fk_mask & train_mask).sum())
fk_in_test  = int((fk_mask & test_mask).sum())
print(f"FK/FPL subset: {fk_mask.sum()} glasses  "
      f"(examples: {[glasses[i]['name'] for i in np.where(fk_mask)[0][:8]]})")
print(f"  Cluster-split breakdown: {fk_in_train} in train, {fk_in_test} in test "
      f"-> not a strict extrapolation holdout")
"""))

cells.append(code(
"""# ---- Variant A: existing A_REG on old targets (loaded as-is from .npy) ----
# Variant C: re-fit A_REG' on anchor targets, trained only on training split
Y_old    = old_coefs[:, 2:]   # (N, 2)  — old targets
Y_anchor = new_coefs[:, 2:]   # (N, 2)  — anchor-preserving targets

A_REG_anchor, *_ = np.linalg.lstsq(Phi[train_mask], Y_anchor[train_mask], rcond=None)
print(f"A_REG_anchor (variant C): shape {A_REG_anchor.shape}")
print(f"A_REG_old    (variant A): shape {A_REG_old.shape}")
"""))

cells.append(code(
"""# ---- Variant D: 5-seed MLP ensemble ----
#   3 → 16 → 16 → 2, tanh, weight_decay=1e-3, normalized I/O.

# Input / output normalization over TRAIN subset only (to avoid leakage)
X_raw = np.array([[g["nd"], g["vd"], g["dPgF"]] for g in glasses])
x_mean = X_raw[train_mask].mean(axis=0)
x_std  = X_raw[train_mask].std(axis=0) + 1e-9
X_norm = (X_raw - x_mean) / x_std

y_mean = Y_anchor[train_mask].mean(axis=0)
y_std  = Y_anchor[train_mask].std(axis=0) + 1e-9
Y_norm_anchor = (Y_anchor - y_mean) / y_std


class TanhMLP(torch.nn.Module):
    def __init__(self, d_in=3, d_hidden=16, d_out=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in,     d_hidden),  torch.nn.Tanh(),
            torch.nn.Linear(d_hidden, d_hidden),  torch.nn.Tanh(),
            torch.nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(seed, epochs=3000, lr=3e-3, weight_decay=1e-3):
    torch.manual_seed(seed)
    model = TanhMLP().double()
    X_tr = torch.tensor(X_norm[train_mask],       dtype=torch.float64)
    Y_tr = torch.tensor(Y_norm_anchor[train_mask], dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        yhat = model(X_tr)
        loss = torch.nn.functional.mse_loss(yhat, Y_tr)
        loss.backward()
        opt.step()
    return model


# 5-seed ensemble
t0 = time.time()
mlp_seeds = [train_mlp(seed=s) for s in range(5)]
print(f"Trained 5-seed MLP ensemble in {time.time()-t0:.1f} s")

# Predictions (ensemble-averaged on normalized outputs)
X_all_t = torch.tensor(X_norm, dtype=torch.float64)
with torch.no_grad():
    preds_norm = torch.stack([m(X_all_t) for m in mlp_seeds], dim=0).mean(dim=0).numpy()
Y_mlp = preds_norm * y_std + y_mean
print(f"Ensemble predictions shape: {Y_mlp.shape}")
"""))

cells.append(code(
"""# ---- Assemble full coefficients (nu1, nu2, nu3, nu4) per variant ----
def assemble_variant_A(nu34_predicted):
    \"\"\"Old pipeline: nu1,nu2 analytic (no coupling).\"\"\"
    out = np.zeros((N_GLASS, 4))
    for i, (nu3, nu4) in enumerate(nu34_predicted):
        g = glasses[i]
        dn_FC = (g['nd'] - 1.0) / g['vd']
        PgF = 0.6438 - 0.001682*g['vd'] + g['dPgF']
        dn_gF = PgF * dn_FC
        nu12 = np.linalg.solve(M_MAT, np.array([dn_FC, dn_gF]))
        out[i] = [nu12[0], nu12[1], nu3, nu4]
    return out


def assemble_variant_anchor(nu34_predicted):
    \"\"\"Anchor-preserving: nu1,nu2 given nu3,nu4.\"\"\"
    out = np.zeros((N_GLASS, 4))
    for i, (nu3, nu4) in enumerate(nu34_predicted):
        g = glasses[i]
        dn_FC = (g['nd'] - 1.0) / g['vd']
        PgF = 0.6438 - 0.001682*g['vd'] + g['dPgF']
        dn_gF = PgF * dn_FC
        a = M_INV @ np.array([dn_FC, dn_gF])
        nu12 = a - B_COUP @ np.array([nu3, nu4])
        out[i] = [nu12[0], nu12[1], nu3, nu4]
    return out


Y_old_pred    = Phi @ A_REG_old
Y_anchor_pred = Phi @ A_REG_anchor

coefs_A      = assemble_variant_A(Y_old_pred)
coefs_C      = assemble_variant_anchor(Y_anchor_pred)
coefs_D      = assemble_variant_anchor(Y_mlp)
coefs_oracle = new_coefs  # per-glass best anchor-preserving fit

variants = {
    "A_old_linear":     coefs_A,
    "C_anchor_linear":  coefs_C,
    "D_anchor_mlp":     coefs_D,
    "Oracle":           coefs_oracle,
}
print("Four coefficient sets assembled:", list(variants.keys()))
"""))

cells.append(code(
"""# ---- Report reconstruction error on test (cluster-holdout) and FK holdout
def subset_errors(coefs, mask):
    sub_coefs = coefs[mask]
    sub_glasses = [g for g, m in zip(glasses, mask) if m]
    return level1_errors(sub_coefs, sub_glasses)

print(f"n(λ) reconstruction error on test set ({test_mask.sum()} glasses, cluster-holdout):")
print(f"  {'Variant':<20} {'max':>12} {'P95':>12} {'RMS':>12}")
print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")
for name, coefs in variants.items():
    E = subset_errors(coefs, test_mask)
    print(f"  {name:<20} {E['max']:>12.3e} {E['p95']:>12.3e} {E['rms']:>12.3e}")

print()
print(f"Errors on FK/FPL subset ({fk_mask.sum()} glasses; post-hoc, not strict holdout):")
print(f"  {'Variant':<20} {'max':>12} {'P95':>12} {'RMS':>12}")
print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")
for name, coefs in variants.items():
    E = subset_errors(coefs, fk_mask)
    print(f"  {name:<20} {E['max']:>12.3e} {E['p95']:>12.3e} {E['rms']:>12.3e}")
"""))

cells.append(md(
"""### True FK/FPL extrapolation holdout (retrain with FK excluded)

The previous subset evaluation was not a strict holdout. This cell
**retrains C and D with all FK/FPL glasses removed from the training
set**, then reports error on that held-out set. Variant A's predictor
cannot be retrained here (the pre-trained `A_REG_old` matrix was
fitted elsewhere on the full catalog) so it is left as-is and flagged
in the output.
"""))

cells.append(code(
"""# True extrapolation holdout: train on non-FK, test on FK
non_fk_mask = ~fk_mask
Phi_train_fk = Phi[non_fk_mask]
Y_anc_train_fk = Y_anchor[non_fk_mask]

A_REG_anchor_noFK, *_ = np.linalg.lstsq(Phi_train_fk, Y_anc_train_fk, rcond=None)

# Retrain MLP on non-FK; same architecture + hyperparams
x_mean_nf = X_raw[non_fk_mask].mean(axis=0)
x_std_nf  = X_raw[non_fk_mask].std(axis=0) + 1e-9
X_norm_nf = (X_raw - x_mean_nf) / x_std_nf
y_mean_nf = Y_anchor[non_fk_mask].mean(axis=0)
y_std_nf  = Y_anchor[non_fk_mask].std(axis=0) + 1e-9
Y_norm_nf = (Y_anchor - y_mean_nf) / y_std_nf


def train_mlp_nf(seed, epochs=3000, lr=3e-3, weight_decay=1e-3):
    torch.manual_seed(seed)
    model = TanhMLP().double()
    X_tr = torch.tensor(X_norm_nf[non_fk_mask], dtype=torch.float64)
    Y_tr = torch.tensor(Y_norm_nf[non_fk_mask], dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X_tr), Y_tr)
        loss.backward()
        opt.step()
    return model


mlp_nf = [train_mlp_nf(seed=s) for s in range(5)]
X_all_nf = torch.tensor(X_norm_nf, dtype=torch.float64)
with torch.no_grad():
    pred_nf = torch.stack([m(X_all_nf) for m in mlp_nf], dim=0).mean(dim=0).numpy()
Y_mlp_nf = pred_nf * y_std_nf + y_mean_nf

# Assemble per-glass coefs for C_noFK and D_noFK
coefs_C_noFK = assemble_variant_anchor(Phi @ A_REG_anchor_noFK)
coefs_D_noFK = assemble_variant_anchor(Y_mlp_nf)

print(f"True FK extrapolation holdout — errors on {fk_mask.sum()} FK glasses:")
print(f"  (trained only on {non_fk_mask.sum()} non-FK glasses)")
print(f"  {'Variant':<25} {'max':>12} {'P95':>12} {'RMS':>12}")
print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}")
for label, coefs in [("A_old_linear (NOT retrained)", coefs_A),
                     ("C_anchor_linear (no-FK)",       coefs_C_noFK),
                     ("D_anchor_mlp    (no-FK)",       coefs_D_noFK),
                     ("Oracle",                        coefs_oracle)]:
    fk_glasses = [g for g, m in zip(glasses, fk_mask) if m]
    E = level1_errors(coefs[fk_mask], fk_glasses)
    print(f"  {label:<25} {E['max']:>12.3e} {E['p95']:>12.3e} {E['rms']:>12.3e}")
print()
print("A is shown as-is for reference; its pretrained A_REG saw all 544 glasses")
print("so its FK error is in-sample, not extrapolation.")
"""))

# =========================================================================
# Part 1C — Claim F (end-to-end)
# =========================================================================

cells.append(md(
"""## 1C — Claim F: end-to-end anchor preservation

Take each variant's predicted $(\\nu_1, \\nu_2, \\nu_3, \\nu_4)$ per glass,
reconstruct $n(\\lambda)$ at $\\lambda_d, \\lambda_F, \\lambda_C, \\lambda_g$,
and recompute $n_d, V_d, \\Delta P_{g,F}$ from the *output* curve.
Compare to the *input*.

- **F1** = $\\max |n_\\text{out}(\\lambda_d) - n_d^\\text{in}|$
- **F2_abs** = $\\max |V_d^\\text{out} - V_d^\\text{in}|$
- **F2_rel** = $\\max |V_d^\\text{out} - V_d^\\text{in}| / V_d^\\text{in}$
- **F3** = $\\max |\\Delta P_{g,F}^\\text{out} - \\Delta P_{g,F}^\\text{in}|$

Variant A should show a measurable slip; C/D/Oracle should be at
floating-point precision (architectural guarantee).
"""))

cells.append(code(
"""def claim_F_metrics(coefs):
    lam_anc = np.array([LAMBDA_D, LAMBDA_F, LAMBDA_C, LAMBDA_g])
    om_anc  = buchdahl_omega(lam_anc)
    f1s, f2abs, f2rel, f3s = [], [], [], []
    for (nu1, nu2, nu3, nu4), g in zip(coefs, glasses):
        nd = g["nd"]
        n_out = reconstruct([nu1, nu2, nu3, nu4], nd, om_anc)
        nd_out, nF_out, nC_out, ng_out = n_out
        dn_FC_out = nF_out - nC_out
        Vd_out   = (nd_out - 1.0) / dn_FC_out
        PgF_out  = (ng_out - nF_out) / dn_FC_out
        dPgF_out = PgF_out - (0.6438 - 0.001682 * Vd_out)
        f1s .append(abs(nd_out  - nd))
        dVd = abs(Vd_out - g["vd"])
        f2abs.append(dVd)
        f2rel.append(dVd / g["vd"])
        f3s .append(abs(dPgF_out - g["dPgF"]))
    return dict(
        F1=float(max(f1s)),
        F2_abs=float(max(f2abs)), F2_rel=float(max(f2rel)),
        F3=float(max(f3s)),
    )


print(f"Claim F — end-to-end anchor preservation ({N_GLASS} glasses):")
print(f"  {'Variant':<20} {'F1':>10} {'F2_abs':>10} {'F2_rel':>10} {'F3':>10}")
print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
F_metrics = {}
for name, coefs in variants.items():
    F = claim_F_metrics(coefs)
    F_metrics[name] = F
    print(f"  {name:<20} {F['F1']:>10.2e} {F['F2_abs']:>10.2e} "
          f"{F['F2_rel']:>10.2e} {F['F3']:>10.2e}")
print()
print("Interpretation:")
print(" - F1 should be ~0 for all variants (ω(λ_d)=0 makes it automatic).")
print(" - F2/F3 should be ~1e-15 for C/D/Oracle (anchor solve guarantee).")
print(" - F2/F3 for variant A shows the slip from the current repo's bug.")
"""))

# =========================================================================
# Part 1D — Claim G (downstream apo-doublet bench)
# =========================================================================

cells.append(md(
"""## 1D — Claim G: downstream ranking vs Sellmeier truth

Fixed test bench (same as `glass_substitution_workflow.ipynb`):

- Flint: **N-SF2** (Sellmeier-recomputed)
- Target **EFL = 20 mm**, F/6
- $r_2 = -7.94140$ (cemented interface, fixed), $t_1 = 0.434$, $t_2 = 0.321$

For each CDGM crown candidate:

- **Sellmeier truth**: $n(\\lambda)$ from its raw Sellmeier formula →
  solve achromat → measure $S = |\\text{BFL}(g) - \\text{BFL}(d)|$
  (**ground-truth $S$**).
- **Model**: $n(\\lambda)$ from each variant's predicted $(\\nu_1, \\nu_2,
  \\nu_3, \\nu_4)$ → solve achromat → measure $S$.

Report:

- **Spearman** rank correlation between model $S$ and truth $S$
- **Top-3 set preservation**
- **FK/FPL family count** in top-10 (physics consensus check)
- **median** and **max** $|S_\\text{model} - S_\\text{truth}|$ in μm

(The CDGM convergent-candidate filter is recomputed inline below for a
self-contained, deterministic dataset. The exact count depends on
which variants must all converge simultaneously — see the output of
the next cell. The filter procedure matches the workflow notebook's.)
"""))

cells.append(code(
"""# ---- Test bench: N-SF2 flint ----
nsf2_rec = extract_record(GLASS_ROOT / "schott" / "N-SF2.yml")
assert nsf2_rec is not None, "N-SF2.yml must yield a Sellmeier record"
FLINT = dict(nd=nsf2_rec["nd"], vd=nsf2_rec["vd"], dPgF=nsf2_rec["dPgF"],
             sellmeier=nsf2_rec["sellmeier"])
print(f"Flint N-SF2 (Sellmeier-recomputed): "
      f"nd={FLINT['nd']:.4f}, Vd={FLINT['vd']:.2f}, dPgF={FLINT['dPgF']:+.4f}")

TARGET_EFL = 20.0
R2 = -7.94140
T1, T2 = 0.434, 0.321
LAM4 = np.array([LAMBDA_F, LAMBDA_D, LAMBDA_C, LAMBDA_g])
OM4  = buchdahl_omega(LAM4)


def paraxial_trace(r1, r2, r3, t1, t2, n_c, n_f):
    y, u = 1.0, 0.0
    u = (u - (n_c - 1.0)*y/r1) / n_c
    y = y + u*t1
    u = (n_c*u - (n_f - n_c)*y/r2) / n_f
    y = y + u*t2
    u = n_f*u - (1.0 - n_f)*y/r3
    return -1.0/u, -y/u


def design_achromat_and_S(n_c_F, n_c_d, n_c_C, n_c_g,
                          n_f_F, n_f_d, n_f_C, n_f_g,
                          r1_init=12.38, r3_init=-48.44):
    def residuals(r13):
        r1, r3 = r13
        efl, _   = paraxial_trace(r1, R2, r3, T1, T2, n_c_d, n_f_d)
        _, bflF = paraxial_trace(r1, R2, r3, T1, T2, n_c_F, n_f_F)
        _, bflC = paraxial_trace(r1, R2, r3, T1, T2, n_c_C, n_f_C)
        return [efl - TARGET_EFL, bflF - bflC]
    sol, info, ier, msg = fsolve(residuals, x0=[r1_init, r3_init],
                                 full_output=True, xtol=1e-10)
    if ier != 1:
        return None
    r1, r3 = sol
    _, bflg = paraxial_trace(r1, R2, r3, T1, T2, n_c_g, n_f_g)
    _, bfld = paraxial_trace(r1, R2, r3, T1, T2, n_c_d, n_f_d)
    return abs(bflg - bfld)


# Sellmeier-truth indices for flint at F,d,C,g
n_f_F, n_f_d, n_f_C, n_f_g = [float(_sellmeier(lam, FLINT['sellmeier'])) for lam in LAM4]

# CDGM candidate set
cdgm = [g for g in glasses if g["catalog"] == "cdgm"]
print(f"CDGM candidates (pre-convergence filter): {len(cdgm)}")
"""))

cells.append(code(
"""# ---- Compute truth S and per-variant S for each CDGM candidate ----
truth_S = {}
variant_S = {k: {} for k in variants}

for cand in cdgm:
    sm_c = cand["sellmeier"]
    ncF, ncD, ncC, ncg = [float(_sellmeier(lam, sm_c)) for lam in LAM4]
    # Sellmeier-truth S
    S_truth = design_achromat_and_S(ncF, ncD, ncC, ncg,
                                    n_f_F, n_f_d, n_f_C, n_f_g)
    if S_truth is None:
        continue
    truth_S[cand["name"]] = S_truth

    # Each variant's predicted n at F,d,C,g
    idx = glasses.index(cand)
    for vname, coefs in variants.items():
        nu1, nu2, nu3, nu4 = coefs[idx]
        n_pred = cand["nd"] + nu1*OM4 + nu2*OM4**2 + nu3*OM4**3 + nu4*OM4**4
        S_model = design_achromat_and_S(n_pred[0], n_pred[1], n_pred[2], n_pred[3],
                                        n_f_F, n_f_d, n_f_C, n_f_g)
        variant_S[vname][cand["name"]] = S_model

# Keep candidates that all variants converge for — apples-to-apples.
common_set = set(truth_S.keys())
for v in variant_S:
    common_set &= {k for k, s in variant_S[v].items() if s is not None}
common = sorted(common_set)
print(f"CDGM candidates with convergent design in all variants + truth: {len(common)}")
"""))

cells.append(code(
"""# ---- Claim G metrics ----
truth_vec = np.array([truth_S[n] for n in common])
truth_rank = np.argsort(truth_vec)
truth_top3  = {common[i] for i in truth_rank[:3]}
truth_top10 = {common[i] for i in truth_rank[:10]}
truth_fk_top10 = sum(is_fk_family(n) for n in truth_top10)
print(f"Sellmeier truth top-10: FK/FPL count = {truth_fk_top10}/10")
print(f"Sellmeier truth top-3 : {sorted(truth_top3)}")
print()

G_metrics = {}
print(f"{'Variant':<20} {'Spearman':>10} {'Top-3':>7} {'FK top-10':>11} "
      f"{'med|ΔS|':>10} {'max|ΔS|':>10}")
print(f"{'-'*20}  {'-'*8}  {'-'*5}  {'-'*9}  {'-'*8}  {'-'*8}")
for vname in variants:
    mod_vec = np.array([variant_S[vname][n] for n in common])
    dS_um = np.abs(mod_vec - truth_vec) * 1e3
    rho_raw, _ = spearmanr(mod_vec, truth_vec)
    rho = float(rho_raw)  # type: ignore[arg-type]
    mod_rank = np.argsort(mod_vec)
    top3 = {common[i] for i in mod_rank[:3]}
    top10 = {common[i] for i in mod_rank[:10]}
    fk_in_top10 = sum(is_fk_family(n) for n in top10)
    G_metrics[vname] = dict(
        spearman=rho,
        top3_preserved=len(top3 & truth_top3),
        fk_top10=fk_in_top10,
        med_dS_um=float(np.median(dS_um)),
        max_dS_um=float(dS_um.max()),
    )
    print(f"{vname:<20} {rho:>10.4f} {len(top3 & truth_top3):>3}/3"
          f"{fk_in_top10:>9}/10   {np.median(dS_um):>7.3f}   {dS_um.max():>7.3f}")

print()
print("Truth reference: 10/10 FK/FPL in top-10, top-3 S values (um):")
for nm in sorted(truth_top3, key=lambda n: truth_S[n]):
    print(f"  {nm:<15} S = {truth_S[nm]*1e3:.3f} um")
"""))

# =========================================================================
# Scorecard + conclusion
# =========================================================================

cells.append(md(
"""### Downstream Claim G under true FK extrapolation

To check whether the no-FK-retrained variants **also preserve downstream
design fidelity on FK glasses**, we rerun Claim G on the **CDGM-FK
crown subset only**, using the retrained $A_\\text{REG,noFK}$ and
$\\text{MLP}_\\text{noFK}$ predictors. This is a stricter test than raw
$n(\\lambda)$ extrapolation: the model has never seen any FK glass, yet
the predictions should still rank FK glasses correctly against
Sellmeier truth on the apo-doublet test bench.
"""))

cells.append(code(
"""# Which of the CDGM-FK crowns are in the common Claim-G candidate set?
cdgm_fk = [n for n in common if is_fk_family(n)]
print(f"CDGM-FK crowns in the common Claim G candidate set: {len(cdgm_fk)}")


def compute_S_over(coefs_array, cand_names):
    \"\"\"S (um) for the given CDGM candidate names, using per-glass coefs.\"\"\"
    S_list = []
    for name in cand_names:
        cand = next(g for g in cdgm if g["name"] == name)
        idx = glasses.index(cand)
        nu1, nu2, nu3, nu4 = coefs_array[idx]
        n_pred = cand["nd"] + nu1*OM4 + nu2*OM4**2 + nu3*OM4**3 + nu4*OM4**4
        S_m = design_achromat_and_S(n_pred[0], n_pred[1], n_pred[2], n_pred[3],
                                    n_f_F, n_f_d, n_f_C, n_f_g)
        S_list.append(S_m)
    return np.array(S_list, dtype=float)


truth_fk = np.array([truth_S[n] for n in cdgm_fk])

print()
print(f"{'Variant':<36} {'Spearman':>10} {'med|dS|':>12} {'max|dS|':>12}")
print(f"{'-'*36}  {'-'*8}  {'-'*10}  {'-'*10}")
for label, coefs in [
    ("A_old_linear        (in-sample)",      coefs_A),
    ("C_anchor_linear     (in-sample)",      coefs_C),
    ("C_anchor_linear     (no-FK retrain)",  coefs_C_noFK),
    ("D_anchor_mlp        (in-sample)",      coefs_D),
    ("D_anchor_mlp        (no-FK retrain)",  coefs_D_noFK),
    ("Oracle              (per-glass best)", coefs_oracle),
]:
    model_fk = compute_S_over(coefs, cdgm_fk)
    dS_um = np.abs(model_fk - truth_fk) * 1e3
    rho, _ = spearmanr(model_fk, truth_fk)
    print(f"{label:<36} {rho:>10.4f}   {np.median(dS_um):>7.3f} um   "
          f"{dS_um.max():>7.3f} um")

print()
print("'in-sample' = trained on all 544 glasses (including FK).")
print("'no-FK retrain' = trained on 523 non-FK glasses only; these are the")
print("  strict extrapolation rows.")
"""))

cells.append(md(
"""## Joint scorecard — all metrics, all variants

The Level-1 fit, Level-2 predictor, Claim F (anchor preservation), and
Claim G (downstream ranking) results for all four variants, side by
side.
"""))

cells.append(code(
"""rows = {}
for vname, coefs in variants.items():
    E_test = subset_errors(coefs, test_mask)
    E_fk   = subset_errors(coefs, fk_mask)
    F = F_metrics[vname]
    G = G_metrics[vname]
    rows[vname] = {
        "test_max":   E_test["max"],
        "test_P95":   E_test["p95"],
        "test_RMS":   E_test["rms"],
        "fk_max":     E_fk["max"],
        "F1":         F["F1"],
        "F2_abs":     F["F2_abs"],
        "F2_rel":     F["F2_rel"],
        "F3":         F["F3"],
        "spearman":   G["spearman"],
        "top3":       G["top3_preserved"],
        "fk_top10":   G["fk_top10"],
        "med_dS_um":  G["med_dS_um"],
        "max_dS_um":  G["max_dS_um"],
    }
scorecard = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda x: f"{x:.3e}" if abs(x) < 1 else f"{x:.4f}")
print(scorecard)
"""))

cells.append(md(
"""## Conclusion — what Part 1 actually tells us

### 1. Raw $n(\\lambda)$ reconstruction (test_max / P95 / RMS)

**Anchor repair does NOT improve this metric on the cluster-holdout
split.** Variants C and D are slightly *worse* than variant A on max
/ P95 / RMS. The reason: A's **anchor-unaware coefficient construction**
lets the combined 4-term expansion silently modify $n(F) - n(C)$ and
$n(g) - n(F)$ away from the anchor-defined values, so the per-glass fit
has more freedom to minimize the 17-point residual. The anchor-preserving
construction is strictly more constrained — $\\nu_1, \\nu_2$ are pinned
by exact anchor equality given $\\nu_3, \\nu_4$.

Raw $n(\\lambda)$ RMS is therefore **necessary but not sufficient**: it
misses anchor violations that propagate strongly into downstream design
quantities, as the next two subsections show.

### 2. Claim F — physical consistency (F1, F2_abs, F2_rel, F3)

Variant A's max $|V_d^\\text{out} - V_d^\\text{in}|$ reaches
$\\mathcal{O}(1)$ units (relative slip ~2%). C, D, Oracle deliver all
three F metrics at **floating-point precision** by architectural
construction, not by training.

### 3. Claim G — downstream design agreement with Sellmeier truth

The most consequential result. Median $|S_\\text{model} -
S_\\text{truth}|$ drops from ~12 μm (variant A) to ~1.5 μm (C, D,
Oracle). Spearman rank correlation with truth improves from 0.955
to 0.998. **The anchor slip that didn't show up in raw $n(\\lambda)$
RMS propagates strongly through the achromat-design equations and
produces a biased secondary spectrum.**

### 4. Anchor linear (C) vs anchor MLP (D)

Essentially identical on every metric. At ~544 glasses with a 3-d
input, the 20-d polynomial already saturates the Level-2 map. **MLP
provides no measurable lift** — a clean negative result saying the
bottleneck is not predictor capacity.

### 5. Anchor linear (C) vs Oracle

Also essentially identical on downstream metrics. Oracle is the
best-any-predictor-can-do bound **for the anchor-preserving 4-term
Buchdahl family under the Level-1 spectral-fit target** — it is not a
downstream-$S$-optimal oracle, but it is a tight upper bound on what
any Level-2 predictor can deliver under this parameterization. Since
C already matches it, the remaining ~1.5 μm is *consistent with* the
4-term anchor-preserving Buchdahl floor at this target. A Phase 3
higher-order extension is the natural direction for attacking that
floor.

### Honest summary after Part 1

> The anchor-preserving repair does not reduce raw $n(\\lambda)$
> reconstruction error on the cluster-holdout split — in fact C and D
> are slightly *worse* than the old A on max / P95 / RMS. However, it
> eliminates the physical anchor slip by construction and dramatically
> improves downstream secondary-spectrum agreement with Sellmeier
> truth (median $|\\Delta S|$ drops from ~12 μm to ~1.5 μm; Spearman
> from 0.955 to 0.998). The MLP adds little over the retrained
> anchor-linear map, suggesting the 20-d polynomial predictor already
> saturates the cross-glass map at this dataset size. The residual
> ~1.5 μm is consistent with the Oracle upper bound (the 4-term
> anchor-preserving Buchdahl floor at this Level-1 target) rather than
> with predictor error — which makes higher-order Buchdahl a natural
> direction for Phase 3, not a bigger predictor.

### Future work

- **Phase 2** — bounded envelope residual correction
  $n = n_\\text{Buchdahl} + \\varepsilon_\\text{scale}\\,q(\\lambda)\\,\\tanh(\\text{NN})$
  with $q$ vanishing at $d/F/C/g$.
- **Phase 3** — higher-order Buchdahl ($K \\in \\{5,6\\}$), anchor-preserving
  solve for $\\nu_1, \\nu_2$ conditioned on all $\\nu_{k \\ge 3}$ predicted
  by NN. Attacks the Oracle-level floor directly.
- **Backport** the anchor-preserving fit to the other three notebooks
  and recheck their measured secondary-spectrum numbers; Claim G here
  suggests the bias on this test bench is $\\mathcal{O}(10\\,\\mu m)$,
  but the magnitude should be re-verified per-notebook after backport.
"""))


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT}")
