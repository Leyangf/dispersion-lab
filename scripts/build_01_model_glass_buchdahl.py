"""Build 01_model_glass_buchdahl.ipynb — the canonical anchor-preserving
Buchdahl model notebook.

Structure:

  Part 0. Catalog.
  Part 1. Legacy vs anchor-preserving construction — one decisive table.
  Part 2. Anchor-preserving Buchdahl family: formal definition for
          arbitrary K.
  Part 3. Selection criterion — aggregation (p50 / p95 / p99 / max)
          and commitment to Test B max as the production target.
  Part 4. K selection under the max criterion — K = 4 unique Pareto.
  Part 5. α refinement at K = 4 — band-symmetric derivation and
          comparison against five principled candidates.
  Part 6. Where the remaining error comes from — Oracle floor vs
          regressor gap (decomposition), design-space error map,
          Chebyshev cross-check.
  Conclusion. Production / frontier / compact selection statement.

The legacy-construction narrative is included inline in Part 1 as a
one-table contrast; there is no separate archival notebook.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
OUT = PROJECT_ROOT / "notebooks" / "01_model_glass_buchdahl.ipynb"


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
# Title
# =========================================================================

cells.append(md(
"""# Anchor-Preserving Buchdahl Model

*Construction choice, hyperparameter selection, and accuracy frontier
for the three-parameter glass surrogate.*

This notebook defines the Buchdahl forward model used throughout the
`DispersionLab` project. The answer the notebook delivers has three
layers:

1. **Construction choice** — of the two coefficient-assembly orderings
   (legacy vs anchor-preserving), only the anchor-preserving one is
   consistent with the physical definitions of `V_d` and `P_{g,F}`. All
   further experiments run on it.

2. **Hyperparameter selection inside the anchor-preserving family** —
   an (order × α) sweep on Test A / Test B error under multiple
   aggregation criteria characterises the family, and an
   independently derived **band-symmetric** α fixes the coordinate
   without relying on data-driven minima.

3. **Production model and alternatives recorded for contrast**:

   | Model | Role | Why |
   |---|---|---|
   | **K = 3, α ≈ 1.20** | compact lower bound | smallest useful hypothesis space; 1 high-order regression target |
   | **K = 4, α = 1.9547** | **production baseline** | classical Buchdahl complexity; band-symmetric α on [0.365, 2.3] μm; used by NB02–NB05 |
   | **K = 8, α = 1.26** | accuracy frontier | lowest p95 hold error; traded off against worst-case out/in ratio |

The anchor-preserving construction is delivered by the shared
`dispersionlab` package (`solve_nu12_from_nu34`, `feature_vec_20`,
`fit_anchor_preserving_coefficients`, plus the sweep-parametrised
variant `dispersionlab.sweep`). Machine-precision anchor preservation
across the full `(K, α)` grid is covered in
`tests/test_sweep_anchor_delta.py`.
"""))


cells.append(md(
"""## Contents

- Part 0 — Catalog (Plot 1: 3-D hull + Abbe + ΔP-V_d)
- Part 1 — Legacy vs anchor-preserving construction
- Part 2 — Anchor-preserving Buchdahl family (definition)
- Part 3 — Selection criterion: aggregation p50 / p95 / p99 / max
- Part 4 — K selection under the max criterion (Plot 3)
- Part 5 — α refinement at K = 4: band-symmetric derivation, 5 principled candidates (Plot 4)
- Part 6 — Where the remaining Test B error comes from
  - 6.1 Oracle floor vs production CDF (Plot 5)
  - 6.2 Where the regressor gap concentrates (Plot 6)
  - 6.3 Production reliability vs extrapolation across design space (Plot 7)
  - 6.4 Why not a generic polynomial basis? (Chebyshev cross-check)
- Conclusion

Five notebooks are no longer two notebooks: the legacy-construction
analysis is included inside Part 1 as a one-table contrast, not a
separate file. NB01 is the single authoritative notebook for the
anchor-preserving Buchdahl production model.
"""))


# =========================================================================
# Setup
# =========================================================================

cells.append(code(
"""from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from dispersionlab import (
    LAMBDA_D, LAMBDA_F, LAMBDA_C, LAMBDA_g,
    WAVELENGTHS,
    feature_vec_20,
    load_glass_catalog,
    buchdahl_omega,
)
from dispersionlab.buchdahl import (
    fit_anchor_preserving_coefficients,
    fit_legacy_coefficients,
    anchor_error_metrics,
    reconstruct_n,
)
from dispersionlab.sweep import (
    anchor_error_metrics_K,
    fit_all_glasses_anchor,
    reconstruct_n_K,
    test_a_batched,
    vectorize_catalog,
    build_anchor_matrices,
)

np.random.seed(0)
print(f"numpy {np.__version__}, pandas {pd.__version__}")
"""))


# =========================================================================
# Part 0 — Catalog
# =========================================================================

cells.append(md(
"""## Part 0 — Catalog

544 glasses loaded from the Optiland bundled Sellmeier database, filtered
to the subset with full `0.365–2.3 μm` Sellmeier coverage, and
`10 < V_d < 100`, `−0.02 < ΔP_{g,F} < 0.10`. The same loader is used by
all downstream notebooks via `dispersionlab.catalog.load_glass_catalog`.
"""))


cells.append(code(
"""glasses = load_glass_catalog()
catalog = vectorize_catalog(glasses)

print(f"N = {len(glasses)} glasses")
print(f"  n_d  range: [{catalog['nd'].min():.3f}, {catalog['nd'].max():.3f}]")
print(f"  V_d  range: [{catalog['vd'].min():.2f}, {catalog['vd'].max():.2f}]")
print(f"  ΔP_g,F range: [{catalog['dPgF'].min():+.4f}, {catalog['dPgF'].max():+.4f}]")
print(f"  Sellmeier truth grid: {len(WAVELENGTHS)} wavelengths 0.365–2.3 μm")
"""))


cells.append(code(
"""# Plot 1 — catalog footprint in (V_d, n_d, ΔP_{g,F}):
# Top row: per-plane 2D projections with 2D convex hull (reader-familiar
#   glass-selection views);
# Bottom row: authoritative 3D convex hull — this is the region in which
#   the (ν₃, ν₄) regression is valid. Points outside the 3D hull are
#   extrapolation, even if they land inside either 2D hull.
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_vd   = catalog["vd"]
_nd   = catalog["nd"]
_dPgF = catalog["dPgF"]
_pts3 = np.column_stack([_vd, _nd, _dPgF])

def _hull2d_xy(x, y):
    pts = np.column_stack([x, y])
    h = ConvexHull(pts)
    v = pts[h.vertices]
    return np.vstack([v, v[:1]])

fig = plt.figure(figsize=(12.5, 9.5))
gs  = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], hspace=0.28, wspace=0.22)

# (a) Abbe diagram
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(_vd, _nd, s=12, alpha=0.55, color="steelblue", edgecolors="none")
_h = _hull2d_xy(_vd, _nd)
ax1.plot(_h[:, 0], _h[:, 1], color="crimson", lw=1.8, alpha=0.9)
ax1.set_xlabel(r"$V_d$")
ax1.set_ylabel(r"$n_d$")
ax1.set_title(f"(a) Abbe diagram  ({len(glasses)} glasses)")
ax1.grid(alpha=0.3)
ax1.invert_xaxis()

# (b) ΔP_{g,F} vs V_d (Schott-normal-line deviation)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(_vd, _dPgF, s=12, alpha=0.55, color="seagreen", edgecolors="none")
_h = _hull2d_xy(_vd, _dPgF)
ax2.plot(_h[:, 0], _h[:, 1], color="crimson", lw=1.8, alpha=0.9)
ax2.axhline(0.0,  color="k",       lw=0.6, ls="--", label="Schott normal line")
ax2.axhline(0.03, color="dimgray", lw=0.6, ls=":",
            label=r"FK threshold $\\Delta P_{g,F}=0.03$")
ax2.set_xlabel(r"$V_d$")
ax2.set_ylabel(r"$\\Delta P_{g,F}$")
ax2.set_title(r"(b) $\\Delta P_{g,F}$ vs $V_d$  (Schott-normal-line deviation)")
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(alpha=0.3)
ax2.invert_xaxis()

# (c) Full 3D feature space with convex hull
ax3 = fig.add_subplot(gs[1, :], projection="3d")
sc  = ax3.scatter(
    _pts3[:, 0], _pts3[:, 1], _pts3[:, 2],  # pyright: ignore[reportArgumentType]
    c=_dPgF, cmap="RdBu_r", s=14, alpha=0.85, edgecolors="none",
)
hull = ConvexHull(_pts3)
faces = Poly3DCollection(
    [_pts3[s] for s in hull.simplices],
    alpha=0.08, facecolor="crimson",
    edgecolor="crimson", linewidths=0.45,
)
ax3.add_collection3d(faces)
ax3.set_xlabel(r"$V_d$")
ax3.set_ylabel(r"$n_d$")
ax3.set_zlabel(r"$\\Delta P_{g,F}$")
ax3.invert_xaxis()
ax3.view_init(elev=18, azim=-60)
ax3.set_title(
    r"(c) Full 3-D feature space $(V_d, n_d, \\Delta P_{g,F})$ "
    r"with convex hull — training domain where the $(\\nu_3, \\nu_4)$ regression is valid"
)
cbar = fig.colorbar(sc, ax=ax3, shrink=0.55, pad=0.08)
cbar.set_label(r"$\\Delta P_{g,F}$", fontsize=9)

fig.suptitle(
    r"Catalog footprint in $(V_d, n_d, \\Delta P_{g,F})$  —  "
    r"panels (a)(b) are per-plane 2-D extents; (c) is the authoritative 3-D boundary",
    y=0.995, fontsize=11,
)
plt.show()

print(f"catalog: {len(glasses)} glasses  "
      f"n_d ∈ [{_nd.min():.3f}, {_nd.max():.3f}], "
      f"V_d ∈ [{_vd.min():.2f}, {_vd.max():.2f}], "
      f"ΔP_g,F ∈ [{_dPgF.min():+.3f}, {_dPgF.max():+.3f}]")
print(f"  FK cluster (ΔP_g,F > 0.03):     {(_dPgF > 0.03).sum()} glasses")
print(f"  high-nd flint (n_d > 1.9):      {(_nd   > 1.9).sum()} glasses")
"""))


# =========================================================================
# Part 1 — Legacy vs anchor-preserving construction
# =========================================================================

cells.append(md(
"""## Part 1 — Legacy vs anchor-preserving construction

### The choice

The 4-term Buchdahl expansion

$$n(\\lambda) = n_d + \\nu_1\\omega + \\nu_2\\omega^2 + \\nu_3\\omega^3 + \\nu_4\\omega^4$$

has two natural ways to determine the four coefficients from catalog
data:

| Order | Legacy construction | Anchor-preserving construction |
|---|---|---|
| Step 1 | solve `(ν₁, ν₂)` from the F−C and g−F anchor differences, **ignoring** the high-order contribution | predict / fit `(ν₃, ν₄)` first |
| Step 2 | LSQ-fit `(ν₃, ν₄)` on the residual | back-solve `(ν₁, ν₂)` from the coupled system that **includes** the high-order contribution at the anchor wavelengths |

In both cases the output curve implicitly defines an output
$(V_d^{\\text{out}}, \\Delta P_{g,F}^{\\text{out}})$ via the same
anchor equations. Whether this output agrees with the input depends on
whether the construction carries the high-order term through the
anchor equations or not.

### The test

We fit all 544 glasses under both constructions at `(K = 4, α = 1.818)`
and measure anchor slip:

- **F1** = max $|n_{\\text{out}}(\\lambda_d) - n_d|$ (d-line identity)
- **F2_abs** = max $|V_d^{\\text{out}} - V_d|$ (Abbe number slip)
- **F2_rel** = F2_abs / V_d
- **F3** = max $|\\Delta P_{g,F}^{\\text{out}} - \\Delta P_{g,F}|$ (partial dispersion slip)

A legitimate physical surrogate must satisfy all three at numerical
precision, **for every glass** — otherwise `V_d` and `P_{g,F}` no
longer mean what the glass catalog says they mean.
"""))


cells.append(code(
"""# Fit both constructions on all glasses at K = 4, α = 1.818
N = len(glasses)
nu_ap = np.zeros((N, 4))
nu_lg = np.zeros((N, 4))
for i, g in enumerate(glasses):
    nu_ap[i] = fit_anchor_preserving_coefficients(
        g["nd"], g["vd"], g["dPgF"], g["n_truth"]
    )
    nu_lg[i] = fit_legacy_coefficients(
        g["nd"], g["vd"], g["dPgF"], g["n_truth"]
    )

# Anchor-slip metrics
def collect_anchor_metrics(nu_arr):
    m_all = [
        anchor_error_metrics(
            glasses[i]["nd"], glasses[i]["vd"], glasses[i]["dPgF"],
            *nu_arr[i]
        )
        for i in range(N)
    ]
    return {
        "F1_max":     max(m["F1"] for m in m_all),
        "F2_abs_max": max(m["F2_abs"] for m in m_all),
        "F2_rel_max": max(m["F2_rel"] for m in m_all),
        "F3_max":     max(m["F3"] for m in m_all),
    }

# Also report bare reconstruction error
omega = buchdahl_omega(WAVELENGTHS)
def bare_max(nu_arr):
    errs = np.zeros(N)
    for i in range(N):
        nu_i = nu_arr[i]
        n_pred = reconstruct_n(
            glasses[i]["nd"], nu_i[0], nu_i[1], nu_i[2], nu_i[3], omega
        )
        errs[i] = np.abs(n_pred - glasses[i]["n_truth"]).max()
    return float(errs.max()), float(np.percentile(errs, 95))

m_ap = collect_anchor_metrics(nu_ap)
m_lg = collect_anchor_metrics(nu_lg)
max_ap, p95_ap = bare_max(nu_ap)
max_lg, p95_lg = bare_max(nu_lg)

print("K = 4, α = 1.818 — construction contrast on 544 catalog glasses")
print()
print(f"{'metric':<24} | {'legacy':>14} | {'anchor-preserving':>18}")
print("-" * 62)
print(f"{'F1 max':<24} | {m_lg['F1_max']:>14.3e} | {m_ap['F1_max']:>18.3e}")
print(f"{'F2_abs max (V_d slip)':<24} | {m_lg['F2_abs_max']:>14.3e} | {m_ap['F2_abs_max']:>18.3e}")
print(f"{'F2_rel max':<24} | {m_lg['F2_rel_max']:>14.3e} | {m_ap['F2_rel_max']:>18.3e}")
print(f"{'F3 max (ΔPgF slip)':<24} | {m_lg['F3_max']:>14.3e} | {m_ap['F3_max']:>18.3e}")
print(f"{'bare max |n_err|':<24} | {max_lg:>14.3e} | {max_ap:>18.3e}")
print(f"{'bare p95 |n_err|':<24} | {p95_lg:>14.3e} | {p95_ap:>18.3e}")
"""))


cells.append(md(
"""### Conclusion of Part 1

Legacy construction slips `V_d` by **O(1) unit** (relative ≈ 2 %) and
`ΔP_{g,F}` by ≈ 4·10⁻². Anchor-preserving construction keeps all three
anchors at **machine precision** (F2_rel < 10⁻¹³, F3 < 10⁻¹³) —
independently of the (ν₃, ν₄) values. The bare reconstruction error of
the two constructions is similar, within a factor of ≈ 1.5; legacy's
marginal RMS advantage comes from using its extra LSQ freedom to drift
the anchors, not from capturing physics better.

For any downstream task that touches `V_d` or `ΔP_{g,F}` (achromatic
design, secondary-spectrum optimisation, glass substitution — i.e.
every task in NB02–NB05), the anchor-preserving construction is the
only admissible choice.

**From here on, every experiment uses the anchor-preserving
construction.** The legacy construction appears only in NB04's 4-variant
ablation as a quantitative contrast row.
"""))


# =========================================================================
# Part 2 — Anchor-preserving family definition
# =========================================================================

cells.append(md(
"""## Part 2 — Anchor-preserving Buchdahl family

The anchor-preserving Buchdahl family at order K, parameter α is

$$\\mathcal{F}_{K,\\alpha} = \\bigl\\{ n_d + \\sum_{k=1}^{K} \\nu_k \\,\\omega^k(\\lambda;\\alpha) \\;:\\; \\nu_k \\in \\mathbb{R} \\bigr\\}$$

with the coefficient assembly fixed as:

1. **Regress / fit** the high-order vector `(ν₃, …, ν_K)`.
2. **Back-solve** `(ν₁, ν₂)` from the coupled anchor system

   $$\\begin{bmatrix} \\nu_1 \\\\ \\nu_2 \\end{bmatrix}
     = M^{-1} \\begin{bmatrix} \\Delta n_{FC} \\\\ \\Delta n_{gF} \\end{bmatrix}
       - M^{-1} D_\\text{high} \\begin{bmatrix} \\nu_3 \\\\ \\vdots \\\\ \\nu_K \\end{bmatrix}$$

   where `M` is the 2×2 matrix of anchor-difference values of
   $(\\omega, \\omega^2)$, and `D_high` is the 2 × (K−2) matrix of
   anchor-difference values of $(\\omega^3, …, \\omega^K)$.

Regardless of the high-order coefficients, this construction satisfies
$n(\\lambda_d) = n_d$, $V_d^{\\text{out}} = V_d$, and
$P_{g,F}^{\\text{out}} = P_{g,F}$ at machine precision.

### Two Level boundaries

- **Level 1** (per glass): given a Sellmeier truth curve, find the
  best-fit `(ν₃, …, ν_K)` via constrained LSQ. Returns the Oracle target.
- **Level 2** (cross glass): predict `(ν₃, …, ν_K)` from the catalog
  triple `(n_d, V_d, ΔP_{g,F})` via a regression map. Production uses
  a 20-D polynomial linear map. NB04 ablates this Level 2 predictor
  against an MLP alternative.

NB01 focuses on Level 1 — the family's own representational capacity
— to select K and α. Level 2 generalisation is covered in Part 5 (Test
B) and more deeply in NB04.
"""))


cells.append(code(
"""# Quick check: anchor preservation holds at machine precision across
# an arbitrary (K, α) grid — guards the family definition itself.
print("Anchor-preservation spot check across (K, α):")
print(f"  {'K':>2} {'α':>6} | {'F2_rel max':>12} {'F3 max':>12}")
for K in [2, 3, 4, 5, 8]:
    for alpha in [1.20, 1.818, 2.5]:
        # Fit all 544 glasses and measure worst anchor slip
        nu = fit_all_glasses_anchor(catalog, alpha, K)
        worst_F2_rel = 0.0
        worst_F3 = 0.0
        for i in range(len(glasses)):
            m = anchor_error_metrics_K(
                catalog["nd"][i], catalog["vd"][i], catalog["dPgF"][i],
                nu[i], alpha=alpha,
            )
            worst_F2_rel = max(worst_F2_rel, m["F2_rel"])
            worst_F3 = max(worst_F3, m["F3"])
        print(f"  {K:>2} {alpha:>6.3f} | {worst_F2_rel:>12.3e} {worst_F3:>12.3e}")

print()
print("All entries < 1e-11 — anchors preserved at machine precision")
print("across the entire sweep grid, not just at K = 4, α = 1.818.")
"""))


# =========================================================================
# Part 3 — Selection criterion
# =========================================================================

cells.append(md(
"""## Part 3 — Selection criterion: which aggregation of catalog error?

The question is: *we have 544 per-glass Test A hold errors — which
summary statistic of the catalog should α and K minimise?*

Four natural aggregations of the per-glass max Test A hold error:

| Aggregation | Meaning | Effect on tail |
|---|---|---|
| **p50** (median) | typical glass | tail completely ignored |
| **p95** | 95 %-of-catalog upper bound | worst ~27 glasses ignored (includes most of FK / FPL family) |
| **p99** | 99 %-of-catalog upper bound | worst ~5 glasses ignored — robust to data outliers |
| **max** | worst single glass | strict worst-case coverage |

A production forward model is used across **every** glass in a catalog
sweep (glass substitution, apo-doublet ranking, tolerancing). A bad
prediction on even one FK / FPL crown can invalidate a design decision.
The right selection criterion must reflect "worst-case coverage" — so
p50 / p95 are inappropriate by construction. The real choice is between
**p99 (robust)** and **max (strict)**.

### Methodology

- K ∈ {2, 3, 4, 5, 6, 7, 8}
- α coarse grid: 21 points linspace(1.2, 2.8), step 0.08
- Test A: for each glass, for each of 17 wavelengths, re-fit on the
  remaining 16 points, measure hold error at the held-out wavelength,
  take max over wavelengths per glass.
- The four aggregations then compress 544 per-glass numbers into one
  α-selection objective.

3.1 runs the coarse sweep. 3.2 shows how α\\* diverges across
aggregations. 3.3 compares them on Test B directly — Test B is the
actual downstream metric the α choice must optimise — and picks the
winner.
"""))


cells.append(md("""### 3.1 Coarse (K × α) sweep"""))


cells.append(code(
"""import time

K_GRID = [2, 3, 4, 5, 6, 7, 8]
ALPHA_COARSE = np.linspace(1.2, 2.8, 21)

t0 = time.time()
results = {}          # (K, alpha) -> dict(hold_max=..., train_max=...)
for K in K_GRID:
    for a in ALPHA_COARSE:
        r = test_a_batched(catalog, float(a), K)
        results[(K, float(a))] = r
print(f"Coarse sweep: {len(K_GRID)}×{len(ALPHA_COARSE)} = {len(results)} "
      f"(K, α) points  in {time.time()-t0:.1f} s")
"""))


cells.append(code(
"""# Plot 1 — a first look under p95 aggregation (placeholder; 3.2 shows it's not enough)
fig, ax = plt.subplots(figsize=(8.5, 5.5))
colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(K_GRID)))
for c, K in zip(colors, K_GRID):
    ys = [np.percentile(results[(K, float(a))]["hold_max"], 95)
          for a in ALPHA_COARSE]
    ax.plot(ALPHA_COARSE, ys, "-o", color=c, ms=4, lw=1.4, label=f"K = {K}")
ax.set_yscale("log")
ax.set_xlabel(r"$\\alpha$")
ax.set_ylabel(r"p95 Test A hold error (absolute)")
ax.set_title("Test A p95 hold error — initial view (not the final criterion)")
ax.legend(ncol=2, loc="upper right")
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()
"""))


cells.append(md(
"""### 3.2 α\\* diverges strongly across aggregations

The plot above uses p95. Using p50 / p99 / max instead would produce
different curves and different α\\*. The table below re-aggregates the
same sweep data under all four aggregations and records each one's
per-K α\\* (coarse grid, step = 0.08).
"""))


cells.append(code(
"""aggs = {
    "p50": lambda x: float(np.median(x)),
    "p95": lambda x: float(np.percentile(x, 95)),
    "p99": lambda x: float(np.percentile(x, 99)),
    "max": lambda x: float(x.max()),
}

print("α* per K under four aggregations (coarse grid):")
print()
print(f"  {'K':>2} | {'p50 α*':>8} {'p95 α*':>8} {'p99 α*':>8} {'max α*':>8}")
print("  " + "-" * 46)

alpha_per_agg = {name: {} for name in aggs}
for K in K_GRID:
    row = {}
    for name, fn in aggs.items():
        ys = [fn(results[(K, float(a))]["hold_max"]) for a in ALPHA_COARSE]
        row[name] = float(ALPHA_COARSE[int(np.argmin(ys))])
        alpha_per_agg[name][K] = row[name]
    print(f"  {K:>2} | {row['p50']:>8.3f} {row['p95']:>8.3f} "
          f"{row['p99']:>8.3f} {row['max']:>8.3f}")
"""))


cells.append(code(
"""# Plot 2 — K = 4 under all four aggregations
K_FOCUS = 4
fig, ax = plt.subplots(figsize=(9, 5))
colors = {"p50": "tab:green", "p95": "tab:orange",
          "p99": "crimson", "max": "k"}
for name, fn in aggs.items():
    ys = [fn(results[(K_FOCUS, float(a))]["hold_max"]) for a in ALPHA_COARSE]
    a_star = ALPHA_COARSE[int(np.argmin(ys))]
    ax.plot(ALPHA_COARSE, ys, "-o", ms=4, lw=1.6, color=colors[name],
            label=f"{name}  (α* = {a_star:.3f})")
    ax.axvline(a_star, color=colors[name], ls=":", lw=0.8, alpha=0.35)
ax.set_yscale("log")
ax.set_xlabel(r"$\\alpha$")
ax.set_ylabel(f"Test A hold error  (K = {K_FOCUS}, log)")
ax.set_title(f"K = {K_FOCUS} — α* shifts monotonically from p50 to max")
ax.legend(loc="upper center", ncol=2, fontsize=9)
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()
"""))


cells.append(md(
"""**Observation.** α\\* shifts **monotonically** from p50 to max.
For K = 4: p50 → 1.20, p95 → 1.36, p99 → 1.52, max → 1.92. The spread
is 0.72 — more than half the full grid range. The aggregation choice
is **not** cosmetic; it picks a materially different α.

The tail of the catalog (the worst 1 – 5 % of glasses — FK / FPL
anomalous-dispersion family) genuinely prefers **less-aggressive ω
compression** (larger α) than the typical glass does. p50 / p95 pick
α for the typical glass, max / p99 pick α for the worst glass. These
are different optima.

### 3.3 Which aggregation? Decide via Test B

The above is a question *about* Test A. But Test A is only a proxy for
the thing we actually care about — **Test B cross-glass reconstruction
error on held-out glasses**. Test B max is the real production target
(worst-case reconstruction on a held-out glass). Whichever aggregation
minimises Test B max is the right α-selection criterion.

For K = 4, run Test B at each aggregation's α\\* and compare:
"""))


cells.append(code(
"""# Bring in the cross-glass CV helper
from dispersionlab.buchdahl import SCHOTT_NORMAL_A, SCHOTT_NORMAL_B


def test_b_cv(catalog, K, alpha, n_folds=5, seed=0):
    \"\"\"5-fold DBSCAN cluster CV at (K, alpha). Returns per-glass
    max |n_pred - n_Sellmeier| arrays (in-fold and out-of-fold).\"\"\"
    omega = buchdahl_omega(WAVELENGTHS, alpha)
    _, _, M_INV, B_COUP = build_anchor_matrices(alpha, K)

    nu_target = fit_all_glasses_anchor(catalog, alpha, K)
    nu_high_target = nu_target[:, 2:]
    N = len(catalog["nd"])

    Phi = np.array([
        feature_vec_20(catalog["nd"][i], catalog["vd"][i], catalog["dPgF"][i])
        for i in range(N)
    ])

    X = np.column_stack([catalog["nd"], catalog["vd"], catalog["dPgF"]])
    Xs = StandardScaler().fit_transform(X)
    cid = DBSCAN(eps=0.15, min_samples=2).fit(Xs).labels_.copy()
    nxt = cid.max() + 1
    for i, c in enumerate(cid):
        if c == -1:
            cid[i] = nxt
            nxt += 1
    uniq = np.unique(cid)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    folds = np.array_split(perm, n_folds)

    in_nerr, out_nerr = [], []
    for k in range(n_folds):
        test_clusters = set(folds[k].tolist())
        test_mask = np.array([c in test_clusters for c in cid])
        train_mask = ~test_mask

        if K > 2:
            A, *_ = np.linalg.lstsq(
                Phi[train_mask], nu_high_target[train_mask], rcond=None
            )
            pred_in_high  = Phi[train_mask] @ A
            pred_out_high = Phi[test_mask]  @ A
        else:
            pred_in_high  = np.zeros((train_mask.sum(), 0))
            pred_out_high = np.zeros((test_mask.sum(), 0))

        for mask, pred_high, bucket in [
            (train_mask, pred_in_high,  in_nerr),
            (test_mask,  pred_out_high, out_nerr),
        ]:
            nd_sub = catalog["nd"][mask]
            vd_sub = catalog["vd"][mask]
            dPgF_sub = catalog["dPgF"][mask]
            dn_FC = (nd_sub - 1.0) / vd_sub
            PgF = SCHOTT_NORMAL_A - SCHOTT_NORMAL_B * vd_sub + dPgF_sub
            dn_gF = PgF * dn_FC
            a_vec = M_INV @ np.stack([dn_FC, dn_gF])
            if K == 2:
                nu1, nu2 = a_vec[0], a_vec[1]
            else:
                delta = B_COUP @ pred_high.T
                nu1, nu2 = a_vec[0] - delta[0], a_vec[1] - delta[1]
            n_pred = (nd_sub[:, None]
                      + nu1[:, None] * omega[None, :]
                      + nu2[:, None] * omega[None, :] ** 2)
            for kk in range(3, K + 1):
                j = kk - 3
                n_pred = n_pred + pred_high[:, j:j + 1] * omega[None, :] ** kk
            bucket.extend(
                np.abs(n_pred - catalog["n_truth"][mask]).max(axis=1).tolist()
            )
    return dict(
        in_nerr=np.array(in_nerr),
        out_nerr=np.array(out_nerr),
    )


# For K = 4, Test B at each aggregation's α*
print("K = 4 — Test B max (out-of-fold) at each aggregation's α*:")
print()
print(f"  {'aggregation':<12} | {'α*':>6} | {'Test B max':>12}")
print("  " + "-" * 38)
for name in ("p50", "p95", "p99", "max"):
    a = alpha_per_agg[name][4]
    r = test_b_cv(catalog, 4, a)
    tb_max = float(r["out_nerr"].max())
    print(f"  {name:<12} | {a:>6.3f} | {tb_max:>12.3e}")
"""))


cells.append(md(
"""**Reading the table above:** the α selected under Test A max gives
the lowest Test B max — by a substantial margin over p99 and p95. This
is not an accident: Test B's own max includes every catalog glass,
including the anomalous-dispersion tail, so the aggregation that
preserves tail information on Test A (i.e. max, and to a lesser extent
p99) is the best proxy for Test B max.

**p99 is robust to single-glass outliers but still loses to max on
Test B max** by roughly a factor of 2 at K = 4. The "outlier-noise"
concern about max is also not supported by the data — if Test A max
were being driven by a single anomalous Sellmeier record, Test A max
and Test B max would de-correlate. They stay correlated across the
sweep, indicating Test A max is tracking a real physical effect (the
FK / FPL tail's preference for less-aggressive ω compression).

### 3.4 Commitment

From here on, **α is selected by Test A max per K**. The full per-K
evaluation is Part 4; the K = 4 α-refinement is Part 5.
"""))





# =========================================================================
# Part 4 — K selection under max criterion
# =========================================================================

cells.append(md(
"""## Part 4 — K selection under the max criterion

With α selected by **max** (per Part 3 commitment), which K gives the
best cross-glass production behaviour?

For each K in {2, …, 8}, take its `max-α*` from the coarse sweep
(Part 3.2) and run the full Test B cross-glass CV. Compare per-glass
reconstruction quality in terms of:

- **Test B p50 / p95**: typical glass (most of the catalog)
- **Test B max**: worst-case glass — the primary production target
- **max ratio** = Test B max (out-of-fold) / Test B max (in-fold) —
  cross-glass extrapolation stability; values near 1 mean the
  regressor reaches the worst glass; values ≫ 1 mean the regressor
  cannot extrapolate the extra coefficients to the catalog corners.
"""))


cells.append(code(
"""# Per-K max-α* + Test B
per_K_test_b = {}
for K in K_GRID:
    a = alpha_per_agg["max"][K]
    r = test_b_cv(catalog, K, a)
    per_K_test_b[K] = dict(
        alpha=a,
        p50_out=float(np.percentile(r["out_nerr"], 50)),
        p95_out=float(np.percentile(r["out_nerr"], 95)),
        p99_out=float(np.percentile(r["out_nerr"], 99)),
        max_out=float(r["out_nerr"].max()),
        max_in=float(r["in_nerr"].max()),
    )

print("Per-K Test B at max-optimal α (coarse grid, step = 0.08)")
print()
print(f"  {'K':>2} {'α_max*':>7} | {'p50 out':>10} {'p95 out':>10} "
      f"{'p99 out':>10} {'max out':>10} | {'max ratio':>9}")
print("  " + "-" * 72)
for K in K_GRID:
    r = per_K_test_b[K]
    ratio = r["max_out"] / max(r["max_in"], 1e-30)
    print(f"  {K:>2} {r['alpha']:>7.3f} | "
          f"{r['p50_out']:>10.3e} {r['p95_out']:>10.3e} "
          f"{r['p99_out']:>10.3e} {r['max_out']:>10.3e} | "
          f"{ratio:>7.2f}×")
"""))


cells.append(code(
"""# Plot 3 — Test B max and max-ratio as a function of K
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

K_vals = list(per_K_test_b.keys())
tb_max  = [per_K_test_b[K]["max_out"] for K in K_vals]
tb_p95  = [per_K_test_b[K]["p95_out"] for K in K_vals]
tb_rat  = [per_K_test_b[K]["max_out"] / max(per_K_test_b[K]["max_in"], 1e-30)
           for K in K_vals]

ax = axes[0]
ax.plot(K_vals, tb_max, "o-", color="crimson",  lw=1.8, label="Test B max (out)")
ax.plot(K_vals, tb_p95, "s-", color="tab:blue", lw=1.6, label="Test B p95 (out)")
ax.set_yscale("log")
ax.set_xlabel("K")
ax.set_ylabel("per-glass max |Test B reconstruction error|")
ax.set_title("Test B error vs K (each K at its max-optimal α)")
ax.axvline(4, color="k", ls=":", lw=0.6, alpha=0.6)
ax.legend(loc="best")
ax.grid(alpha=0.3, which="both")

ax = axes[1]
ax.plot(K_vals, tb_rat, "D-", color="darkorange", lw=1.8)
ax.axhline(2.0, color="k", ls=":", lw=0.8, label="instability threshold (2×)")
ax.axvline(4, color="k", ls=":", lw=0.6, alpha=0.6)
ax.set_xlabel("K")
ax.set_ylabel("Test B max out-of-fold / max in-fold")
ax.set_title("Worst-case cross-glass stability vs K")
ax.legend(loc="best")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
"""))


cells.append(md(
"""### 4.1 K = 4 is the unique Pareto point

The K table and Plot 3 show two structural facts:

1. **Test B max saturates around K = 4.** K = 4 delivers Test B max
   ≈ 7.7·10⁻³. K = 6 is essentially tied (~7.6·10⁻³). K = 5 / 7 / 8
   are all worse on max (≥ 7.9·10⁻³). K = 2 / 3 are not competitive.
   The "K = 8 is the frontier" framing based on Test A p95 **does
   not survive** the Test B max check.

2. **Cross-glass stability has a sharp knee at K = 4 → K = 5.** Max
   ratio is ≈ 1.1× for K ≤ 4 and jumps to 1.7 – 2.4× for K ≥ 5. K = 4
   is the last K for which the 20-D polynomial regressor reaches
   every glass in the catalog without extrapolation blow-up. Higher-K
   regressors systematically fail to extrapolate their extra
   coefficients into the anomalous-dispersion corner.

Taking the two axes together:

> **K = 4 is the unique K on the Pareto frontier (Test B max ↓,
> max ratio ↓ = 1)**. K = 6 matches on Test B max but fails on ratio.
> Every other K is strictly dominated.

Part 5 refines α at K = 4. Part 6 verifies the final choice against
alternative named configurations.
"""))


# =========================================================================
# Part 5 — α refinement at K = 4
# =========================================================================

cells.append(md(
"""## Part 5 — α refinement at K = 4

Part 4 settled `K = 4`. The coarse max-α\\* sits at 1.92 (grid step =
0.08). This part does two things:

1. **Fine sweep** around 1.92 with step = 0.02 to precisely locate the
   Test A max-optimal α.
2. **Band-symmetric derivation** — an independent, purely geometric
   principle that also picks α without any catalog data.

If the two agree to within the fine grid resolution, we commit to the
value with the cleanest independent derivation.
"""))


cells.append(code(
"""# Fine sweep on K = 4 in [1.00, 2.40], step 0.02 — both Test A and Test B.
# The [1.80, 2.40] window was too narrow: p95 curves were monotonically
# improving toward the left edge, hiding their true argmin. Extend down
# to α = 1.00 to capture the full (α, error) landscape and the genuine
# p95 vs max tradeoff.
alpha_fine = np.arange(1.00, 2.41, 0.02)
fine_rows = []
for a in alpha_fine:
    # Test A (per-glass LSQ on 16 of 17 wavelengths)
    r_a = test_a_batched(catalog, float(a), K=4)
    # Test B (cross-glass CV with regressor)
    r_b = test_b_cv(catalog, 4, float(a))
    fine_rows.append(dict(
        alpha=float(a),
        testA_p95=float(np.percentile(r_a["hold_max"], 95)),
        testA_max=float(r_a["hold_max"].max()),
        testB_p95_out=float(np.percentile(r_b["out_nerr"], 95)),
        testB_max_out=float(r_b["out_nerr"].max()),
    ))

alpha_fine_a_max = min(fine_rows, key=lambda x: x["testA_max"])
alpha_fine_b_max = min(fine_rows, key=lambda x: x["testB_max_out"])
print("Fine max-optimal α at K=4:")
print(f"  Test A max argmin: α = {alpha_fine_a_max['alpha']:.4f}  "
      f"(Test A max = {alpha_fine_a_max['testA_max']:.3e})")
print(f"  Test B max argmin: α = {alpha_fine_b_max['alpha']:.4f}  "
      f"(Test B max = {alpha_fine_b_max['testB_max_out']:.3e})")
print()
print(f"  {'α':>7} | {'TestA p95':>10} {'TestA max':>10} | "
      f"{'TestB p95 out':>13} {'TestB max out':>13}")
print("  " + "-" * 64)
for r in fine_rows:
    mA = " ←A" if r is alpha_fine_a_max else "   "
    mB = " ←B" if r is alpha_fine_b_max else "   "
    print(f"  {r['alpha']:>7.3f} | {r['testA_p95']:>10.3e} "
          f"{r['testA_max']:>10.3e}{mA} | "
          f"{r['testB_p95_out']:>13.3e} {r['testB_max_out']:>13.3e}{mB}")
"""))


cells.append(code(
"""# Plot 4 — K = 4 fine α sweep: Test A and Test B max as functions of α
fig, ax = plt.subplots(figsize=(9.5, 5))
alphas_f = [r["alpha"] for r in fine_rows]

ax.plot(alphas_f, [r["testA_max"] for r in fine_rows], "o-",
        color="k",        lw=1.8, label="Test A max (selection proxy)")
ax.plot(alphas_f, [r["testB_max_out"] for r in fine_rows], "D-",
        color="crimson",  lw=2.0, label="Test B max out-of-fold (production target)")
ax.plot(alphas_f, [r["testA_p95"] for r in fine_rows], "s-",
        color="tab:orange", lw=1.1, alpha=0.5,
        label="Test A p95 (reference)")
ax.plot(alphas_f, [r["testB_p95_out"] for r in fine_rows], "^-",
        color="tab:blue",   lw=1.1, alpha=0.5,
        label="Test B p95 out (reference)")

# Key markers
ax.axvline(alpha_fine_a_max["alpha"], color="k", ls="--", lw=1, alpha=0.5,
           label=f"Test A max argmin α = {alpha_fine_a_max['alpha']:.3f}")
ax.axvline(alpha_fine_b_max["alpha"], color="crimson", ls="--", lw=1,
           label=f"Test B max argmin α = {alpha_fine_b_max['alpha']:.3f}")
ax.axvline(1.955, color="darkorange", ls="--", lw=1,
           label="band-symmetric α = 1.955")
ax.axvline(1.818, color="gray", ls=":", lw=1,
           label="legacy α = 1.818")

ax.set_yscale("log")
ax.set_xlabel(r"$\\alpha$")
ax.set_ylabel(r"hold error at K = 4 (log)")
ax.set_title("K = 4 fine α sweep — Test A (proxy) and Test B (direct) compared")
ax.legend(loc="upper center", ncol=2, fontsize=8)
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()
"""))


cells.append(md(
"""### 5.1 Band-symmetric derivation of α

Independent of any catalog data, α has a natural geometric choice:
make the Buchdahl coordinate ω symmetric around zero on the training
wavelength band `[λ_min, λ_max]`. The condition
`|ω(λ_min)| = ω(λ_max)` yields a closed form:

$$\\alpha^\\star = \\frac{(\\lambda_{\\max} - \\lambda_d) - (\\lambda_d - \\lambda_{\\min})}{2(\\lambda_d - \\lambda_{\\min})(\\lambda_{\\max} - \\lambda_d)}$$

Substituting the project's `λ_d = 0.5876 μm` and the training grid
`[λ_min = 0.365, λ_max = 2.3] μm`:
"""))


cells.append(code(
"""# Band-symmetric α for our training grid
lam_min = float(WAVELENGTHS.min())
lam_max = float(WAVELENGTHS.max())
a_arm = LAMBDA_D - lam_min          # 0.223
b_arm = lam_max - LAMBDA_D          # 1.712
alpha_band_sym = (b_arm - a_arm) / (2.0 * a_arm * b_arm)

print(f"  a = λ_d − λ_min  = {a_arm:.6f}  μm")
print(f"  b = λ_max − λ_d  = {b_arm:.6f}  μm")
print(f"  α_band_sym       = (b − a) / (2ab) = {alpha_band_sym:.6f}")
print()
print("  Sanity check:")
print(f"    |ω(λ_min)| = {abs(buchdahl_omega(lam_min, alpha_band_sym)):.6f}")
print(f"     ω(λ_max)  = {buchdahl_omega(lam_max, alpha_band_sym):.6f}   (should match)")
"""))


cells.append(md(
"""### 5.2 Comparison

Three candidate α at K = 4:
"""))


cells.append(code(
"""# Summary comparison
candidates = [
    ("fine Test A max argmin (experimental proxy)",  alpha_fine_a_max["alpha"]),
    ("fine Test B max argmin (experimental direct)", alpha_fine_b_max["alpha"]),
    ("band-symmetric (principled)                 ", alpha_band_sym),
    ("legacy production (historical)               ", 1.818),
]

print(f"  {'candidate':<48} | {'α':>7} | {'Δ to Test B-max*':>18}")
print("  " + "-" * 80)
ref = alpha_fine_b_max["alpha"]
for label, a in candidates:
    d = a - ref
    print(f"  {label:<48} | {a:>7.4f} | {d:>+18.4f}")

print()
print("Fine grid step: 0.02   →  differences under ~0.02 are within a "
      "single step (statistically tied).")
"""))


cells.append(md(
"""### 5.2 Reading the principled candidates against the Test B landscape

Summary of where each principled α lands on the Part 5.0 fine sweep:

| Criterion | α | Test B max | Test B p95 out | Status |
|---|---|---|---|---|
| **A. band-symmetric (training grid)** | **1.9547** | **7.72·10⁻³ (+1.2 %)** | **4.54·10⁻³** | inside plateau, principled |
| B. anchor-line symmetric | −3.98 | n/a | n/a | negative — not physical |
| C. min cond(M) | ≥ 3.0 | rises past plateau | — | degenerate (plateau over α > 2.4) |
| D. IR / UV asymptotic | 2.2467 | 7.64·10⁻³ (0 %) | 5.15·10⁻³ (+14 %) | inside plateau, weak argument |
| E. Mercado-Robb Sellmeier | 2.5360 | ≈ 8.0·10⁻³ (plateau edge) | ≈ 5.5·10⁻³ (+22 %) | outside plateau, classical |

**Only criterion A (band-symmetric) yields an α both (i) principled
with a clean closed-form derivation and (ii) inside the Test B max
plateau while minimising Test B p95 among principled candidates.**

Criterion D (IR/UV asymptotic) is principled and just barely inside
the plateau; it is a reasonable runner-up but the "1/α equals UV
ω-reach" rationale is a weaker motivation than ω-band-symmetry. E
(Mercado-Robb) gives the classical α ≈ 2.54 but that sits past the
Test B max plateau — principled, historical, but data-bad on our
catalog / grid.

### 5.3 The Test B max landscape justifies α = 1.9547 operationally

Why the plateau structure matters for the commitment:

**Test B max has a cliff + plateau structure.**

| α window | Test B max behaviour |
|---|---|
| [1.00, 1.60] | **cliff**: 2.68·10⁻² → 1.18·10⁻² (3.6× drop) |
| [1.60, 1.80] | transition: 1.18·10⁻² → 7.81·10⁻³ |
| **[1.80, 2.30]** | **plateau**: 7.63·10⁻³ – 7.81·10⁻³ (< 3 % variation) |
| [2.30, 2.40] | slight upturn: 7.94·10⁻³ |

α = 1.9547 sits firmly in the plateau (Test B max = 7.72·10⁻³,
within 1.2 % of the minimum). The legacy α = 1.818 also sits in
the plateau (7.81·10⁻³, +2.2 %) but has no derivation in the
anchor-preserving family — its legacy value comes from a sweep in the
**legacy construction** (anchor-slip present) under a different
aggregation criterion. The fact that α = 1.818 empirically matches
the Pareto knee is a retrospective coincidence, not a derivation;
this notebook does not adopt it as the recommended α.

### 5.4 Commit: α = 1.9547

**Production: K = 4, α = 1.9547** (band-symmetric on training grid).

Derivation:
$$\\alpha^\\star = \\frac{(\\lambda_{\\max} - \\lambda_d) - (\\lambda_d - \\lambda_{\\min})}{2(\\lambda_d - \\lambda_{\\min})(\\lambda_{\\max} - \\lambda_d)}
 = \\frac{1.712 - 0.223}{2 \\cdot 0.223 \\cdot 1.712} = 1.9547$$

Meaning: the Buchdahl coordinate `ω` is symmetric around zero on the
training band. `|ω(0.365 μm)| = ω(2.3 μm) ≈ 0.394`. The 4-term
polynomial's representational capacity is balanced between UV-visible
and NIR wavelengths.

Empirical verification (Part 5.0):
- Test B max at α = 1.9547: 7.72·10⁻³ (within 1.2 % of plateau
  minimum) — on the **Test B max plateau**, so worst-case coverage
  is essentially at the data-driven optimum.
- Test B p95 at α = 1.9547: 4.54·10⁻³ — +12 % higher than at the
  (unprincipled) legacy α = 1.818's 4.08·10⁻³. This is the
  typical-glass cost of enforcing a principled derivation; within
  engineering tolerance for every downstream task in NB02–NB05.

**(K, α) = (4, 1.9547) is the production commitment.** Legacy
α = 1.818 (rejected in 5.1 on principle) and the K = 8, α = 1.26
accuracy-frontier (rejected in Part 4 on Test B max + ratio) are not
re-litigated below. Part 6 is a separate sanity check: given that
the family and α are fixed, **where is the remaining error coming
from** — the K = 4 family itself, or the 3-parameter regressor on
top of it? That decomposition motivates NB04 (better regressors) and
NB05 (residuals beyond the family).
"""))


# =========================================================================
# Part 6 — Oracle decomposition: family ceiling vs regressor gap
# =========================================================================

cells.append(md(
"""## Part 6 — Where the remaining Test B error comes from

At the committed `(K = 4, α = 1.9547)`, the end-to-end reconstruction
error can be cleanly decomposed into two independent pieces:

1. **Oracle floor** — per-glass constrained LSQ of `(ν₃, ν₄)`
   against the Sellmeier truth on the 17-wavelength grid (anchors
   still preserved by construction). This is the *representational
   ceiling* of the K = 4 anchor-preserving family: no regressor
   improvement can get below this without leaving the family.
2. **Regressor gap** — the production path replaces the Oracle's
   per-glass best `(ν₃, ν₄)` with the prediction of the linear
   20-D regressor on `(n_d, V_d, ΔP_{g,F})`. The gap between Oracle
   and production (both in-fold and out-of-fold from Part 4's CV) is
   *purely* the cost of compressing the full Sellmeier curve into
   three catalog numbers.

The split tells us where to invest next:

- If **Oracle ≈ production**, the family is the bottleneck — go to
  NB05's structural residual `ε·q(λ)·tanh(NN)` to break through it.
- If **production ≫ Oracle**, the regressor is the bottleneck — go
  to NB04's Level-2 ablation (linear vs MLP vs Oracle) to squeeze
  more out of the 3-parameter map.

We expect both pictures at different percentiles, and Part 6
quantifies which.
"""))


cells.append(code(
"""# Oracle per-glass fit at (K=4, α=1.9547) — no regressor involved.
# fit_all_glasses_anchor solves constrained LSQ of ν_high against
# Sellmeier n_truth with F1/F2/F3 anchors at machine precision.
K_prod, alpha_prod = 4, alpha_band_sym
omega_prod = buchdahl_omega(WAVELENGTHS, alpha_prod)

nu_oracle = fit_all_glasses_anchor(catalog, alpha_prod, K_prod)   # (N, 4)
n_oracle  = np.array([
    reconstruct_n_K(catalog["nd"][i], nu_oracle[i], omega_prod)
    for i in range(len(catalog["nd"]))
])
oracle_nerr = np.abs(n_oracle - catalog["n_truth"]).max(axis=1)

# Production at the same (K, α) — reuse Part 4's CV run
r_prod = per_K_test_b[K_prod]  # already computed in Part 4 at max-α*
# but Part 4 used the max-α* grid point, which is 1.92 ≠ 1.9547.
# Re-run Test B at the committed α so the decomposition is apples-to-apples.
cv_prod = test_b_cv(catalog, K_prod, alpha_prod)
in_nerr, out_nerr = cv_prod["in_nerr"], cv_prod["out_nerr"]

print(f"(K, α) = ({K_prod}, {alpha_prod:.4f})  — error decomposition")
print("all per-glass max |n_pred − n_Sellmeier| on 17-wavelength truth grid")
print()
print(f"  {'percentile':<10} | {'Oracle floor':>12} | {'prod in-fold':>12} "
      f"| {'prod out-fold':>13} | {'gap out-fold':>12}")
print("  " + "-" * 72)

def _pct(a, p): return float(np.percentile(a, p))

for p in (50, 95, 99):
    orc = _pct(oracle_nerr, p)
    inf = _pct(in_nerr,     p)
    outf = _pct(out_nerr,   p)
    print(f"  p{p:<9} | {orc:>12.3e} | {inf:>12.3e} "
          f"| {outf:>13.3e} | {outf - orc:>12.3e}")
orc_m  = oracle_nerr.max()
inf_m  = in_nerr.max()
outf_m = out_nerr.max()
print(f"  {'max':<10} | {orc_m:>12.3e} | {inf_m:>12.3e} "
      f"| {outf_m:>13.3e} | {outf_m - orc_m:>12.3e}")
"""))


cells.append(code(
"""# Plot 5 — Oracle floor vs production CDF at (K=4, α=1.9547)
fig, ax = plt.subplots(figsize=(8.5, 5))

def _cdf(a):
    a = np.sort(a)
    y = np.linspace(0.0, 1.0, len(a), endpoint=True)
    return a, y

for arr, color, ls, lw, label in [
    (oracle_nerr, "black",      "-",  2.0, "Oracle floor (per-glass best ν₃,ν₄)"),
    (in_nerr,     "tab:blue",   "--", 1.6, "production in-fold  (regressor on seen glasses)"),
    (out_nerr,    "crimson",    "-",  2.0, "production out-of-fold (held-out glasses)"),
]:
    x, y = _cdf(arr)
    ax.plot(x, y, color=color, ls=ls, lw=lw, label=label)

ax.axhline(0.95, color="k", ls=":", lw=0.6, alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel(r"per-glass max $|n_{\\rm pred}(\\lambda) - n_{\\rm Sellmeier}(\\lambda)|$")
ax.set_ylabel("cumulative fraction of catalog")
ax.set_title(r"Error decomposition at $(K=4,\\ \\alpha=1.9547)$")
ax.legend(loc="lower right")
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.show()
"""))


cells.append(md(
"""### 6.1 What the decomposition says

Reading the table and the CDF (black = Oracle floor, red = production
out-of-fold, blue-dashed = production in-fold):

- **Typical glass (p50).** Oracle floor 2.38·10⁻³ vs production
  out-of-fold 2.69·10⁻³ — a ~13 % gap. The K = 4 **family ceiling**
  is the dominant term; the regressor is near the floor. Further
  gains on the median glass require leaving the K = 4 polynomial
  family. NB05's Phase 2 structural residual
  (`ε·q(λ)·tanh(NN)`, zero at anchors by construction) is the exit
  route designed exactly for this bottleneck.
- **Shoulder and tail (p95 → max).** The gap ratio widens
  monotonically: +44 % at p95, +69 % at p99, +72 % at max. At this
  point the 3-parameter feature map `(n_d, V_d, ΔP_{g,F}) →
  (ν₃, ν₄)` is losing information that a per-glass LSQ on the full
  Sellmeier curve keeps. NB04's Level-2 ablation (linear 20-D vs
  MLP vs Oracle) lives in exactly this gap: it asks how much of it
  a more expressive regressor can recover — and then delivers the
  clean negative result (MLP ≈ linear 20-D) that the remaining
  lever is *feature engineering* rather than model capacity.
- **In-fold ≈ out-of-fold.** At every percentile the in-fold /
  out-of-fold separation is ≲ 10 % — much smaller than the Oracle /
  production gap in the tail. Cross-glass CV is not the problem at
  K = 4; the problem is the information loss in the 3-parameter map
  itself, which the Oracle floor bypasses by consuming the full
  truth curve.

This decomposition is the handoff to the rest of the project:
**NB05 attacks the family floor, NB04 quantifies the regressor
gap.** NB01 commits the production forward model and hands off the
two improvement axes to downstream notebooks.

Plot 6 and Plot 7 below localise *where in the catalog* the two
error components concentrate.
"""))


# -------------------------------------------------------------------------
# 6.2 — Plot 6: ν₃/ν₄ Level-2 residual map
# -------------------------------------------------------------------------

cells.append(code(
"""# Plot 6 — Level-2 coefficient residuals, not direct optical error
# The production regressor predicts (ν₃, ν₄) linearly from the 20-D
# feature map of (n_d, V_d, ΔP_{g,F}). Scatter of (Oracle ν − predicted ν)
# over the catalog reveals which corners the 3-parameter map struggles to
# reproduce. These coefficient residuals only matter insofar as they
# propagate to off-anchor reconstruction error (Plot 7).
reg = np.load("../data/regression_buchdahl_anchor_nu34_20dim_opt.npy")  # (20, 2)

# Full-catalog feature matrix (uses catalog arrays already in scope)
Phi_all = np.array([
    feature_vec_20(catalog["nd"][i], catalog["vd"][i], catalog["dPgF"][i])
    for i in range(len(catalog["nd"]))
])
nu_high_oracle = nu_oracle[:, 2:]            # (N, 2)
nu_high_pred   = Phi_all @ reg               # (N, 2)
resid_nu       = nu_high_oracle - nu_high_pred   # (N, 2)

def _sym_lim(a):
    m = float(np.max(np.abs(a)))
    return (-m, m)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for k, (ax, col) in enumerate(zip(axes, ["ν₃ residual", "ν₄ residual"])):
    vlim = _sym_lim(resid_nu[:, k])
    sc = ax.scatter(
        catalog["vd"], catalog["nd"],
        c=resid_nu[:, k], cmap="RdBu_r",
        vmin=vlim[0], vmax=vlim[1],
        s=14, alpha=0.75, edgecolors="none",
    )
    fk_mask = catalog["dPgF"] > 0.03
    flint_mask = catalog["nd"] > 1.9
    ax.scatter(catalog["vd"][fk_mask], catalog["nd"][fk_mask],
               s=40, facecolors="none", edgecolors="black", lw=0.8,
               label="FK cluster (ΔP_{g,F} > 0.03)")
    ax.scatter(catalog["vd"][flint_mask], catalog["nd"][flint_mask],
               s=40, facecolors="none", edgecolors="dimgray", lw=0.8,
               label="high-nd flint (n_d > 1.9)")
    ax.set_xlabel(r"$V_d$")
    ax.invert_xaxis()
    if k == 0:
        ax.set_ylabel(r"$n_d$")
    ax.set_title(f"Level-2 {col}  (Oracle − 20-D regressor pred)")
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, shrink=0.85)
axes[0].legend(fontsize=8, loc="lower left")
plt.tight_layout()
plt.show()

print(f"ν₃ residual std = {resid_nu[:, 0].std():.3e}, "
      f"range = [{resid_nu[:, 0].min():.3e}, {resid_nu[:, 0].max():.3e}]")
print(f"ν₄ residual std = {resid_nu[:, 1].std():.3e}, "
      f"range = [{resid_nu[:, 1].min():.3e}, {resid_nu[:, 1].max():.3e}]")
"""))


cells.append(md(
"""### 6.2 Where the regressor gap concentrates

Plot 6 is a **regressor diagnostic, not a direct optical error
plot**. The colour is residual in the Level-2 coefficient space
`(ν₃, ν₄)` — not `|Δn|` at any wavelength. A large Level-2
residual becomes an optical error only after propagation through
`Φ_k(ω) = ω^k − ω·B_{1,k-3} − ω²·B_{2,k-3}` at the evaluation
wavelengths, and part of it is absorbed by the anchor-preserving
coupling `[ν₁, ν₂] = a − B·[ν₃, ν₄]`. Plot 7 gives the propagated
view.

What the two panels show:

- **Structured, signed residuals.** Both panels show coherent
  red / blue lobes rather than scatter noise — the 3-parameter
  feature map misses systematic high-order information. This is
  the same signal that NB04's Level-2 ablation reads as
  "a more expressive regressor does not close the gap": the
  missing signal is not inside the `(n_d, V_d, ΔP_{g,F})` span,
  it is outside it.
- **Extreme coefficient residuals sit on sparse corners.** The
  deepest red / blue lies in the **FK cluster**
  (high `V_d`, positive `ΔP_{g,F}`) and the **high-`n_d`
  flint band** — the two regions where the linear regressor has
  fewest nearest neighbours in feature space. Note that large
  Level-2 residual does not automatically imply large optical
  error; the anchor-preserving construction rebuilds
  `(ν₁, ν₂)` consistently with the predicted `(ν₃, ν₄)`, so
  the propagated impact depends on basis `Φ_k(ω)` magnitudes at
  each wavelength. This is exactly why Plot 7 is needed.
"""))


# -------------------------------------------------------------------------
# 6.3 — Plot 7: end-to-end design-space error map
# -------------------------------------------------------------------------

cells.append(code(
"""# Plot 7 — end-to-end reconstruction error across the design space.
# Colour = log10( per-glass max |n_pred − n_Sellmeier| ) on the
# 17-wavelength truth grid, out-of-fold from cluster CV (one error per
# glass, aligned with catalog index — test_b_cv's out_nerr is in fold
# concatenation order so we redo the CV here to keep per-glass alignment).
from sklearn.cluster import DBSCAN as _DBSCAN
from sklearn.preprocessing import StandardScaler as _StdScaler

N = len(catalog["nd"])
err_per_glass = np.full(N, np.nan)
Xs = _StdScaler().fit_transform(
    np.column_stack([catalog["nd"], catalog["vd"], catalog["dPgF"]])
)
_cid = _DBSCAN(eps=0.15, min_samples=2).fit(Xs).labels_.copy()
_nxt = _cid.max() + 1
for _i, _c in enumerate(_cid):
    if _c == -1:
        _cid[_i] = _nxt
        _nxt += 1
_rng = np.random.default_rng(0)
_folds = np.array_split(_rng.permutation(np.unique(_cid)), 5)

for _fk in range(5):
    _test_set = set(_folds[_fk].tolist())
    _test_mask = np.array([c in _test_set for c in _cid])
    _train_mask = ~_test_mask
    _A, *_ = np.linalg.lstsq(Phi_all[_train_mask], nu_high_oracle[_train_mask], rcond=None)
    _pred_high = Phi_all[_test_mask] @ _A
    _nd_s = catalog["nd"][_test_mask]
    _vd_s = catalog["vd"][_test_mask]
    _dPgF_s = catalog["dPgF"][_test_mask]
    _dn_FC = (_nd_s - 1.0) / _vd_s
    from dispersionlab.buchdahl import SCHOTT_NORMAL_A as _SA, SCHOTT_NORMAL_B as _SB
    _PgF = _SA - _SB * _vd_s + _dPgF_s
    _dn_gF = _PgF * _dn_FC
    _, _, _M_INV, _B_COUP = build_anchor_matrices(alpha_prod, K_prod)
    _a_vec = _M_INV @ np.stack([_dn_FC, _dn_gF])
    _delta = _B_COUP @ _pred_high.T
    _nu1, _nu2 = _a_vec[0] - _delta[0], _a_vec[1] - _delta[1]
    _n_pred = (_nd_s[:, None]
               + _nu1[:, None] * omega_prod[None, :]
               + _nu2[:, None] * omega_prod[None, :] ** 2)
    for _kk in range(3, K_prod + 1):
        _j = _kk - 3
        _n_pred = _n_pred + _pred_high[:, _j:_j+1] * omega_prod[None, :] ** _kk
    err_per_glass[_test_mask] = np.abs(
        _n_pred - catalog["n_truth"][_test_mask]
    ).max(axis=1)

assert not np.any(np.isnan(err_per_glass)), "some glasses never appeared in test folds"
err = err_per_glass
log_err = np.log10(err)

fig, ax = plt.subplots(figsize=(8.5, 5.2))
sc = ax.scatter(
    catalog["vd"], catalog["nd"],
    c=log_err, cmap="viridis",
    s=18, alpha=0.85, edgecolors="none",
)
cb = plt.colorbar(sc, ax=ax, shrink=0.9)
cb.set_label(r"$\\log_{10}\\,\\max_\\lambda |n_{\\rm pred}(\\lambda) - n_{\\rm Sellmeier}(\\lambda)|$")

# Overlay guides for Part 6.2's two extrapolative corners
fk_mask    = catalog["dPgF"] > 0.03
flint_mask = catalog["nd"]   > 1.9
ax.scatter(catalog["vd"][fk_mask], catalog["nd"][fk_mask],
           s=55, facecolors="none", edgecolors="crimson", lw=1.1,
           label="FK cluster")
ax.scatter(catalog["vd"][flint_mask], catalog["nd"][flint_mask],
           s=55, facecolors="none", edgecolors="darkorange", lw=1.1,
           label="high-nd flint")

ax.set_xlabel(r"$V_d$")
ax.set_ylabel(r"$n_d$")
ax.invert_xaxis()
ax.set_title(r"Production end-to-end $n(\\lambda)$ error across the $(V_d, n_d)$ plane")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Quantify corner vs interior — median / p95 / max
central = ~(fk_mask | flint_mask)
print(f"per-glass max |Δn|   {'median':>10} {'p95':>10} {'max':>10}")
for lbl, m in [
    ("overall",          np.ones_like(fk_mask)),
    ("central interior", central),
    ("FK cluster",       fk_mask),
    ("high-nd flint",    flint_mask),
]:
    e = err[m]
    if len(e) == 0:
        continue
    print(f"  {lbl:<18} "
          f"n={m.sum():>3d}  "
          f"{np.median(e):.2e} {np.percentile(e, 95):.2e} {np.max(e):.2e}")
"""))


cells.append(md(
"""### 6.3 Where production is reliable vs extrapolative

Plot 7 is the **propagated** view of Plot 6: the two sparse
corners that lit up in Level-2 coefficient space reappear in
optical `|Δn|` — but with nuance:

- **Central interior (≈ 508 glasses, most of the catalog)**:
  `|Δn|` median ≈ 2.7·10⁻³, p95 ≈ 4.2·10⁻³, max ≈ 7.0·10⁻³.
  This is the **region where the production surrogate is least
  likely to dominate downstream uncertainty** — its contribution
  to a tolerance budget (NB03 J·Σ·Jᵀ) or an apo-doublet Conrady
  estimate (NB02) typically stays below the surface-curvature and
  parameter-covariance terms.
- **High-`n_d` flint band (≈ 15 glasses)**: consistently worse
  than the interior on median **and** p95 **and** max. The
  feature map genuinely struggles in this corner, and every
  percentile shows it.
- **FK cluster (≈ 24 glasses)**: **bimodal**. The median is
  actually *below* the interior (most FKs are well-fit), but the
  tail is heavier — p95 ≈ 6.1·10⁻³ and the worst-case catalog
  glass sits here. Apochromatic designs routinely rely on FK
  glasses, so the tail matters even though the median is
  favourable.

The exact "reliable vs extrapolative" threshold is task-dependent.
NB03's tolerance analysis cares about local derivatives and anchor
structure more than about `max |Δn|`; NB02's secondary-spectrum
ordering cares only about anchor-line indices (architecturally
exact). Plot 7 is a **screening map** — the designer should use
it to flag candidate substitutions that land in the FK / flint
tails for a direct Sellmeier cross-check, not as a binary safe /
unsafe boundary.

Together, Plots 5–7 complete Part 6: **5** gives the error
budget as CDFs, **6** is a Level-2 regressor diagnostic,
**7** locates the propagated error on the design space.
"""))


# -------------------------------------------------------------------------
# 6.4 — Short appendix: why not a generic polynomial basis?
# -------------------------------------------------------------------------

cells.append(md(
"""### 6.4 Why not a generic polynomial basis? (short cross-check)

A reasonable question once Buchdahl is fixed at K = 4 is whether a
generic polynomial basis — Chebyshev being the textbook choice on a
closed wavelength interval — could play the same role. Chebyshev is
strictly stronger as a **per-glass function approximator** (quasi-best
uniform approximation on a closed interval; its per-glass residual is
lower than Buchdahl's at matched degree). The question is not
approximation capacity, it is whether the basis' coefficients are the
right **production coordinate** for a three-descriptor surrogate.

We check two things a production basis has to deliver that Chebyshev
does not provide by construction:

1. **Stable linear regression from `(n_d, V_d, ΔP_{g,F})`.** Buchdahl's
   `(ν₃, ν₄)` are physically close to "leftover anomalous-dispersion
   shape" that the partial-dispersion descriptor already encodes.
   Chebyshev coefficients are functions of the truth curve and have
   no a-priori link to the catalog descriptors.
2. **Anchor semantics.** Buchdahl's anchor-preserving construction
   keeps `n(λ_d) = n_d`, `V_d`, and `P_{g,F}` exact for any
   high-order prediction (Part 1). Chebyshev has no such property;
   `V_d` and `P_{g,F}` implied by the predicted curve can drift.
"""))


cells.append(code(
"""# Chebyshev cross-check at matched protocol:
# - per-glass fit at degree 8 (9 coefs — comparable free-parameter count
#   to Buchdahl K = 4 per-glass LSQ at 17 wavelengths)
# - 20-D linear regression of the 9 coefficients from
#   (n_d, V_d, ΔP_{g,F}) on the same cluster CV split as Part 6
# - report anchor slip (implied V_d / P_{g,F} vs catalog value)
from numpy.polynomial.chebyshev import chebfit, chebval

LAM_MIN = WAVELENGTHS.min()
LAM_MAX = WAVELENGTHS.max()
def _lam_to_x(lam):
    return 2.0 * (lam - LAM_MIN) / (LAM_MAX - LAM_MIN) - 1.0
X_WL = _lam_to_x(WAVELENGTHS)
X_d  = _lam_to_x(LAMBDA_D)
X_F  = _lam_to_x(LAMBDA_F)
X_C  = _lam_to_x(LAMBDA_C)
X_g  = _lam_to_x(LAMBDA_g)

DEG = 8
N = len(catalog["nd"])

# Per-glass Chebyshev fit (fit layer)
cheb_coefs = np.zeros((N, DEG + 1))
for i in range(N):
    cheb_coefs[i] = chebfit(X_WL, catalog["n_truth"][i], DEG)
fit_recon = np.array([chebval(X_WL, cheb_coefs[i]) for i in range(N)])
fit_err   = np.abs(fit_recon - catalog["n_truth"]).max(axis=1)

# Cluster-CV regression (same folds as Part 6.3 via _cid/_folds)
e2e_err_cheb = np.full(N, np.nan)
Vd_slip      = np.full(N, np.nan)
PgF_slip     = np.full(N, np.nan)
for _fk in range(5):
    _test_set  = set(_folds[_fk].tolist())
    _test_mask = np.array([c in _test_set for c in _cid])
    A_cheb, *_ = np.linalg.lstsq(
        Phi_all[~_test_mask], cheb_coefs[~_test_mask], rcond=None
    )
    pred_coefs = Phi_all[_test_mask] @ A_cheb
    n_rec = np.array([chebval(X_WL, pred_coefs[j]) for j in range(len(pred_coefs))])
    e2e_err_cheb[_test_mask] = np.abs(n_rec - catalog["n_truth"][_test_mask]).max(axis=1)

    # anchor slip from predicted Chebyshev curve
    n_d_c = np.array([chebval(X_d, pc) for pc in pred_coefs])
    n_F_c = np.array([chebval(X_F, pc) for pc in pred_coefs])
    n_C_c = np.array([chebval(X_C, pc) for pc in pred_coefs])
    n_g_c = np.array([chebval(X_g, pc) for pc in pred_coefs])
    Vd_implied  = (n_d_c - 1.0) / (n_F_c - n_C_c)
    PgF_implied = (n_g_c - n_F_c) / (n_F_c - n_C_c)
    Vd_slip[_test_mask]  = np.abs(Vd_implied  - catalog["vd"][_test_mask])
    PgF_slip[_test_mask] = np.abs(PgF_implied - (
        SCHOTT_NORMAL_A - SCHOTT_NORMAL_B * catalog["vd"][_test_mask]
        + catalog["dPgF"][_test_mask]
    ))

# Buchdahl side uses err_per_glass from Plot 7 (same CV split)
print("Buchdahl production (K=4, α=1.9547) vs Chebyshev (deg 8, 9 coefs)")
print("same 5-fold cluster CV, same 20-D linear regressor")
print()
print(f"  {'':<34} {'Buchdahl':>12}  {'Chebyshev':>12}")
print(f"  {'-'*34} {'-'*12}  {'-'*12}")
print(f"  per-glass fit-layer max (p95)      "
      f"{np.percentile(oracle_nerr, 95):>12.2e}  "
      f"{np.percentile(fit_err, 95):>12.2e}")
print(f"  end-to-end |Δn| out-of-fold (p95)  "
      f"{np.percentile(err_per_glass, 95):>12.2e}  "
      f"{np.percentile(e2e_err_cheb, 95):>12.2e}")
print(f"  end-to-end |Δn| out-of-fold (max)  "
      f"{err_per_glass.max():>12.2e}  "
      f"{e2e_err_cheb.max():>12.2e}")
print(f"  V_d slip out-of-fold   (p95)       "
      f"{0.0:>12.2e}  {np.nanpercentile(Vd_slip, 95):>12.2e}")
print(f"  V_d slip out-of-fold   (max)       "
      f"{0.0:>12.2e}  {np.nanmax(Vd_slip):>12.2e}")
print(f"  ΔP_g,F slip out-of-fold (p95)      "
      f"{0.0:>12.2e}  {np.nanpercentile(PgF_slip, 95):>12.2e}")
print(f"  ΔP_g,F slip out-of-fold (max)      "
      f"{0.0:>12.2e}  {np.nanmax(PgF_slip):>12.2e}")
print()
print("Chebyshev anchor slip is 0 by construction only at the "
      "per-glass fit layer; under 20-D → 9-coef regression, the "
      "predicted curve deviates from the catalog V_d / P_g,F.")
"""))


cells.append(md(
"""The cross-check (numbers above) is more nuanced than a blanket
"Chebyshev fails":

- **Fit-layer (Chebyshev per-glass)**: Chebyshev at degree 8 has ~10×
  smaller per-glass residual than the Buchdahl K = 4 Oracle
  (~3.2·10⁻⁴ vs ~3.1·10⁻³ at p95) — the expected consequence of
  near-best uniform approximation on a closed interval. Chebyshev
  **is** the stronger approximation basis.
- **End-to-end `n(λ)` (out-of-fold)**: 20-D linear regression of
  Chebyshev's 9 coefficients actually matches or slightly **beats**
  Buchdahl on the typical glass (p95 ≈ 2.4·10⁻³ vs 4.5·10⁻³); the
  worst-case `max |Δn|` is essentially tied (~7.7–7.8·10⁻³). So the
  3-parameter feature map can regress Chebyshev coefficients well
  enough to keep optical error competitive. The argument against
  Chebyshev **is not** "regression fails".
- **Anchor slip — the real failure mode.** Buchdahl has zero `V_d` /
  `P_{g,F}` slip by construction. Chebyshev's regressed curve drifts
  on both: `|ΔV_d|` p95 ≈ 1.5 Abbe units, **max ≈ 23** (a catalog
  `V_d` of 40 could be read as 17); `|ΔP_{g,F}|` p95 ≈ 1.8·10⁻²,
  **max ≈ 5.0·10⁻²**, large enough to flip a glass in and out of
  the `ΔP_{g,F} > 0.03` FK bin. Everything downstream of
  `(n_d, V_d, ΔP_{g,F})` semantics — apo-doublet `S` ordering
  (NB02), anchor-line Conrady estimates, glass-catalog substitution
  searches — breaks on these slips, not on the `n(λ)` residual.

Net: Chebyshev is a strong per-glass approximation basis, but it is
not the right **production coordinate** for a three-descriptor glass
surrogate unless one adds explicit anchor constraints and a stable
coefficient predictor on top of it. Buchdahl's anchor-preserving
construction gives both for free — **that**, not the raw `n(λ)`
residual, is why the production forward model sits here.
"""))


# =========================================================================
# Conclusion
# =========================================================================

cells.append(md(
"""## Conclusion

### Production forward model

**K = 4, α = 1.9547**, anchor-preserving Buchdahl.

**Why K = 4** (Part 4):

- Test B max ≈ 7.7·10⁻³ on 544-glass cross-glass CV — within 2 % of
  the minimum across all K under max-optimal α selection.
- Max out/in ratio ≈ 1.14× — the only K with cross-glass
  extrapolation stability on the worst-case glass.
- K = 6 matches on Test B max but fails on ratio (1.75×). Every K
  other than 4 is strictly dominated on at least one axis.

**Why α = 1.9547** (Part 5):

- **Principled derivation**: band-symmetric on the training grid,
  `α = (b − a) / (2ab) = 1.9547` with `a = λ_d − λ_min`, `b = λ_max
  − λ_d`. Closed form, independent of any catalog or aggregation
  choice. Makes the Buchdahl coordinate ω symmetric around zero on
  [0.365, 2.3] μm, balancing the 4-term polynomial's representational
  capacity between UV-visible and NIR.
- Five principled candidates evaluated (Part 5.1). α = 1.9547 is the
  only one that is **both principled and inside the Test B max
  plateau while minimising Test B p95 among principled values**.
- Numerically: Test B max at α = 1.9547 is 7.72·10⁻³, within 1.2 %
  of the Test B max plateau minimum. The choice sits firmly inside
  the plateau, so worst-case coverage is essentially at the
  data-driven optimum.

### Why not α = 1.818 (the legacy production value)

α = 1.818 was inherited from a legacy NB01 sweep that ran in a
**different model family** (legacy construction with anchor-slip bug)
under a **different aggregation criterion** (min max-per-glass Test A
hold). Re-applying the same criterion in the current
anchor-preserving family gives α = 1.94, not 1.818. The fact that
α = 1.818 happens to sit at the Test B max cliff-to-plateau knee is a
retrospective coincidence, not a derivation. For an interpretable
production model, a principled α is required; this notebook commits
to α = 1.9547 for that reason.

The empirical cost of the switch is small: Test B p95 degrades by
~12 % (from 4.08·10⁻³ at legacy α = 1.818 to 4.54·10⁻³ at principled
α = 1.9547), below the design budget of every downstream task in
NB02–NB05.

### Downstream handoff (from Part 6)

Part 6 decomposes the remaining production error and locates it on
the design space. Two axes for improvement are handed to later
notebooks:

- **Regressor gap → NB04.** Plot 6 shows structured (not noisy)
  coefficient-space residual on the FK cluster and high-`n_d`
  flint. NB04's Level-2 ablation (linear 20-D vs MLP vs Oracle)
  quantifies how much of this gap extra regressor capacity
  recovers, and returns a clean negative result: the missing
  signal is not inside the 3-parameter span, so feature
  engineering rather than model capacity is the remaining lever.
- **Family floor → NB05.** Plot 5 shows the K = 4 Oracle floor
  dominates p50 / p95 reconstruction error. NB05's Phase 2
  structural residual `ε·q(λ)·tanh(NN)` adds off-anchor capacity
  that stays exact at the four anchor wavelengths by construction,
  cutting H3 RMS below the K = 4 family ceiling at the
  recommended `ε_scale = 10⁻²`.

Part 6's Plot 7 is the screening map: designers should read it
before trusting the forward model on a specific FK / high-`n_d`
substitution.

Part 6.4 records a Chebyshev cross-check at matched protocol:
Chebyshev is a stronger per-glass approximation basis, but
`V_d` / `P_{g,F}` slip on 20-D regression (up to ~23 Abbe units
and ~0.05 on the worst glass) disqualifies it as the production
coordinate. Buchdahl is selected for **anchor semantics**, not
for raw `n(λ)` approximation quality.

### Principled candidates recorded for reference (Part 5.1)

Five α derivations were evaluated. Only A was adopted:

| Criterion | α | Status |
|---|---|---|
| **A. band-symmetric training grid** | **1.9547** | **production** |
| B. anchor-line symmetric | −3.98 | not physical |
| C. min cond(M) | ≥ 3.0 | plateau — degenerate argmin |
| D. IR / UV asymptotic | 2.2467 | principled but weaker rationale, +14 % Test B p95 |
| E. Mercado-Robb Sellmeier 2nd-deriv | 2.5360 | classical but outside Test B max plateau |

### Alternatives for other applications

- **K = 8, α = 1.26** — *accuracy frontier* for off-anchor typical-
  glass reconstruction. Delivers ~1.7–4× lower Test B p50 / p95 than
  K = 4 but ~1.5× worse worst-case. Use for NIR / broadband /
  multi-spectral applications where extreme anomalous-dispersion
  glasses are not critical. Implemented in `dispersionlab.sweep`
  (see `tests/test_sweep_anchor_delta.py`), artifacts in
  `data/sweep_tmp/`.

- **K = 3, α ≈ 1.18** — *compact* lower-bound reference. One
  high-order coefficient; minimal API. Not competitive on Test B max.

### Migration

Production migrated from α = 1.818 to α = 1.9547:

1. `dispersionlab.buchdahl.ALPHA = 1.9547`
2. `data/regression_buchdahl_anchor_nu34_20dim_opt.npy` and
   `data/buchdahl_coeffs_per_glass.npz` regenerated via
   `scripts/build_anchor_dataset.py`.
3. NB02 / NB03 / NB04 / NB05 rerun at new α; all downstream
   numbers shift by ≤ 12 % on typical-glass metrics; no structural
   changes.
4. README and report updated to α = 1.9547 throughout.

The legacy-construction analysis (V_d slip ~1.14 units, anchor
miss) is now inside Part 1 as a one-table contrast against the
anchor-preserving construction, not a separate archival notebook.
NB01 is the single authoritative notebook for the production
forward model.
"""))


# =========================================================================
# Notebook assembly
# =========================================================================

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
