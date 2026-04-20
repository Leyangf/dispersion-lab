"""Anchor-preserving Buchdahl **accuracy-frontier sweep** (diagnostic).

This script is **not** the production-artifact generator. The production
baseline is K = 4, α = 1.818, shipped via
``scripts/build_anchor_dataset.py`` and consumed by NB02/NB03/NB05.

Purpose of this sweep: identify the (K, α) point that minimises p95
absolute Test A wavelength-holdout error under the anchor-preserving
family. The result is reported as an **accuracy frontier** — not as a
replacement for production — so the report can honestly cite both.

Selection criterion (chosen by project lead):
  * **Primary**: p95 across 544 glasses of per-glass max absolute hold
    error (Test A wavelength leave-one-out). Smaller is better.
  * **Diagnostic only** (never hard-filtered): hold/train ratio,
    median, max.

Stages:
  A. **Full (K × α) grid** — K ∈ {2, …, 8}, α ∈ linspace(1.2, 2.8, 21):
     batched Test A, report median / p95 / max hold error plus ratio.
     Find global p95 minimum (K*_coarse, α*_coarse). Expand grid if on
     boundary.
  B. **Fine α sweep at K*_coarse**: α ∈ linspace(α*_coarse − 0.08,
     α*_coarse + 0.08, 9). Pick α*_fine.
  C. **Legacy vs anchor-preserving contrast at (K*, α*)**.
  D. **Test B glass-level CV at (K*, α*)**: DBSCAN cluster split.

Artifacts are written to ``data/sweep_tmp/`` with descriptive filenames
so the (K, α) provenance is visible and the directory cannot be
accidentally consumed as production.

Run:

    uv run python scripts/build_01_anchor_model.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

from dispersionlab import WAVELENGTHS, feature_vec_20, load_glass_catalog
from dispersionlab.buchdahl import (
    SCHOTT_NORMAL_A,
    SCHOTT_NORMAL_B,
    buchdahl_omega,
)
from dispersionlab.sweep import (
    anchor_error_metrics_K,
    build_anchor_matrices,
    fit_all_glasses_anchor,
    fit_and_predict_batched,
    fit_legacy_coefficients_K,
    reconstruct_n_K,
    test_a_batched,
    vectorize_catalog,
)


K_GRID = [2, 3, 4, 5, 6, 7, 8]
ALPHA_COARSE = np.linspace(1.2, 2.8, 21)       # step 0.08
ALPHA_FINE_WIDTH = 0.08                        # ±width around coarse winner
ALPHA_FINE_STEPS = 9                           # 9 fine points spanning width

# Production baseline (kept fixed independently of this sweep)
K_PROD = 4
ALPHA_PROD = 1.818


# =====================================================================
# (Batched Test A, vectorize_catalog, fit_and_predict_batched,
#  fit_all_glasses_anchor are imported from dispersionlab.sweep above.)
# =====================================================================


per_glass_anchor_fit_batched = fit_all_glasses_anchor   # local alias


# =====================================================================
# Aggregation helpers
# =====================================================================

def _stats(arr):
    return dict(
        median=float(np.median(arr)),
        p95=float(np.percentile(arr, 95)),
        max=float(arr.max()),
    )


def _fmt_e(x, n=3):
    return f"{x:.{n}e}"


# =====================================================================
# Stages
# =====================================================================

def stage_coarse_grid(catalog_arr, K_grid, alpha_grid):
    """Run Test A on the full K × α coarse grid."""
    print("\n" + "=" * 78)
    print("STAGE A — Full (K × α) coarse grid, Test A absolute hold error")
    print("=" * 78)
    print(f"  K in {list(K_grid)}   α in linspace(1.2, 2.8, 21)   "
          f"N = {len(catalog_arr['nd'])} glasses")
    print(f"  primary metric: p95 of per-glass max hold error")
    print()

    results = {}  # (K, α) -> dict
    for K in K_grid:
        for a in alpha_grid:
            ta = test_a_batched(catalog_arr, float(a), K)
            h = _stats(ta["hold_max"])
            t = _stats(ta["train_max"])
            results[(K, float(a))] = dict(
                hold=h, train=t,
                ratio=h["p95"] / max(t["p95"], 1e-30),
            )

    # Print per-K summary table (show α that minimizes p95 for each K)
    print(f"  per-K best α under p95(hold_max):")
    print()
    print(f"    {'K':>2} | {'best α':>8} | {'med hold':>10} {'p95 hold':>10} {'max hold':>10} | "
          f"{'p95 train':>10} | ratio")
    print(f"    {'-'*2}-+-{'-'*8}-+-{'-'*10} {'-'*10} {'-'*10}-+-{'-'*10}-+-{'-'*6}")
    k_best_rows = {}
    for K in K_grid:
        rows = [(a, results[(K, float(a))]) for a in alpha_grid]
        best_a, best_r = min(rows, key=lambda x: x[1]["hold"]["p95"])
        k_best_rows[K] = dict(alpha=best_a, **best_r)
        print(f"    {K:>2} | {best_a:>8.3f} | {_fmt_e(best_r['hold']['median']):>10} "
              f"{_fmt_e(best_r['hold']['p95']):>10} {_fmt_e(best_r['hold']['max']):>10} | "
              f"{_fmt_e(best_r['train']['p95']):>10} | {best_r['ratio']:>6.2f}×")

    # Global best
    global_best_K, global_best_row = min(
        k_best_rows.items(), key=lambda x: x[1]["hold"]["p95"]
    )
    global_best_alpha = float(global_best_row["alpha"])
    print()
    print(f"  Coarse-grid winner: K = {global_best_K}, α = {global_best_alpha:.3f}")
    print(f"    p95 hold error = {_fmt_e(global_best_row['hold']['p95'])}")
    print(f"    ratio (diagnostic) = {global_best_row['ratio']:.2f}×")
    return results, global_best_K, global_best_alpha


def stage_fine_alpha(catalog_arr, K_star, alpha_center, width, n_steps):
    """Fine α sweep at K = K_star."""
    print("\n" + "=" * 78)
    print(f"STAGE B — Fine α sweep at K = {K_star}")
    print("=" * 78)
    lo = alpha_center - width
    hi = alpha_center + width
    alpha_fine = np.linspace(lo, hi, n_steps)
    print(f"  α in linspace({lo:.3f}, {hi:.3f}, {n_steps})")
    print()
    print(f"    {'α':>8} | {'med hold':>10} {'p95 hold':>10} {'max hold':>10} | "
          f"{'p95 train':>10} | ratio")
    print(f"    {'-'*8}-+-{'-'*10} {'-'*10} {'-'*10}-+-{'-'*10}-+-{'-'*6}")
    rows = []
    for a in alpha_fine:
        ta = test_a_batched(catalog_arr, float(a), K_star)
        h = _stats(ta["hold_max"])
        t = _stats(ta["train_max"])
        ratio = h["p95"] / max(t["p95"], 1e-30)
        rows.append(dict(alpha=float(a), hold=h, train=t, ratio=ratio))
        print(f"    {a:>8.4f} | {_fmt_e(h['median']):>10} {_fmt_e(h['p95']):>10} "
              f"{_fmt_e(h['max']):>10} | {_fmt_e(t['p95']):>10} | {ratio:>6.2f}×")

    best = min(rows, key=lambda r: r["hold"]["p95"])
    print()
    print(f"  Fine winner: α = {best['alpha']:.4f}")
    print(f"    p95 hold error = {_fmt_e(best['hold']['p95'])}")
    return best["alpha"], rows


def stage_legacy_vs_anchor(catalog_arr, K_star, alpha_star):
    """Per-glass legacy vs anchor contrast at the selected (K*, α*)."""
    print("\n" + "=" * 78)
    print(f"STAGE C — Legacy vs anchor-preserving at (K* = {K_star}, α* = {alpha_star:.4f})")
    print("=" * 78)
    nu_ap = per_glass_anchor_fit_batched(catalog_arr, float(alpha_star), K_star)
    omega = buchdahl_omega(WAVELENGTHS, float(alpha_star))
    N = len(catalog_arr["nd"])
    n_pred_ap_arr = np.zeros((N, len(WAVELENGTHS)))
    for i in range(N):
        n_pred_ap_arr[i] = reconstruct_n_K(
            catalog_arr["nd"][i], nu_ap[i], omega
        )
    err_ap = np.abs(n_pred_ap_arr - catalog_arr["n_truth"])
    rms_ap = np.sqrt((err_ap ** 2).mean(axis=1))
    max_ap = err_ap.max(axis=1)

    # Legacy per-glass loop (not batched here since only one (K, α) point)
    nu_lg = np.zeros((N, K_star))
    for i in range(N):
        nu_lg[i] = fit_legacy_coefficients_K(
            catalog_arr["nd"][i], catalog_arr["vd"][i], catalog_arr["dPgF"][i],
            catalog_arr["n_truth"][i], alpha=float(alpha_star), K=K_star,
        )
    n_pred_lg = np.zeros((N, len(WAVELENGTHS)))
    for i in range(N):
        n_pred_lg[i] = reconstruct_n_K(catalog_arr["nd"][i], nu_lg[i], omega)
    err_lg = np.abs(n_pred_lg - catalog_arr["n_truth"])
    rms_lg = np.sqrt((err_lg ** 2).mean(axis=1))
    max_lg = err_lg.max(axis=1)

    # Anchor slip
    f2_abs_lg = np.zeros(N)
    f2_rel_lg = np.zeros(N)
    f3_lg = np.zeros(N)
    f2_abs_ap = np.zeros(N)
    f2_rel_ap = np.zeros(N)
    f3_ap = np.zeros(N)
    for i in range(N):
        m_lg = anchor_error_metrics_K(
            catalog_arr["nd"][i], catalog_arr["vd"][i], catalog_arr["dPgF"][i],
            nu_lg[i], alpha=float(alpha_star),
        )
        m_ap = anchor_error_metrics_K(
            catalog_arr["nd"][i], catalog_arr["vd"][i], catalog_arr["dPgF"][i],
            nu_ap[i], alpha=float(alpha_star),
        )
        f2_abs_lg[i] = m_lg["F2_abs"]; f2_rel_lg[i] = m_lg["F2_rel"]; f3_lg[i] = m_lg["F3"]
        f2_abs_ap[i] = m_ap["F2_abs"]; f2_rel_ap[i] = m_ap["F2_rel"]; f3_ap[i] = m_ap["F3"]

    print()
    print(f"  {'metric':<22} | {'legacy':>13} | {'anchor-preserving':>18}")
    print(f"  {'-'*22} | {'-'*13} | {'-'*18}")
    print(f"  {'bare RMS median':<22} | {_fmt_e(np.median(rms_lg)):>13} | {_fmt_e(np.median(rms_ap)):>18}")
    print(f"  {'bare RMS p95':<22} | {_fmt_e(np.percentile(rms_lg, 95)):>13} | {_fmt_e(np.percentile(rms_ap, 95)):>18}")
    print(f"  {'bare RMS max':<22} | {_fmt_e(rms_lg.max()):>13} | {_fmt_e(rms_ap.max()):>18}")
    print(f"  {'max |err| median':<22} | {_fmt_e(np.median(max_lg)):>13} | {_fmt_e(np.median(max_ap)):>18}")
    print(f"  {'max |err| p95':<22} | {_fmt_e(np.percentile(max_lg, 95)):>13} | {_fmt_e(np.percentile(max_ap, 95)):>18}")
    print(f"  {'max |err| max':<22} | {_fmt_e(max_lg.max()):>13} | {_fmt_e(max_ap.max()):>18}")
    print(f"  {'F2_abs (Vd) max':<22} | {_fmt_e(f2_abs_lg.max()):>13} | {_fmt_e(f2_abs_ap.max()):>18}")
    print(f"  {'F2_rel max':<22} | {_fmt_e(f2_rel_lg.max()):>13} | {_fmt_e(f2_rel_ap.max()):>18}")
    print(f"  {'F3 (ΔPgF) max':<22} | {_fmt_e(f3_lg.max()):>13} | {_fmt_e(f3_ap.max()):>18}")
    return nu_ap


def stage_cv_test_b(catalog_arr, nu_ap, K_star):
    """Test B: 5-fold glass-level DBSCAN-cluster CV on (nd, Vd, dPgF) -> nu_high."""
    print("\n" + "=" * 78)
    print(f"STAGE D — Test B glass-level 5-fold CV (20-D poly → ν_high) at K* = {K_star}")
    print("=" * 78)
    if K_star < 3:
        print("  K = 2: no high-order coefficients to regress; Test B vacuous.")
        return None
    nu_high = nu_ap[:, 2:]
    N = len(catalog_arr["nd"])
    Phi = np.zeros((N, 20))
    for i in range(N):
        Phi[i] = feature_vec_20(
            catalog_arr["nd"][i], catalog_arr["vd"][i], catalog_arr["dPgF"][i]
        )

    # DBSCAN cluster-level 5-fold split (same settings as NB04)
    X = np.column_stack([catalog_arr["nd"], catalog_arr["vd"], catalog_arr["dPgF"]])
    Xs = StandardScaler().fit_transform(X)
    cid = DBSCAN(eps=0.15, min_samples=2).fit(Xs).labels_.copy()
    nxt = cid.max() + 1
    for i, c in enumerate(cid):
        if c == -1:
            cid[i] = nxt; nxt += 1
    uniq = np.unique(cid)
    rng = np.random.default_rng(0)
    perm = rng.permutation(uniq)
    folds = np.array_split(perm, 5)

    in_errs = []
    out_errs = []
    for k in range(5):
        test_clusters = set(folds[k].tolist())
        test_mask = np.array([c in test_clusters for c in cid])
        train_mask = ~test_mask
        A, *_ = np.linalg.lstsq(Phi[train_mask], nu_high[train_mask], rcond=None)
        pred_in = Phi[train_mask] @ A
        pred_out = Phi[test_mask] @ A
        in_errs.extend(np.abs(pred_in - nu_high[train_mask]).max(axis=1).tolist())
        out_errs.extend(np.abs(pred_out - nu_high[test_mask]).max(axis=1).tolist())
    in_errs = np.array(in_errs); out_errs = np.array(out_errs)

    def pct(a, p): return float(np.percentile(a, p))

    print()
    print(f"  {'percentile':<12} | {'in-fold':>12} | {'out-of-fold':>12} | ratio")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*6}")
    for p in [50, 95, 99]:
        i = pct(in_errs, p); o = pct(out_errs, p)
        print(f"  p{p:<11} | {_fmt_e(i):>12} | {_fmt_e(o):>12} | "
              f"{o / max(i, 1e-30):>6.2f}×")
    print(f"  {'max':<12} | {_fmt_e(in_errs.max()):>12} | {_fmt_e(out_errs.max()):>12} | "
          f"{out_errs.max() / max(in_errs.max(), 1e-30):>6.2f}×")
    return dict(in_errs=in_errs, out_errs=out_errs)


def save_artifacts(catalog_arr, glasses, K_star, alpha_star, nu_ap):
    """Train 20-D regressor on full catalog, save npz + npy with
    descriptive names under data/sweep_tmp/ (NOT production)."""
    print("\n" + "=" * 78)
    print("ARTIFACTS (accuracy-frontier sweep, written to data/sweep_tmp/)")
    print("=" * 78)
    data_dir = REPO / "data" / "sweep_tmp"
    data_dir.mkdir(exist_ok=True)
    alpha_tag = f"{alpha_star:.4f}".replace(".", "_").rstrip("0").rstrip("_")
    out_npz = data_dir / f"buchdahl_coeffs_per_glass_K{K_star}_a{alpha_tag}.npz"
    out_npy = data_dir / f"regression_buchdahl_anchor_K{K_star}_a{alpha_tag}_nuhigh_20dim.npy"

    names = np.array([g["name"] for g in glasses])
    cats = np.array([g["catalog"] for g in glasses])
    sm = np.array([g["sellmeier"] for g in glasses])
    np.savez(
        out_npz,
        name=names, catalog=cats,
        nd=catalog_arr["nd"], vd=catalog_arr["vd"],
        PgF=np.array([g["PgF"] for g in glasses]),
        dPgF=catalog_arr["dPgF"],
        sellmeier=sm,
        n_truth=catalog_arr["n_truth"],
        nu_anchor=nu_ap,
        K=K_star, alpha=alpha_star,
        wavelengths=WAVELENGTHS,
    )
    print(f"  Wrote {out_npz.name}  ({out_npz.stat().st_size / 1024:.1f} KB)")

    if K_star >= 3:
        nu_high = nu_ap[:, 2:]
        N = len(catalog_arr["nd"])
        Phi = np.zeros((N, 20))
        for i in range(N):
            Phi[i] = feature_vec_20(
                catalog_arr["nd"][i], catalog_arr["vd"][i], catalog_arr["dPgF"][i]
            )
        A_reg, *_ = np.linalg.lstsq(Phi, nu_high, rcond=None)
        np.save(out_npy, A_reg)
        print(f"  Wrote {out_npy.name}  shape = {A_reg.shape}  "
              f"(K = {K_star}, α = {alpha_star:.4f})")
    else:
        print("  K = 2: no high-order coefficients; no regression matrix saved.")


def main():
    t_all = time.time()
    print("Loading glass catalog...")
    glasses = load_glass_catalog()
    print(f"  {len(glasses)} glasses")
    catalog_arr = vectorize_catalog(glasses)

    # Stage A: coarse grid
    _, K_coarse, alpha_coarse = stage_coarse_grid(
        catalog_arr, K_GRID, ALPHA_COARSE
    )

    # Check for boundary collision on coarse grid
    if alpha_coarse in (ALPHA_COARSE[0], ALPHA_COARSE[-1]):
        print(f"\n  ⚠ coarse-grid α* at boundary {alpha_coarse:.3f}; expanding grid")
        if alpha_coarse == ALPHA_COARSE[0]:
            new_grid = np.linspace(0.6, ALPHA_COARSE[1], 9)
        else:
            new_grid = np.linspace(ALPHA_COARSE[-2], 3.4, 9)
        _, K_coarse, alpha_coarse = stage_coarse_grid(
            catalog_arr, [K_coarse], new_grid
        )

    # Stage B: fine α sweep at K_coarse
    alpha_fine, _ = stage_fine_alpha(
        catalog_arr, K_coarse, alpha_coarse,
        ALPHA_FINE_WIDTH, ALPHA_FINE_STEPS,
    )

    K_star = K_coarse
    alpha_star = alpha_fine

    print("\n" + "=" * 78)
    print(f"ACCURACY FRONTIER (this sweep):  K* = {K_star},  α* = {alpha_star:.4f}")
    print(f"PRODUCTION BASELINE (unchanged): K  = {K_PROD},  α  = {ALPHA_PROD:.3f}")
    print("=" * 78)
    print(f"  This sweep is diagnostic. Production artifacts are generated by")
    print(f"  scripts/build_anchor_dataset.py at (K={K_PROD}, α={ALPHA_PROD}) and")
    print(f"  live under data/. The winner of this sweep is stored separately")
    print(f"  under data/sweep_tmp/ for future accuracy-frontier experiments.")

    nu_ap = stage_legacy_vs_anchor(catalog_arr, K_star, alpha_star)
    stage_cv_test_b(catalog_arr, nu_ap, K_star)
    save_artifacts(catalog_arr, glasses, K_star, alpha_star, nu_ap)

    print(f"\nTotal wall time: {time.time() - t_all:.1f} s")


if __name__ == "__main__":
    main()
