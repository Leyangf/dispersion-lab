"""Anchor-preservation tests across the (K, alpha) sweep grid.

These guard the structural claim that anchor preservation holds at
machine precision for **any** (K, alpha) combination, not just the
production (4, 1.818). They also pin the equivalence between the sweep
path (``dispersionlab.sweep``) and the frozen production path
(``dispersionlab.buchdahl``) at the canonical point.
"""
from __future__ import annotations

import numpy as np
import pytest

from dispersionlab import load_glass_catalog
from dispersionlab.buchdahl import fit_anchor_preserving_coefficients
from dispersionlab.sweep import (
    anchor_error_metrics_K,
    build_anchor_matrices,
    fit_anchor_preserving_coefficients_K,
    fit_legacy_coefficients_K,
    solve_nu12_from_nu_high,
)


# Floating-point precision at arbitrary alpha (condition number varies
# with alpha so we use slightly looser tolerances than the frozen path).
TOL_F1       = 1e-13
TOL_F2_REL   = 1e-11
TOL_F3       = 1e-11


@pytest.fixture(scope="module")
def sample_glasses():
    return load_glass_catalog()[:20]


@pytest.mark.parametrize("K", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("alpha", [1.2, 1.5, 1.818, 2.0, 2.5])
def test_fit_anchor_preserving_K_preserves_anchors(sample_glasses, K, alpha):
    """For every (K, alpha) on the sweep grid and every sample glass,
    the per-glass anchor-preserving fit satisfies all three anchor
    equations at machine precision."""
    worst = dict(F1=0.0, F2_rel=0.0, F3=0.0)
    for g in sample_glasses:
        nu = fit_anchor_preserving_coefficients_K(
            g["nd"], g["vd"], g["dPgF"], g["n_truth"], alpha=alpha, K=K
        )
        m = anchor_error_metrics_K(g["nd"], g["vd"], g["dPgF"], nu, alpha=alpha)
        for k in worst:
            worst[k] = max(worst[k], m[k])
    assert worst["F1"]     < TOL_F1,     f"K={K}, alpha={alpha}: F1 = {worst['F1']:.3e}"
    assert worst["F2_rel"] < TOL_F2_REL, f"K={K}, alpha={alpha}: F2_rel = {worst['F2_rel']:.3e}"
    assert worst["F3"]     < TOL_F3,     f"K={K}, alpha={alpha}: F3 = {worst['F3']:.3e}"


@pytest.mark.parametrize("K", [3, 4, 5])
@pytest.mark.parametrize("alpha", [1.5, 1.818, 2.2])
def test_arbitrary_nu_high_preserves_anchors(sample_glasses, K, alpha):
    """Anchor preservation is a structural property: injecting arbitrary
    high-order coefficients and back-solving ``(ν₁, ν₂)`` still yields
    anchors at machine precision."""
    rng = np.random.default_rng(42)
    worst_F2 = 0.0
    worst_F3 = 0.0
    for g in sample_glasses:
        nu_high = rng.normal(0.0, 5e-2, size=K - 2)
        nu1, nu2 = solve_nu12_from_nu_high(
            g["nd"], g["vd"], g["dPgF"], nu_high, alpha=alpha, K=K
        )
        nu = np.concatenate([[nu1, nu2], nu_high])
        m = anchor_error_metrics_K(g["nd"], g["vd"], g["dPgF"], nu, alpha=alpha)
        worst_F2 = max(worst_F2, m["F2_rel"])
        worst_F3 = max(worst_F3, m["F3"])
    assert worst_F2 < TOL_F2_REL, f"K={K}, alpha={alpha}: F2_rel = {worst_F2:.3e}"
    assert worst_F3 < TOL_F3,     f"K={K}, alpha={alpha}: F3 = {worst_F3:.3e}"


def test_sweep_path_matches_frozen_at_canonical_point(sample_glasses):
    """At the module-level ALPHA, the sweep-version fit must match the
    frozen production-version fit to machine precision. This guards
    against any divergence between the two code paths."""
    from dispersionlab.buchdahl import ALPHA as FROZEN_ALPHA
    for g in sample_glasses:
        nu_sweep = fit_anchor_preserving_coefficients_K(
            g["nd"], g["vd"], g["dPgF"], g["n_truth"],
            alpha=FROZEN_ALPHA, K=4,
        )
        nu_frozen = fit_anchor_preserving_coefficients(
            g["nd"], g["vd"], g["dPgF"], g["n_truth"]
        )
        assert np.allclose(nu_sweep, nu_frozen, atol=1e-14, rtol=0.0), (
            f"sweep vs frozen disagree on {g['name']}: "
            f"Δν = {nu_sweep - np.asarray(nu_frozen)}"
        )


def test_legacy_fit_slips_anchors_for_K_ge_3(sample_glasses):
    """The legacy construction must visibly slip at F/C/g anchors once
    K ≥ 3 (then ν₃+ high-order terms alter anchor differences)."""
    worst_F2_rel = 0.0
    for g in sample_glasses:
        nu_legacy = fit_legacy_coefficients_K(
            g["nd"], g["vd"], g["dPgF"], g["n_truth"], alpha=1.818, K=4
        )
        m = anchor_error_metrics_K(
            g["nd"], g["vd"], g["dPgF"], nu_legacy, alpha=1.818
        )
        worst_F2_rel = max(worst_F2_rel, m["F2_rel"])
    assert worst_F2_rel > 1e-5, (
        f"legacy K=4 fit unexpectedly anchor-preserving: "
        f"max F2_rel = {worst_F2_rel:.3e}"
    )


def test_K2_constructions_coincide(sample_glasses):
    """At K=2 there is no high-order contribution, so the legacy and
    anchor-preserving constructions collapse to the same 2×2 anchor
    solve. This boundary case must be numerically identical."""
    for g in sample_glasses[:5]:
        nu_ap = fit_anchor_preserving_coefficients_K(
            g["nd"], g["vd"], g["dPgF"], g["n_truth"], alpha=1.818, K=2
        )
        nu_lg = fit_legacy_coefficients_K(
            g["nd"], g["vd"], g["dPgF"], g["n_truth"], alpha=1.818, K=2
        )
        assert np.allclose(nu_ap, nu_lg, atol=1e-14, rtol=0.0)


def test_anchor_matrices_shapes():
    """build_anchor_matrices returns correctly shaped matrices for the
    full K sweep range; K=2 edge case yields zero-width high-order
    matrices."""
    for K in [2, 3, 4, 5, 6, 7, 8]:
        M, D_high, M_INV, B_COUP = build_anchor_matrices(1.818, K)
        assert M.shape == (2, 2)
        assert M_INV.shape == (2, 2)
        assert D_high.shape == (2, K - 2)
        assert B_COUP.shape == (2, K - 2)
