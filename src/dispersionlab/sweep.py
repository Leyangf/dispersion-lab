"""Sweep-capable anchor-preserving Buchdahl helpers.

The frozen production path (``dispersionlab.buchdahl``) hardcodes
``K = 4`` and ``ALPHA = 1.818`` for speed and safety. This submodule adds
the variable-(K, alpha) helpers needed to *select* those hyperparameters
inside the anchor-preserving family — the order/alpha choice should
happen under the construction actually being deployed, not under a
legacy construction from which numbers are then migrated.

All helpers are NumPy-only. Once the sweep picks K*, α*, the production
module can either be updated to match or keep its current values if the
sweep confirms them.
"""
from __future__ import annotations

import numpy as np

from .buchdahl import (
    LAMBDA_C,
    LAMBDA_D,
    LAMBDA_F,
    LAMBDA_g,
    SCHOTT_NORMAL_A,
    SCHOTT_NORMAL_B,
    WAVELENGTHS,
    buchdahl_omega,
)


# ---------------------------------------------------------------------
# Matrices (depend on both alpha and K)
# ---------------------------------------------------------------------

def build_anchor_matrices(alpha, K):
    """Anchor-constraint matrices for order-K Buchdahl at this alpha.

    Returns ``(M, D_high, M_INV, B_COUP)`` where:

      * ``M``      — (2, 2)       anchor differences of ω, ω²
      * ``D_high`` — (2, K-2)     anchor differences of ω³, …, ω^K
      * ``M_INV``  — (2, 2)
      * ``B_COUP`` — (2, K-2)     = ``M_INV @ D_high``

    At K = 2 the high-order matrices have shape (2, 0) — legal and
    interpreted as "no high-order coefficients to couple".

    The constraint system reads:
        M @ [ν₁, ν₂]ᵀ + D_high @ [ν₃, …, ν_K]ᵀ = [Δn_FC, Δn_gF]ᵀ
    so given any ``nu_high``,
        [ν₁, ν₂]ᵀ = M_INV @ [Δn_FC, Δn_gF]ᵀ − B_COUP @ nu_high.
    """
    if K < 2:
        raise ValueError(f"K must be >= 2 (got {K})")
    OMg = float(buchdahl_omega(LAMBDA_g, alpha))
    OMf = float(buchdahl_omega(LAMBDA_F, alpha))
    OMc = float(buchdahl_omega(LAMBDA_C, alpha))
    M = np.array([
        [OMf - OMc,            OMf**2 - OMc**2],
        [OMg - OMf,            OMg**2 - OMf**2],
    ])
    if K == 2:
        D_high = np.zeros((2, 0))
    else:
        cols = []
        for p in range(3, K + 1):
            cols.append([OMf**p - OMc**p, OMg**p - OMf**p])
        D_high = np.array(cols).T                    # (2, K-2)
    M_INV = np.linalg.inv(M)
    B_COUP = M_INV @ D_high                          # (2, K-2)
    return M, D_high, M_INV, B_COUP


# ---------------------------------------------------------------------
# Anchor-preserving low-order solve (arbitrary K)
# ---------------------------------------------------------------------

def solve_nu12_from_nu_high(nd, vd, dPgF, nu_high, *, alpha, K):
    """Return ``(nu1, nu2)`` s.t. Buchdahl order-K anchors hold exactly
    at this alpha, given a length ``K-2`` ``nu_high`` vector.

    For K == 2, ``nu_high`` is ignored (must be empty).
    """
    if K < 2:
        raise ValueError(f"K must be >= 2 (got {K})")
    nu_high = np.asarray(nu_high, dtype=np.float64).reshape(-1)
    if nu_high.size != K - 2:
        raise ValueError(
            f"nu_high length {nu_high.size} does not match K-2 = {K-2}"
        )
    _, _, M_INV, B_COUP = build_anchor_matrices(alpha, K)
    dn_FC = (nd - 1.0) / vd
    PgF = SCHOTT_NORMAL_A - SCHOTT_NORMAL_B * vd + dPgF
    dn_gF = PgF * dn_FC
    a = M_INV @ np.array([dn_FC, dn_gF])
    if K == 2:
        nu12 = a
    else:
        nu12 = a - B_COUP @ nu_high
    return float(nu12[0]), float(nu12[1])


# ---------------------------------------------------------------------
# Per-glass fits (arbitrary K)
# ---------------------------------------------------------------------

def fit_anchor_preserving_coefficients_K(
    nd, vd, dPgF, n_truth, *, alpha, K, wavelengths=WAVELENGTHS,
):
    """Per-glass anchor-constrained LSQ for order K at the given alpha.

    Returns a length-K array ``[ν₁, ν₂, …, ν_K]``. Anchors F1, F2, F3 are
    satisfied at machine precision by construction.
    """
    if K < 2:
        raise ValueError(f"K must be >= 2 (got {K})")
    omega = np.asarray(buchdahl_omega(np.asarray(wavelengths), alpha))
    n_truth = np.asarray(n_truth)

    _, _, M_INV, B_COUP = build_anchor_matrices(alpha, K)
    dn_FC = (nd - 1.0) / vd
    PgF = SCHOTT_NORMAL_A - SCHOTT_NORMAL_B * vd + dPgF
    dn_gF = PgF * dn_FC
    a = M_INV @ np.array([dn_FC, dn_gF])

    if K == 2:
        return np.array([a[0], a[1]], dtype=np.float64)

    # Anchor-consistent high-order basis:
    #   Φ_k(ω) = ω^k − ω·B[0, k−3] − ω²·B[1, k−3]
    basis_cols = []
    for kk in range(3, K + 1):
        j = kk - 3
        phi_k = omega**kk - omega * B_COUP[0, j] - omega**2 * B_COUP[1, j]
        basis_cols.append(phi_k)
    basis = np.column_stack(basis_cols)              # (N_wl, K-2)
    target = n_truth - nd - omega * a[0] - omega**2 * a[1]
    nu_high, *_ = np.linalg.lstsq(basis, target, rcond=None)
    nu12 = a - B_COUP @ nu_high
    return np.concatenate([[nu12[0], nu12[1]], nu_high])


def fit_legacy_coefficients_K(
    nd, vd, dPgF, n_truth, *, alpha, K, wavelengths=WAVELENGTHS,
):
    """Per-glass **legacy** (anchor-unaware) fit for order K.

    Legacy construction: solve (ν₁, ν₂) from the anchor diffs ignoring
    any high-order contribution, then LSQ-fit ``(ν₃, …, ν_K)`` on the
    residual using the bare ``[ω³, …, ω^K]`` basis. The resulting
    coefficients slip at F/C/g anchors by exactly the high-order
    contribution at those wavelengths.

    Retained only for legacy-vs-anchor-preserving contrast tables.
    """
    if K < 2:
        raise ValueError(f"K must be >= 2 (got {K})")
    omega = np.asarray(buchdahl_omega(np.asarray(wavelengths), alpha))
    n_truth = np.asarray(n_truth)

    M, _, _, _ = build_anchor_matrices(alpha, K)
    dn_FC = (nd - 1.0) / vd
    PgF = SCHOTT_NORMAL_A - SCHOTT_NORMAL_B * vd + dPgF
    dn_gF = PgF * dn_FC
    nu12 = np.linalg.solve(M, np.array([dn_FC, dn_gF]))

    if K == 2:
        return np.array([nu12[0], nu12[1]], dtype=np.float64)

    residual = n_truth - nd - nu12[0] * omega - nu12[1] * omega**2
    A = np.column_stack([omega**p for p in range(3, K + 1)])
    nu_high, *_ = np.linalg.lstsq(A, residual, rcond=None)
    return np.concatenate([[nu12[0], nu12[1]], nu_high])


# ---------------------------------------------------------------------
# Reconstruction and anchor metrics (arbitrary K)
# ---------------------------------------------------------------------

def reconstruct_n_K(nd, nu_1_to_K, omega):
    """Evaluate Buchdahl K-order expansion at the given ω values."""
    nu = np.asarray(nu_1_to_K)
    n = nd + np.zeros_like(omega)
    for k, nu_k in enumerate(nu, start=1):
        n = n + nu_k * omega**k
    return n


def anchor_error_metrics_K(nd, vd, dPgF, nu_1_to_K, *, alpha):
    """Anchor-slip metrics for arbitrary K at this alpha.

    Returns a dict with ``F1, F2_abs, F2_rel, F3``.
    """
    lam = np.array([LAMBDA_D, LAMBDA_F, LAMBDA_C, LAMBDA_g])
    omega = buchdahl_omega(lam, alpha)
    n_out = reconstruct_n_K(nd, nu_1_to_K, omega)
    nd_out, nF_out, nC_out, ng_out = n_out
    dn_FC_out = nF_out - nC_out
    Vd_out = (nd_out - 1.0) / dn_FC_out
    PgF_out = (ng_out - nF_out) / dn_FC_out
    dPgF_out = PgF_out - (SCHOTT_NORMAL_A - SCHOTT_NORMAL_B * Vd_out)
    F2_abs = abs(Vd_out - vd)
    return dict(
        F1=float(abs(nd_out - nd)),
        F2_abs=float(F2_abs),
        F2_rel=float(F2_abs / vd),
        F3=float(abs(dPgF_out - dPgF)),
    )


# ---------------------------------------------------------------------
# Batched helpers for (K, alpha) sweeps
# ---------------------------------------------------------------------

def vectorize_catalog(glasses, wavelengths=WAVELENGTHS):
    """Pack a list-of-dict glass catalog into batched NumPy arrays.

    Returns a dict with keys ``nd``, ``vd``, ``dPgF``, ``n_truth``, each
    a NumPy array whose leading axis spans the catalog. ``n_truth`` is
    shaped ``(N, len(wavelengths))``; all other fields are ``(N,)``.
    """
    import numpy as np
    nd = np.array([g["nd"] for g in glasses])
    vd = np.array([g["vd"] for g in glasses])
    dPgF = np.array([g["dPgF"] for g in glasses])
    if len(glasses) and "n_truth" in glasses[0]:
        n_truth = np.array([g["n_truth"] for g in glasses])
    else:
        from .catalog import sellmeier_formula_2
        n_truth = np.array([
            [float(sellmeier_formula_2(lam, g["sellmeier"])) for lam in wavelengths]
            for g in glasses
        ])
    return dict(nd=nd, vd=vd, dPgF=dPgF, n_truth=n_truth)


def fit_and_predict_batched(
    nd, vd, dPgF, n_truth_train, omega_train, omega_eval,
    M_INV, B_COUP, K,
):
    """Anchor-preserving LSQ across N glasses, fully vectorised.

    Fits ``(ν_3, …, ν_K)`` per glass to ``n_truth_train`` at
    ``omega_train`` under the anchor-preserving construction, then
    evaluates the reconstructed curve at ``omega_eval``. Used by the
    ``(K, α)`` sweep and by notebook 01's cross-sectional diagnostics.

    Returns ``(n_pred, nu_high, nu1, nu2)`` where:

      * ``n_pred``  : (N, len(omega_eval))
      * ``nu_high`` : (N, K-2)   empty second axis for K == 2
      * ``nu1``     : (N,)
      * ``nu2``     : (N,)
    """
    import numpy as np
    N = len(nd)
    dn_FC = (nd - 1.0) / vd
    PgF = SCHOTT_NORMAL_A - SCHOTT_NORMAL_B * vd + dPgF
    dn_gF = PgF * dn_FC
    a = M_INV @ np.stack([dn_FC, dn_gF], axis=0)
    a0, a1 = a[0], a[1]

    low_train = (a0[:, None] * omega_train[None, :]
                 + a1[:, None] * omega_train[None, :] ** 2)
    target_train = n_truth_train - nd[:, None] - low_train

    if K == 2:
        nu_high = np.zeros((N, 0))
        nu1 = a0
        nu2 = a1
    else:
        basis_cols = []
        for kk in range(3, K + 1):
            j = kk - 3
            phi_k = (omega_train ** kk
                     - omega_train * B_COUP[0, j]
                     - omega_train ** 2 * B_COUP[1, j])
            basis_cols.append(phi_k)
        basis = np.column_stack(basis_cols)
        X, *_ = np.linalg.lstsq(basis, target_train.T, rcond=None)
        nu_high = X.T
        delta = B_COUP @ X
        nu1 = a0 - delta[0]
        nu2 = a1 - delta[1]

    n_pred = (nd[:, None]
              + nu1[:, None] * omega_eval[None, :]
              + nu2[:, None] * omega_eval[None, :] ** 2)
    for kk in range(3, K + 1):
        j = kk - 3
        n_pred = n_pred + nu_high[:, j:j + 1] * omega_eval[None, :] ** kk
    return n_pred, nu_high, nu1, nu2


def test_a_batched(catalog_arr, alpha, K, wavelengths=WAVELENGTHS):
    """Batched Test A (wavelength leave-one-out) at (alpha, K) across
    the whole catalog.

    Returns a dict with::

        train_max : (N,)  per-glass max |err| using all wavelengths
        hold_max  : (N,)  per-glass max over leave-one-out hold errors

    ``catalog_arr`` is the output of ``vectorize_catalog``.
    """
    import numpy as np
    omega = buchdahl_omega(wavelengths, alpha)
    _, _, M_INV, B_COUP = build_anchor_matrices(alpha, K)
    nd = catalog_arr["nd"]
    vd = catalog_arr["vd"]
    dPgF = catalog_arr["dPgF"]
    n_truth = catalog_arr["n_truth"]
    N = len(nd)

    n_pred_full, _, _, _ = fit_and_predict_batched(
        nd, vd, dPgF, n_truth, omega, omega, M_INV, B_COUP, K
    )
    train_max = np.abs(n_pred_full - n_truth).max(axis=1)

    hold_errs = np.zeros((N, len(wavelengths)))
    for j in range(len(wavelengths)):
        mask = np.ones(len(wavelengths), dtype=bool)
        mask[j] = False
        omega_train = omega[mask]
        omega_eval = np.array([omega[j]])
        n_pred_j, _, _, _ = fit_and_predict_batched(
            nd, vd, dPgF, n_truth[:, mask],
            omega_train, omega_eval, M_INV, B_COUP, K,
        )
        hold_errs[:, j] = np.abs(n_pred_j[:, 0] - n_truth[:, j])

    hold_max = hold_errs.max(axis=1)
    return dict(train_max=train_max, hold_max=hold_max, hold_errs=hold_errs)


def fit_all_glasses_anchor(catalog_arr, alpha, K, wavelengths=WAVELENGTHS):
    """Fit the anchor-preserving per-glass coefficients on the whole
    catalog at (alpha, K). Returns ``(N, K)`` coefficient array."""
    import numpy as np
    omega = buchdahl_omega(wavelengths, alpha)
    _, _, M_INV, B_COUP = build_anchor_matrices(alpha, K)
    nd = catalog_arr["nd"]
    vd = catalog_arr["vd"]
    dPgF = catalog_arr["dPgF"]
    n_truth = catalog_arr["n_truth"]
    _, nu_high, nu1, nu2 = fit_and_predict_batched(
        nd, vd, dPgF, n_truth, omega, omega, M_INV, B_COUP, K
    )
    N = len(nd)
    nu_all = np.zeros((N, K))
    nu_all[:, 0] = nu1
    nu_all[:, 1] = nu2
    if K > 2:
        nu_all[:, 2:] = nu_high
    return nu_all
