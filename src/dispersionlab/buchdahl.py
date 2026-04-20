"""Anchor-preserving Buchdahl dispersion model — shared math.

NumPy is the computational backbone. Arithmetic primitives used on the
forward path (omega, feature vector, low-order solve, reconstruction,
anchor metrics) are backend-agnostic: the same expressions evaluate
correctly on numpy arrays and on torch tensors. Per-glass LSQ fits are
numpy-only by design — they consume a pre-computed Sellmeier truth grid
and are never differentiated through.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------
# Spectral-line and grid constants
# ---------------------------------------------------------------------

LAMBDA_D = 0.5875618   # d-line, yellow He
LAMBDA_g = 0.4358343   # g-line, blue Hg
LAMBDA_F = 0.4861327   # F-line, blue H
LAMBDA_C = 0.6562725   # C-line, red H

ALPHA = 1.9547         # band-symmetric α: (b-a)/(2ab) with a=λ_d-λ_min,
                       # b=λ_max-λ_d on the training grid [0.365, 2.3] μm.
                       # Makes the Buchdahl coordinate ω symmetric around 0
                       # on the training band: |ω(λ_min)| = ω(λ_max).
                       # Chosen by NB01 as the principled production α;
                       # see NB01 Parts 3–6 for the full derivation and the
                       # comparison against 4 alternative principled criteria.

WAVELENGTHS = np.array([
    0.36501, 0.40466, 0.43583, 0.48613, 0.54607,
    0.58756, 0.58929, 0.6328,  0.64385, 0.65627,
    0.70652, 0.85211, 1.01398, 1.060,   1.52958,
    1.97009, 2.3
])

# Schott normal-line coefficients for dPgF definition:
#   dPgF = PgF - (SCHOTT_NORMAL_A - SCHOTT_NORMAL_B * Vd)
SCHOTT_NORMAL_A = 0.6438
SCHOTT_NORMAL_B = 0.001682

# Catalog filter applied by load_glass_catalog()
VD_MIN, VD_MAX = 10.0, 100.0
DPGF_MIN, DPGF_MAX = -0.02, 0.10


# ---------------------------------------------------------------------
# Backend dispatch (numpy / torch)
# ---------------------------------------------------------------------

def _is_torch(x):
    try:
        import torch
    except ImportError:
        return False
    return isinstance(x, torch.Tensor)


def _stack_last(terms):
    """Stack along a new trailing axis; picks backend from the first term."""
    if _is_torch(terms[0]):
        import torch
        return torch.stack(list(terms), dim=-1)
    return np.stack([np.asarray(t) for t in terms], axis=-1)


# ---------------------------------------------------------------------
# Buchdahl coordinate
# ---------------------------------------------------------------------

def buchdahl_omega(lam, alpha=ALPHA):
    """Buchdahl chromatic coordinate, backend-agnostic.

        omega(lam) = (lam - lambda_d) / (1 + alpha * (lam - lambda_d))

    Accepts scalars, numpy arrays, or torch tensors.
    """
    dl = lam - LAMBDA_D
    return dl / (1.0 + alpha * dl)


# Anchor omegas evaluated once at import.
OMEGA_D = 0.0
OMEGA_g = float(buchdahl_omega(LAMBDA_g))
OMEGA_F = float(buchdahl_omega(LAMBDA_F))
OMEGA_C = float(buchdahl_omega(LAMBDA_C))

# Anchor-difference matrices.
#   Dispersion anchors in Buchdahl form:
#     [omega_F - omega_C, omega_F^2 - omega_C^2] @ (nu1, nu2)
#       + [omega_F^3 - omega_C^3, omega_F^4 - omega_C^4] @ (nu3, nu4)
#       = dn_FC
#     (same for g - F row).
M_COUP = np.array([
    [OMEGA_F - OMEGA_C,         OMEGA_F**2 - OMEGA_C**2],
    [OMEGA_g - OMEGA_F,         OMEGA_g**2 - OMEGA_F**2],
])
D_COUP = np.array([
    [OMEGA_F**3 - OMEGA_C**3,   OMEGA_F**4 - OMEGA_C**4],
    [OMEGA_g**3 - OMEGA_F**3,   OMEGA_g**4 - OMEGA_F**4],
])
M_INV = np.linalg.inv(M_COUP)
B_COUP = M_INV @ D_COUP   # (nu1, nu2) = M^-1 @ (dn_FC, dn_gF) - B @ (nu3, nu4)

# Scalar entries used in the backend-agnostic solve.
_MI00, _MI01 = float(M_INV[0, 0]), float(M_INV[0, 1])
_MI10, _MI11 = float(M_INV[1, 0]), float(M_INV[1, 1])
_B00,  _B01  = float(B_COUP[0, 0]), float(B_COUP[0, 1])
_B10,  _B11  = float(B_COUP[1, 0]), float(B_COUP[1, 1])


# ---------------------------------------------------------------------
# 20-dim polynomial feature vector
# ---------------------------------------------------------------------

def feature_vec_20(nd, vd, dPgF):
    """20-dim polynomial feature vector of (nd, Vd, dPgF).

    Inputs may be Python scalars, numpy arrays, or torch tensors — all
    with the same shape. The output has shape ``(..., 20)``.

    Ordering matches NB01's ``feature_vec``: 10 square terms, then 10 cube
    terms.
    """
    # Broadcasting-safe "1" in the same backend/dtype as the inputs.
    one = nd * 0.0 + 1.0
    square = [
        one, nd, vd, dPgF,
        nd*nd, vd*vd, dPgF*dPgF,
        nd*vd, nd*dPgF, vd*dPgF,
    ]
    cube = [
        nd**3, vd**3, dPgF**3,
        nd*nd*vd, nd*nd*dPgF,
        vd*vd*nd, vd*vd*dPgF,
        dPgF*dPgF*nd, dPgF*dPgF*vd,
        nd*vd*dPgF,
    ]
    return _stack_last(square + cube)


# ---------------------------------------------------------------------
# Anchor-preserving low-order solve
# ---------------------------------------------------------------------

def solve_nu12_from_nu34(nd, vd, dPgF, nu3, nu4):
    """Return ``(nu1, nu2)`` such that Buchdahl anchors hold exactly
    given ``(nu3, nu4)``.

    Preserved anchors (machine precision, independent of nu3/nu4):

        n(lambda_d)                   = nd
        n(lambda_F) - n(lambda_C)     = (nd - 1) / Vd
        n(lambda_g) - n(lambda_F)     = P_{g,F} * (nd - 1) / Vd

    where ``P_{g,F} = SCHOTT_NORMAL_A - SCHOTT_NORMAL_B * Vd + dPgF``.

    Works elementwise on scalars, numpy arrays, or torch tensors of the
    same broadcasted shape. Fully differentiable in all five inputs.
    """
    dn_FC = (nd - 1.0) / vd
    PgF = SCHOTT_NORMAL_A - SCHOTT_NORMAL_B * vd + dPgF
    dn_gF = PgF * dn_FC
    a1 = _MI00 * dn_FC + _MI01 * dn_gF
    a2 = _MI10 * dn_FC + _MI11 * dn_gF
    nu1 = a1 - _B00 * nu3 - _B01 * nu4
    nu2 = a2 - _B10 * nu3 - _B11 * nu4
    return nu1, nu2


def assemble_anchor_preserving_coefficients(nd, vd, dPgF, nu3, nu4):
    """Return the full ``(nu1, nu2, nu3, nu4)`` tuple with anchors
    preserved. Thin convenience wrapper around ``solve_nu12_from_nu34``.
    """
    nu1, nu2 = solve_nu12_from_nu34(nd, vd, dPgF, nu3, nu4)
    return nu1, nu2, nu3, nu4


# ---------------------------------------------------------------------
# Per-glass coefficient fits (numpy only)
# ---------------------------------------------------------------------

def fit_anchor_preserving_coefficients(
    nd, vd, dPgF, n_truth,
    *, wavelengths=WAVELENGTHS,
):
    """Per-glass anchor-constrained LSQ fit for ``(nu1, nu2, nu3, nu4)``
    at the module-level ``ALPHA``.

    The (nu3, nu4) LSQ is posed on the anchor-consistent basis

        phi3(omega) = omega^3 - B00*omega - B10*omega^2
        phi4(omega) = omega^4 - B01*omega - B11*omega^2

    against the residual after subtracting the anchor-implied low-order
    part, then ``(nu1, nu2)`` are back-solved via the coupling matrix.
    The resulting coefficients satisfy all three anchors at machine
    precision by construction — this is the canonical "oracle" target
    for the cross-glass regressor.

    ``alpha`` is not accepted as an argument because the anchor matrices
    ``M_COUP, D_COUP, M_INV, B_COUP`` are frozen at ``ALPHA`` at import
    time. Mixing a different ``omega`` with those frozen matrices would
    silently violate anchors. If an alpha sweep is ever needed, build a
    matched (omega, matrix) pair via a new helper that recomputes all
    four quantities from the requested alpha.
    """
    omega = np.asarray(buchdahl_omega(np.asarray(wavelengths)))
    dn_FC = (nd - 1.0) / vd
    PgF = SCHOTT_NORMAL_A - SCHOTT_NORMAL_B * vd + dPgF
    dn_gF = PgF * dn_FC
    a = M_INV @ np.array([dn_FC, dn_gF])
    phi3 = omega**3 - omega * _B00 - omega**2 * _B10
    phi4 = omega**4 - omega * _B01 - omega**2 * _B11
    basis = np.column_stack([phi3, phi4])
    target = np.asarray(n_truth) - nd - omega * a[0] - omega**2 * a[1]
    nu34, *_ = np.linalg.lstsq(basis, target, rcond=None)
    nu12 = a - B_COUP @ nu34
    return float(nu12[0]), float(nu12[1]), float(nu34[0]), float(nu34[1])


def fit_legacy_coefficients(
    nd, vd, dPgF, n_truth,
    *, wavelengths=WAVELENGTHS,
):
    """Per-glass **legacy** (pre-repair) fit at the module-level ``ALPHA``.

    Solves (nu1, nu2) from anchors while ignoring the (nu3, nu4)
    contribution, then fits (nu3, nu4) on the resulting residual. This
    reproduces the NB01 original pipeline used for the pre-fix
    contrast table — the output anchors slip at lambda_F, lambda_C,
    lambda_g by exactly ``nu3*omega^3 + nu4*omega^4`` at those
    wavelengths. Do not use in downstream workflows.

    ``alpha`` is intentionally not exposed; see the matching note on
    ``fit_anchor_preserving_coefficients``.
    """
    omega = np.asarray(buchdahl_omega(np.asarray(wavelengths)))
    dn_FC = (nd - 1.0) / vd
    PgF = SCHOTT_NORMAL_A - SCHOTT_NORMAL_B * vd + dPgF
    dn_gF = PgF * dn_FC
    nu12 = np.linalg.solve(M_COUP, np.array([dn_FC, dn_gF]))
    residual = np.asarray(n_truth) - nd - nu12[0] * omega - nu12[1] * omega**2
    A = np.column_stack([omega**3, omega**4])
    nu34, *_ = np.linalg.lstsq(A, residual, rcond=None)
    return float(nu12[0]), float(nu12[1]), float(nu34[0]), float(nu34[1])


# ---------------------------------------------------------------------
# Reconstruction and diagnostics
# ---------------------------------------------------------------------

def reconstruct_n(nd, nu1, nu2, nu3, nu4, omega):
    """Evaluate the 4-term Buchdahl expansion ``n(omega)``.

    All inputs backend-agnostic; broadcasts over omega.
    """
    return nd + nu1*omega + nu2*omega**2 + nu3*omega**3 + nu4*omega**4


def anchor_error_metrics(nd, vd, dPgF, nu1, nu2, nu3, nu4):
    """Max-norm anchor-slip metrics for a single glass.

    Returns a dict with::

        F1     = |n_out(lambda_d) - nd|
        F2_abs = |Vd_out - Vd|
        F2_rel = F2_abs / Vd
        F3     = |dPgF_out - dPgF|

    Anchor-preserving coefficients yield all four at floating-point
    precision (~1e-14 to 1e-16). Legacy coefficients show F2_abs up to
    O(1) units of Vd on some glasses.
    """
    lam = np.array([LAMBDA_D, LAMBDA_F, LAMBDA_C, LAMBDA_g])
    omega = buchdahl_omega(lam)
    n_out = reconstruct_n(nd, nu1, nu2, nu3, nu4, omega)
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
