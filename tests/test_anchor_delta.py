"""Machine-precision anchor-preservation regression tests.

The anchor-preserving Buchdahl construction must satisfy, for every
glass in the catalog and for arbitrary ``(nu3, nu4)``::

    n_out(lambda_d)               == nd
    Vd_out                        == Vd
    dPgF_out                      == dPgF

all at floating-point precision. These tests pin the structural claim
of the repair so that any future refactor that silently reverts to the
legacy ordering fails in CI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dispersionlab.buchdahl import (
    anchor_error_metrics,
    assemble_anchor_preserving_coefficients,
    buchdahl_omega,
    feature_vec_20,
    solve_nu12_from_nu34,
)

REPO = Path(__file__).parent.parent
DB_NPZ = REPO / "data" / "buchdahl_coeffs_per_glass.npz"
A_REG_ANCHOR_NPY = REPO / "data" / "regression_buchdahl_anchor_nu34_20dim_opt.npy"

# Tolerances reflect unavoidable FP amplification in the metric
# definitions, not a physical slip:
#   F1 = |n(lambda_d) - nd|                — no amplification (omega_d = 0)
#   F2_abs = |Vd_out - Vd|                  — amplified by Vd^2/(nd-1) ~ 2e4
#                                             so ~1e-16 * 2e4 = 2e-12 at Vd ~100
#   F2_rel = F2_abs / Vd                    — dimensionless, ULP-scale
#   F3 = |dPgF_out - dPgF|                  — contains 1/dn_FC division
TOL_F1       = 1e-14
TOL_F2_ABS   = 1e-10    # safe margin above the FP amplification floor
TOL_F2_REL   = 1e-12
TOL_F3       = 1e-12


@pytest.fixture(scope="module")
def db():
    if not DB_NPZ.exists():
        pytest.skip(f"{DB_NPZ} not built; run scripts/build_anchor_dataset.py")
    return np.load(DB_NPZ, allow_pickle=False)


@pytest.fixture(scope="module")
def a_reg_anchor():
    if not A_REG_ANCHOR_NPY.exists():
        pytest.skip(f"{A_REG_ANCHOR_NPY} not built")
    return np.load(A_REG_ANCHOR_NPY)


def test_per_glass_anchor_fit_preserves_anchors(db):
    """The per-glass LSQ target itself must satisfy all three anchors
    at machine precision for every catalog glass."""
    nd_arr = db["nd"]
    vd_arr = db["vd"]
    dPgF_arr = db["dPgF"]
    nu14 = db["nu14_anchor"]
    assert len(nd_arr) > 0

    worst = dict(F1=0.0, F2_abs=0.0, F2_rel=0.0, F3=0.0)
    for i in range(len(nd_arr)):
        nu1, nu2, nu3, nu4 = nu14[i]
        m = anchor_error_metrics(
            nd_arr[i], vd_arr[i], dPgF_arr[i], nu1, nu2, nu3, nu4
        )
        for k in worst:
            worst[k] = max(worst[k], m[k])

    assert worst["F1"]     < TOL_F1,     f"anchor-fit F1 = {worst['F1']:.3e}"
    assert worst["F2_abs"] < TOL_F2_ABS, f"anchor-fit F2_abs = {worst['F2_abs']:.3e}"
    assert worst["F2_rel"] < TOL_F2_REL, f"anchor-fit F2_rel = {worst['F2_rel']:.3e}"
    assert worst["F3"]     < TOL_F3,     f"anchor-fit F3 = {worst['F3']:.3e}"


def test_assemble_preserves_anchors_for_arbitrary_nu34(db):
    """Anchor preservation is a structural property, not a property of
    the LSQ-fitted ν3,ν4: injecting arbitrary (ν3, ν4) and back-solving
    (ν1, ν2) must still satisfy every anchor exactly."""
    nd_arr = db["nd"]
    vd_arr = db["vd"]
    dPgF_arr = db["dPgF"]
    N = len(nd_arr)

    rng = np.random.default_rng(42)
    nu3 = rng.normal(0.0, 5e-2, size=N)
    nu4 = rng.normal(0.0, 5e-2, size=N)

    worst = dict(F1=0.0, F2_abs=0.0, F2_rel=0.0, F3=0.0)
    for i in range(N):
        nu1, nu2, n3, n4 = assemble_anchor_preserving_coefficients(
            nd_arr[i], vd_arr[i], dPgF_arr[i], nu3[i], nu4[i]
        )
        m = anchor_error_metrics(
            nd_arr[i], vd_arr[i], dPgF_arr[i], nu1, nu2, n3, n4
        )
        for k in worst:
            worst[k] = max(worst[k], m[k])

    assert worst["F1"]     < TOL_F1,     f"random-nu34 F1 = {worst['F1']:.3e}"
    assert worst["F2_abs"] < TOL_F2_ABS, f"random-nu34 F2_abs = {worst['F2_abs']:.3e}"
    assert worst["F2_rel"] < TOL_F2_REL, f"random-nu34 F2_rel = {worst['F2_rel']:.3e}"
    assert worst["F3"]     < TOL_F3,     f"random-nu34 F3 = {worst['F3']:.3e}"


def test_regression_prediction_preserves_anchors(db, a_reg_anchor):
    """End-to-end: the 20-D linear regressor's prediction, when piped
    through the anchor-preserving assembler, preserves anchors exactly.
    Reconstruction accuracy is a separate concern — this test only
    checks the structural invariant."""
    nd_arr = db["nd"]
    vd_arr = db["vd"]
    dPgF_arr = db["dPgF"]
    N = len(nd_arr)

    Phi = np.array([
        feature_vec_20(nd_arr[i], vd_arr[i], dPgF_arr[i]) for i in range(N)
    ])
    nu34_hat = Phi @ a_reg_anchor

    worst = dict(F1=0.0, F2_abs=0.0, F2_rel=0.0, F3=0.0)
    for i in range(N):
        nu1, nu2, n3, n4 = assemble_anchor_preserving_coefficients(
            nd_arr[i], vd_arr[i], dPgF_arr[i], nu34_hat[i, 0], nu34_hat[i, 1]
        )
        m = anchor_error_metrics(
            nd_arr[i], vd_arr[i], dPgF_arr[i], nu1, nu2, n3, n4
        )
        for k in worst:
            worst[k] = max(worst[k], m[k])

    assert worst["F1"]     < TOL_F1,     f"regressor F1 = {worst['F1']:.3e}"
    assert worst["F2_abs"] < TOL_F2_ABS, f"regressor F2_abs = {worst['F2_abs']:.3e}"
    assert worst["F2_rel"] < TOL_F2_REL, f"regressor F2_rel = {worst['F2_rel']:.3e}"
    assert worst["F3"]     < TOL_F3,     f"regressor F3 = {worst['F3']:.3e}"


def test_legacy_fit_visibly_slips_anchors(db):
    """Guard against the legacy fit accidentally becoming
    anchor-preserving: if this test ever passes, the legacy semantics
    have changed and NB01 §4's contrast table would be meaningless."""
    nd_arr = db["nd"]
    vd_arr = db["vd"]
    dPgF_arr = db["dPgF"]
    nu14_legacy = db["nu14_legacy"]

    max_f2_abs = 0.0
    for i in range(len(nd_arr)):
        nu1, nu2, nu3, nu4 = nu14_legacy[i]
        m = anchor_error_metrics(
            nd_arr[i], vd_arr[i], dPgF_arr[i], nu1, nu2, nu3, nu4
        )
        max_f2_abs = max(max_f2_abs, m["F2_abs"])

    assert max_f2_abs > 1e-3, (
        f"legacy fit unexpectedly anchor-preserving: max|ΔVd| = "
        f"{max_f2_abs:.2e}. The legacy contrast is no longer meaningful."
    )


def test_solve_nu12_is_torch_autograd_compatible():
    """solve_nu12_from_nu34 must flow gradients through every input so
    NB03's differentiable design pipeline can drive an anchor-preserving
    Buchdahl forward model under autograd."""
    torch = pytest.importorskip("torch")

    nd = torch.tensor(1.5168, dtype=torch.float64, requires_grad=True)
    vd = torch.tensor(64.17, dtype=torch.float64, requires_grad=True)
    dPgF = torch.tensor(-0.0009, dtype=torch.float64, requires_grad=True)
    nu3 = torch.tensor(-0.03, dtype=torch.float64, requires_grad=True)
    nu4 = torch.tensor(-0.05, dtype=torch.float64, requires_grad=True)

    nu1, nu2 = solve_nu12_from_nu34(nd, vd, dPgF, nu3, nu4)
    loss = nu1 ** 2 + nu2 ** 2
    loss.backward()

    for name, t in [("nd", nd), ("vd", vd), ("dPgF", dPgF),
                    ("nu3", nu3), ("nu4", nu4)]:
        assert t.grad is not None, f"{name}.grad is None"
        assert t.grad.abs().item() > 0.0, f"{name}.grad is zero"


def test_feature_vec_20_scalar_and_vector_agree():
    """Scalar-input and batched-input paths of feature_vec_20 must
    produce identical values; the stacking dispatch must not silently
    reorder terms."""
    rng = np.random.default_rng(7)
    nd = rng.uniform(1.45, 1.90, size=8)
    vd = rng.uniform(20.0, 80.0, size=8)
    dPgF = rng.uniform(-0.01, 0.05, size=8)

    batched = feature_vec_20(nd, vd, dPgF)
    assert batched.shape == (8, 20)

    for i in range(8):
        single = feature_vec_20(nd[i], vd[i], dPgF[i])
        assert single.shape == (20,)
        assert np.allclose(single, batched[i], atol=1e-15, rtol=0.0)


def test_omega_at_d_line_is_zero():
    """Guard: omega(lambda_d) must be identically zero so that the
    constant term of the Buchdahl expansion carries nd exactly."""
    from dispersionlab.buchdahl import LAMBDA_D
    assert abs(buchdahl_omega(LAMBDA_D)) < 1e-15
