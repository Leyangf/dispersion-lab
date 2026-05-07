"""ThermalDispersionLabMaterial — invariant + autograd regression tests.

Locks in the structural property that the thermal extension collapses to
DispersionLab's anchor-preserving baseline at T = T_ref, and that the
thermal correction is linear in (T - T_ref). Adds a torch-autograd flow
test verifying gradients reach (nd, Vd, dPgF, dn_dT) and T from a single
n(λ, T) backward pass.
"""
from __future__ import annotations

import pytest

from dispersionlab.buchdahl import (
    LAMBDA_C, LAMBDA_D, LAMBDA_F, LAMBDA_g,
    SCHOTT_NORMAL_A, SCHOTT_NORMAL_B,
)
from dispersionlab.optiland_adapter import (
    DispersionLabMaterial,
    ThermalDispersionLabMaterial,
)


# Tolerances mirror tests/test_anchor_delta.py.
TOL_F1     = 1e-14
TOL_F2_REL = 1e-12
TOL_F3     = 1e-12


def test_thermal_at_T_ref_equals_base():
    """T = T_ref → no thermal contribution → output equals the bare
    DispersionLab base material."""
    base = DispersionLabMaterial(nd=1.5168, Vd=64.17, dPgF=0.0)
    therm = ThermalDispersionLabMaterial(
        nd=1.5168, Vd=64.17, dPgF=0.0, dn_dT=2.0e-6, T_ref=20.0,
    )
    assert abs(float(therm.n(0.5876, T=20.0)[0]) - float(base.n(0.5876)[0])) < 1e-12


def test_thermal_linearity_in_T():
    """At fixed (nd, Vd, dPgF), n(T1) - n(T2) = dn_dT * (T1 - T2) exactly."""
    therm = ThermalDispersionLabMaterial(
        nd=1.5168, Vd=64.17, dPgF=0.0, dn_dT=2.0e-6, T_ref=20.0,
    )
    n70 = float(therm.n(0.5876, T=70.0)[0])
    n20 = float(therm.n(0.5876, T=20.0)[0])
    assert abs((n70 - n20) - 2.0e-6 * 50.0) < 1e-12


def test_thermal_preserves_dispersionlab_anchors_at_T_ref():
    """The three structural anchor invariants must survive the thermal
    subclass at T = T_ref. Sampled at the d/F/C/g lines."""
    nd, Vd, dPgF = 1.5168, 64.17, 0.0
    m = ThermalDispersionLabMaterial(
        nd, Vd, dPgF, dn_dT=2.0e-6, T_ref=20.0,
    )

    n_d = float(m.n(LAMBDA_D, T=20.0)[0])
    n_F = float(m.n(LAMBDA_F, T=20.0)[0])
    n_C = float(m.n(LAMBDA_C, T=20.0)[0])
    n_g = float(m.n(LAMBDA_g, T=20.0)[0])

    # Anchor 1: n(λ_d) = nd
    assert abs(n_d - nd) < TOL_F1

    # Anchor 2: V_d preserved
    Vd_implied = (n_d - 1.0) / (n_F - n_C)
    assert abs(Vd_implied - Vd) / Vd < TOL_F2_REL

    # Anchor 3: dPgF preserved
    PgF_normal = SCHOTT_NORMAL_A - SCHOTT_NORMAL_B * Vd
    PgF_implied = (n_g - n_F) / (n_F - n_C)
    dPgF_implied = PgF_implied - PgF_normal
    assert abs(dPgF_implied - dPgF) < TOL_F3


def test_thermal_autograd_through_T():
    """Gradients must flow from a single n(λ, T) scalar backward pass to
    every learnable parameter (nd, Vd, dPgF, dn_dT) AND to T itself.
    This is the structural claim for opto-thermal joint design."""
    torch = pytest.importorskip("torch")
    import optiland.backend as be

    be.set_backend("torch")
    be.grad_mode.enable()
    try:
        m = ThermalDispersionLabMaterial(
            nd=1.5168, Vd=64.17, dPgF=0.0, dn_dT=2.0e-6, T_ref=20.0,
        )
        T = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)

        n_val = m.n(0.5876, T=T)
        # Sum to scalar (n_val is shape (1,) from be.array([nd]) propagation)
        n_val.sum().backward()

        assert T.grad is not None and T.grad.abs().item() > 0, (
            "T.grad missing — temperature did not participate in autograd"
        )
        for name, t in [("nd", m.nd), ("Vd", m.Vd),
                        ("dPgF", m.dPgF), ("dn_dT", m.dn_dT)]:
            assert t.grad is not None, f"{name}.grad is None"
            assert t.grad.abs().item() > 0, f"{name}.grad is zero"
    finally:
        be.grad_mode.disable()
        be.set_backend("numpy")
