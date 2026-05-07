"""Optiland-compatible material wrapper for DispersionLab.

Subclasses optiland.materials.base.BaseMaterial. The 3-parameter physical
contract (nd, Vd, dPgF) of DispersionLab is preserved exactly; anchor
invariants (n(λ_d) = nd, V_d, dPgF) hold at floating-point precision
because the wrapper composes DispersionLab's existing anchor-preserving
forward path without modification.

ThermalDispersionLabMaterial extends with a linear-in-T correction:

    n(λ, T) = n_base(λ) + dn_dT * (T - T_ref)

At T = T_ref the thermal contribution is exactly zero, so all anchor
invariants are inherited. See tests/test_thermal_anchor.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import optiland.backend as be
from optiland.materials.base import BaseMaterial

from .buchdahl import (
    buchdahl_omega,
    feature_vec_20,
    reconstruct_n,
    solve_nu12_from_nu34,
)

# Production 20-dim regression matrix mapping (nd, Vd, dPgF) features -> (nu3, nu4).
# Located relative to the package install root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_M_NU34_NPY = _REPO_ROOT / "data" / "regression_buchdahl_anchor_nu34_20dim_opt.npy"


class DispersionLabMaterial(BaseMaterial):
    """Anchor-preserving Buchdahl glass model, Optiland-compatible.

    Differentiable in (nd, Vd, dPgF). When optiland.backend grad_mode is
    enabled, these three tensors are leaf nn.Parameter candidates and
    gradients flow through the full forward path: feature_vec_20 →
    regression matmul → solve_nu12_from_nu34 → reconstruct_n.
    """

    # Class-level constant. Loaded once at import. Cast to active backend
    # at use time via be.cast(...) so it never receives requires_grad.
    _M_NU34_NP: np.ndarray = np.load(_M_NU34_NPY)  # (20, 2) float64

    def __init__(self, nd: float, Vd: float, dPgF: float):
        super().__init__()
        self.nd = be.array([nd])
        self.Vd = be.array([Vd])
        self.dPgF = be.array([dPgF])

    def _calculate_n(self, wavelength, **kwargs):
        # The anchor-preserving forward path. **kwargs is accepted but ignored
        # at this layer — subclasses may consume kwargs (e.g. temperature).
        feat = feature_vec_20(self.nd, self.Vd, self.dPgF)  # (1, 20)
        M = be.cast(self._M_NU34_NP)                         # (20, 2), no grad
        nu34 = feat @ M                                      # (1, 2)
        nu3, nu4 = nu34[..., 0], nu34[..., 1]
        nu1, nu2 = solve_nu12_from_nu34(self.nd, self.Vd, self.dPgF, nu3, nu4)

        omega = buchdahl_omega(be.array(wavelength))
        return reconstruct_n(self.nd, nu1, nu2, nu3, nu4, omega)

    def _calculate_k(self, wavelength, **kwargs):
        # Purely refractive — no absorption.
        return be.zeros_like(be.array(wavelength))

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "nd": _scalar(self.nd),
            "Vd": _scalar(self.Vd),
            "dPgF": _scalar(self.dPgF),
        })
        return d

    @classmethod
    def from_dict(cls, data):
        return cls(data["nd"], data["Vd"], data["dPgF"])


class ThermalDispersionLabMaterial(DispersionLabMaterial):
    """DispersionLab glass with linear thermo-optic correction.

        n(λ, T) = n_base(λ) + dn_dT * (T - T_ref)

    Differentiable in (nd, Vd, dPgF, dn_dT) and through T. Anchor invariants
    are preserved exactly at T = T_ref since the thermal term vanishes.
    """

    def __init__(
        self,
        nd: float,
        Vd: float,
        dPgF: float,
        dn_dT: float,
        T_ref: float = 20.0,
    ):
        super().__init__(nd, Vd, dPgF)
        self.dn_dT = be.array([dn_dT])
        self.T_ref = T_ref  # plain float — never differentiated

    def n(self, wavelength, T=None, **kwargs):
        """Override the public path to bypass the parent's cache when T is
        provided.

        Reason: BaseMaterial.n() builds its cache key as
            tuple(sorted(kwargs.items()))
        which raises if any kwarg value is a multi-element torch tensor
        (tensors have no __lt__). When T is None we go through the cached
        path normally; when T is provided we recompute every call.
        """
        if T is None:
            return super().n(wavelength, **kwargs)
        return self._calculate_n(wavelength, T=T, **kwargs)

    def _calculate_n(self, wavelength, T=None, **kwargs):
        n_base = super()._calculate_n(wavelength, **kwargs)
        if T is None:
            return n_base
        dT = be.array(T) - self.T_ref
        return n_base + self.dn_dT * dT

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "dn_dT": _scalar(self.dn_dT),
            "T_ref": float(self.T_ref),
        })
        return d

    @classmethod
    def from_dict(cls, data):
        return cls(
            data["nd"], data["Vd"], data["dPgF"],
            data["dn_dT"], data.get("T_ref", 20.0),
        )


def _scalar(x):
    """Extract a Python float from a 0-d or 1-element backend tensor."""
    if hasattr(x, "item"):
        return float(x.item() if x.ndim == 0 else x[0].item())
    return float(np.asarray(x).reshape(-1)[0])
