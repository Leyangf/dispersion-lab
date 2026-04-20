"""Sellmeier glass catalog loader (Optiland bundled database).

Reproduces NB04's filter exactly so that downstream code sees the same
glass set:

  * Sellmeier "formula 2" records only (6 coefficients)
  * Full spectral coverage of 0.365-2.3 um
  * ``VD_MIN < Vd < VD_MAX``
  * ``DPGF_MIN < dPgF < DPGF_MAX``

Set ``OPTILAND_DB_ROOT`` to override the auto-located database root.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import yaml

from .buchdahl import (
    DPGF_MAX,
    DPGF_MIN,
    LAMBDA_C,
    LAMBDA_D,
    LAMBDA_F,
    LAMBDA_g,
    SCHOTT_NORMAL_A,
    SCHOTT_NORMAL_B,
    VD_MAX,
    VD_MIN,
    WAVELENGTHS,
)


def sellmeier_formula_2(lam, sm):
    """Schott "formula 2" Sellmeier dispersion.

    Parameters
    ----------
    lam : float or numpy array
        Wavelength in micrometers.
    sm : 6-tuple of floats
        ``(B1, C1, B2, C2, B3, C3)``.
    """
    B1, C1, B2, C2, B3, C3 = sm
    wl2 = np.asarray(lam) ** 2
    return np.sqrt(
        1.0
        + B1 * wl2 / (wl2 - C1)
        + B2 * wl2 / (wl2 - C2)
        + B3 * wl2 / (wl2 - C3)
    )


def _default_db_root():
    env = os.environ.get("OPTILAND_DB_ROOT")
    if env:
        return Path(env)
    spec = importlib.util.find_spec("optiland")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "optiland not installed and OPTILAND_DB_ROOT not set"
        )
    return Path(spec.origin).parent / "database"


def _extract_record(yml_path):
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
    if lam_lo > 0.365 + 1e-6 or lam_hi < 2.3 - 1e-6:
        return None
    sm = (B1, C1, B2, C2, B3, C3)
    nd = float(sellmeier_formula_2(LAMBDA_D, sm))
    nF = float(sellmeier_formula_2(LAMBDA_F, sm))
    nC = float(sellmeier_formula_2(LAMBDA_C, sm))
    ng = float(sellmeier_formula_2(LAMBDA_g, sm))
    dn_FC = nF - nC
    if dn_FC < 1e-12:
        return None
    vd = (nd - 1.0) / dn_FC
    PgF = (ng - nF) / dn_FC
    dPgF = PgF - (SCHOTT_NORMAL_A - SCHOTT_NORMAL_B * vd)
    return dict(
        name=yml_path.stem,
        catalog=yml_path.parent.name,
        nd=nd, vd=vd, PgF=PgF, dPgF=dPgF,
        sellmeier=sm,
        n_truth=np.array(
            [float(sellmeier_formula_2(lam, sm)) for lam in WAVELENGTHS]
        ),
    )


def load_glass_catalog(db_root=None):
    """Return the filtered list of glass records.

    Each record is a dict with keys:
    ``name, catalog, nd, vd, PgF, dPgF, sellmeier, n_truth``.
    """
    root = Path(db_root) if db_root else _default_db_root()
    glass_root = root / "data-nk" / "glass"
    out = []
    for yml in glass_root.rglob("*.yml"):
        rec = _extract_record(yml)
        if rec is None:
            continue
        if not (VD_MIN < rec["vd"] < VD_MAX):
            continue
        if not (DPGF_MIN < rec["dPgF"] < DPGF_MAX):
            continue
        out.append(rec)
    return out
