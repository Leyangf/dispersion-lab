"""DispersionLab — shared dispersion-model primitives.

The anchor-preserving Buchdahl surrogate is the canonical forward model;
legacy constructions are exposed only for one-time contrast tables
(NB01 §4) and should not be used in downstream pipelines.
"""
from .buchdahl import (
    ALPHA,
    B_COUP,
    DPGF_MAX,
    DPGF_MIN,
    LAMBDA_C,
    LAMBDA_D,
    LAMBDA_F,
    LAMBDA_g,
    M_COUP,
    M_INV,
    SCHOTT_NORMAL_A,
    SCHOTT_NORMAL_B,
    VD_MAX,
    VD_MIN,
    WAVELENGTHS,
    anchor_error_metrics,
    assemble_anchor_preserving_coefficients,
    buchdahl_omega,
    feature_vec_20,
    fit_anchor_preserving_coefficients,
    fit_legacy_coefficients,
    reconstruct_n,
    solve_nu12_from_nu34,
)
from .catalog import load_glass_catalog, sellmeier_formula_2

__all__ = [
    "ALPHA",
    "B_COUP",
    "DPGF_MAX",
    "DPGF_MIN",
    "LAMBDA_C",
    "LAMBDA_D",
    "LAMBDA_F",
    "LAMBDA_g",
    "M_COUP",
    "M_INV",
    "SCHOTT_NORMAL_A",
    "SCHOTT_NORMAL_B",
    "VD_MAX",
    "VD_MIN",
    "WAVELENGTHS",
    "anchor_error_metrics",
    "assemble_anchor_preserving_coefficients",
    "buchdahl_omega",
    "feature_vec_20",
    "fit_anchor_preserving_coefficients",
    "fit_legacy_coefficients",
    "load_glass_catalog",
    "reconstruct_n",
    "sellmeier_formula_2",
    "solve_nu12_from_nu34",
]
