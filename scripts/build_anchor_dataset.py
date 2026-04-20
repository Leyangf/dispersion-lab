"""Generate anchor-preserving per-glass coefficient dataset and the
cross-glass regression matrix.

Run from the project root::

    uv run python scripts/build_anchor_dataset.py

Outputs:

  data/buchdahl_coeffs_per_glass.npz
      name, catalog           (N,)          unicode arrays
      nd, vd, PgF, dPgF       (N,)          catalog scalars
      sellmeier               (N, 6)        (B1, C1, B2, C2, B3, C3)
      n_truth                 (N, 17)       Sellmeier n at WAVELENGTHS
      nu14_anchor             (N, 4)        anchor-preserving LSQ fit
      nu14_legacy             (N, 4)        legacy fit (NB01 §4 contrast)
      wavelengths             (17,)         training grid
      alpha                   scalar        Buchdahl alpha used

  data/regression_buchdahl_anchor_nu34_20dim_opt.npy
      (20, 2)                 linear map (nd, Vd, dPgF) -> (nu3, nu4)
                              trained on the full catalog, same
                              convention as the legacy
                              regression_buchdahl_nu34_20dim_opt.npy
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

from dispersionlab.buchdahl import (
    ALPHA,
    WAVELENGTHS,
    feature_vec_20,
    fit_anchor_preserving_coefficients,
    fit_legacy_coefficients,
)
from dispersionlab.catalog import load_glass_catalog


def main():
    t0 = time.time()
    print("Loading glass catalog from Optiland database...")
    glasses = load_glass_catalog()
    N = len(glasses)
    if N == 0:
        raise RuntimeError("No glasses loaded — check OPTILAND_DB_ROOT")
    print(f"  {N} glasses after filter "
          f"(VD 10..100, dPgF -0.02..0.10, full 0.365-2.3 um coverage)")

    names     = np.array([g["name"] for g in glasses])
    catalogs  = np.array([g["catalog"] for g in glasses])
    nd_arr    = np.array([g["nd"] for g in glasses])
    vd_arr    = np.array([g["vd"] for g in glasses])
    PgF_arr   = np.array([g["PgF"] for g in glasses])
    dPgF_arr  = np.array([g["dPgF"] for g in glasses])
    sm_arr    = np.array([g["sellmeier"] for g in glasses])
    ntruth    = np.array([g["n_truth"] for g in glasses])

    print("Fitting per-glass anchor-preserving coefficients...")
    nu14_anchor = np.array([
        fit_anchor_preserving_coefficients(
            g["nd"], g["vd"], g["dPgF"], g["n_truth"]
        )
        for g in glasses
    ])

    print("Fitting per-glass legacy coefficients (for NB01 §4 contrast)...")
    nu14_legacy = np.array([
        fit_legacy_coefficients(
            g["nd"], g["vd"], g["dPgF"], g["n_truth"]
        )
        for g in glasses
    ])

    data_dir = REPO / "data"
    data_dir.mkdir(exist_ok=True)
    out_npz = data_dir / "buchdahl_coeffs_per_glass.npz"
    np.savez(
        out_npz,
        name=names, catalog=catalogs,
        nd=nd_arr, vd=vd_arr, PgF=PgF_arr, dPgF=dPgF_arr,
        sellmeier=sm_arr, n_truth=ntruth,
        nu14_anchor=nu14_anchor, nu14_legacy=nu14_legacy,
        wavelengths=WAVELENGTHS, alpha=ALPHA,
    )
    print(f"  Wrote {out_npz} ({out_npz.stat().st_size/1024:.1f} KB)")

    # Train the production (nd, Vd, dPgF) -> (nu3, nu4) map on ALL glasses,
    # matching the legacy convention used by NB02 / NB03 / NB05.
    print("Training anchor-preserving cross-glass regression (20-D, full catalog)...")
    Phi = np.array([
        feature_vec_20(g["nd"], g["vd"], g["dPgF"])
        for g in glasses
    ])
    Y = nu14_anchor[:, 2:]
    A_anchor, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    assert A_anchor.shape == (20, 2)

    out_npy = data_dir / "regression_buchdahl_anchor_nu34_20dim_opt.npy"
    np.save(out_npy, A_anchor)
    print(f"  Wrote {out_npy}  shape={A_anchor.shape}  alpha={ALPHA}")

    # Quick diagnostic: reconstructed ν3,ν4 error on the training set.
    nu34_hat = Phi @ A_anchor
    residual = nu34_hat - Y
    print(f"  training residual |nu3 - nu3_hat|: "
          f"max {np.abs(residual[:, 0]).max():.3e}, "
          f"mean {np.abs(residual[:, 0]).mean():.3e}")
    print(f"  training residual |nu4 - nu4_hat|: "
          f"max {np.abs(residual[:, 1]).max():.3e}, "
          f"mean {np.abs(residual[:, 1]).mean():.3e}")

    print(f"Done in {time.time()-t0:.1f} s.")


if __name__ == "__main__":
    main()
