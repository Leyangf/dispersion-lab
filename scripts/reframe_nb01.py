"""One-shot script: prepend and append framing markdown cells to the
hand-authored 01_model_glass_buchdahl.ipynb so its narrative aligns with
the rest of the repo (production = anchor-preserving K=4 α=1.818 via
dispersionlab; NB01 itself is retained as the legacy-construction
heritage reference).

Idempotent: detected via sentinels in cell source. Running twice is a
no-op.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks" / "01_model_glass_buchdahl.ipynb"

OPENING_SENTINEL = "<!-- NB01_ROLE_FRAMING_V1 -->"
CLOSING_SENTINEL = "<!-- NB01_CLOSING_RECONCILIATION_V1 -->"

OPENING_MD = """<!-- NB01_ROLE_FRAMING_V1 -->

> **Role in the project**
>
> This notebook is the **legacy-construction heritage reference**. It establishes the Buchdahl fitting, the `(K = 4, α = 1.818)` hyperparameter choice, and the Test A / Test B diagnostics under the *legacy* coefficient assembly order — `(ν₁, ν₂)` solved from anchor differences first, then `(ν₃, ν₄)` LSQ-fit on the residual.
>
> **The canonical forward model** consumed by every downstream notebook (NB02 / NB03 / NB04 / NB05) is the **anchor-preserving construction** — derived in report §1.4 and shipped via `dispersionlab.solve_nu12_from_nu34` / `fit_anchor_preserving_coefficients`. It adopts the same `(K = 4, α = 1.818)` as **production baseline** for compactness and classical-Buchdahl compatibility (report §1.6).
>
> **An accuracy-frontier sweep** (report §1.6, `scripts/build_01_anchor_model.py`) shows that under the anchor-preserving family the overfitting threshold is much higher than the legacy `K = 5` ceiling this notebook documents — p95 Test A hold error keeps dropping to ~4.6·10⁻⁴ at `K = 8, α = 1.26`. That frontier is **not** the production path (its downstream Conrady gain is zero because Phase 2's `q(λ)` envelope preserves anchor-dependent metrics by construction); it is recorded as the candidate forward model for a future NIR / broadband extension.
>
> **How to read this notebook** — treat the α / order sweeps below as the **legacy-construction** analysis that *selected* the hyperparameters later adopted as production. The canonical anchor-preserving derivation and its Claim F / G downstream consequences live in report §1.4–§1.6 and NB04; the legacy-vs-anchor contrast at `(K = 4, α = 1.818)` is quantified in NB04 §5 and report §5.

---

"""

CLOSING_MD = """<!-- NB01_CLOSING_RECONCILIATION_V1 -->

---

## Reconciliation with the canonical anchor-preserving forward model

The α / order conclusions of this notebook — `K = 4, α = 1.818` — were adopted as the **production hyperparameters** for the anchor-preserving construction deployed across NB02 / NB03 / NB04 / NB05 (see report §1.4 and §1.6). The same pipeline is *not* re-run here under anchor-preserving construction; this notebook stays in legacy form as the heritage reference.

The anchor-preserving analogues of the key tests in this notebook are:

| This notebook (legacy) | Canonical anchor-preserving path |
|---|---|
| Per-glass Buchdahl fit quality | `dispersionlab.fit_anchor_preserving_coefficients` + `tests/test_anchor_delta.py` |
| `(n_d, V_d, ΔP_{g,F}) → (ν₃, ν₄)` regression | `data/regression_buchdahl_anchor_nu34_20dim_opt.npy` (used by NB02 / NB03) |
| Test A (wavelength leave-one-out) at `α = 1.818, K = 4` | Stage A of `scripts/build_01_anchor_model.py` — shows the anchor family's hold error lower than legacy at the same K |
| α sweep, order sweep | Stage A / B of `scripts/build_01_anchor_model.py` — identifies `K = 8, α = 1.26` as accuracy frontier (report §1.6, `data/sweep_tmp/`) |
| **Downstream validation** | NB02 Conrady ranking (Spearman +1.000, median \\|ΔS\\| 0.54 μm) / NB03 J·Σ·Jᵀ vs MC (1.5 %) / NB04 Claim F / G + MLP = Linear negative result |

For anything the thesis cites as "the model's behaviour on real downstream tasks," the authoritative source is NB02 / NB03 / NB04 / NB05, all running the anchor-preserving forward model.
"""


def _md_cell(source_text, cell_id=None):
    return dict(
        cell_type="markdown",
        metadata={},
        id=cell_id or str(uuid.uuid4())[:8],
        source=source_text.splitlines(keepends=True),
    )


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    has_opening = any(
        c.get("cell_type") == "markdown"
        and any(OPENING_SENTINEL in s for s in c.get("source", []))
        for c in cells
    )
    has_closing = any(
        c.get("cell_type") == "markdown"
        and any(CLOSING_SENTINEL in s for s in c.get("source", []))
        for c in cells
    )

    changed = False
    if not has_opening:
        cells.insert(0, _md_cell(OPENING_MD, cell_id="nb01_role_v1"))
        changed = True
        print("  prepended opening framing cell")
    else:
        print("  opening framing already present; skipping")
    if not has_closing:
        cells.append(_md_cell(CLOSING_MD, cell_id="nb01_reconcil_v1"))
        changed = True
        print("  appended closing reconciliation cell")
    else:
        print("  closing reconciliation already present; skipping")

    if changed:
        NB_PATH.write_text(
            json.dumps(nb, indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  wrote {NB_PATH.name} — cells: {len(cells)}")
    else:
        print(f"  {NB_PATH.name} unchanged")


if __name__ == "__main__":
    main()
