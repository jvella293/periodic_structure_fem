"""Build the Floquet-comparison figure from the points you have run.

Reads validation_points.csv — which main.py appends to on every run — and
plots the measured Re(lambda) against the Floquet reference. Nothing is
simulated here, so this is instant and safe to re-run after every new point.

Floquet reference (optional)
----------------------------
Export a two-column CSV from the Fortran code, header optional:

    m1,Re_lambda
    80.0,-0.0123
    81.0,-0.0098

and save it as floquet_rates.csv. Without it the measured points are
plotted alone, which is still a usable figure.

Usage
-----
    python plot_validation.py                    # all recorded points
    python plot_validation.py --min-m1 80 --max-m1 95
    python plot_validation.py --exclude-inconsistent
    python plot_validation.py --x Vc             # plot against V/c instead
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import main as mn
import thesis_plots as tp

RESULTS_CSV = Path("validation_points.csv")
FLOQUET_CSV = Path("floquet_rates.csv")


def _float(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_points(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_floquet(path: Path):
    if not path.exists():
        return None, None
    data = np.genfromtxt(path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    data = data[~np.isnan(data).any(axis=1)]
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", default="m1", choices=["m1", "Vc", "V"],
                    help="sweep coordinate on the x axis")
    ap.add_argument("--min-m1", type=float, default=None)
    ap.add_argument("--max-m1", type=float, default=None)
    ap.add_argument("--exclude-inconsistent", action="store_true",
                    help="drop points whose split-half check failed")
    ap.add_argument("--stem", default="fig_validation")
    ap.add_argument("--no-tongue", action="store_true",
                    help="do not shade the tongue band from main.py")
    args = ap.parse_args()

    rows = load_points(RESULTS_CSV)
    if not rows:
        print(f"No points in {RESULTS_CSV}. Run main.py on a few masses first "
              f"(RECORD_RESULTS must be True).")
        return

    keep = []
    for r in rows:
        rate = _float(r.get("rate"))
        if np.isnan(rate):
            continue
        m1 = _float(r.get("m1"))
        if args.min_m1 is not None and m1 < args.min_m1:
            continue
        if args.max_m1 is not None and m1 > args.max_m1:
            continue
        if args.exclude_inconsistent and r.get("consistent", "").lower() != "true":
            continue
        keep.append(r)

    if not keep:
        print("No points survived the filters.")
        return

    x = np.array([_float(r[args.x]) for r in keep])
    rate = np.array([_float(r["rate"]) for r in keep])
    err = np.array([_float(r.get("err"), 0.0) for r in keep])
    order = np.argsort(x)
    x, rate, err = x[order], rate[order], err[order]

    print(f"{len(keep)} point{'s' if len(keep) != 1 else ''} from {RESULTS_CSV}:")
    for r in [keep[i] for i in order]:
        flag = "" if r.get("consistent", "").lower() == "true" else "   [halves disagree]"
        print(f"  {args.x} = {_float(r[args.x]):9.4f}   "
              f"Re(lambda) = {_float(r['rate']):+.4f} +/- {_float(r.get('err'), 0.0):.4f}"
              f"   t_max = {_float(r.get('t_max')):g} s{flag}")

    fq_x, fq_rate = load_floquet(FLOQUET_CSV)
    if fq_x is None:
        print(f"\n({FLOQUET_CSV} not found — plotting measured points alone.)")
    elif args.x != "m1":
        print(f"\n(Floquet CSV is in m1; skipping overlay for --x {args.x}.)")
        fq_x, fq_rate = None, None

    tongue = None if (args.no_tongue or args.x != "m1") else mn.TONGUE

    tp.use_thesis_style()
    labels = {
        "m1": r"Contact mass $m_1$ [kg]",
        "Vc": r"Normalised speed $V/c$ [-]",
        "V": r"Speed $V$ [m s$^{-1}$]",
    }
    _, pdf = tp.fig_stability_sweep(
        x, rate, err if np.any(err > 0) else None,
        floquet_m1=fq_x, floquet_rate=fq_rate,
        tongue=tongue, stem=args.stem, xlabel=labels[args.x],
    )
    print(f"\nsaved {pdf}")

    if fq_x is not None and fq_x.size >= 2:
        fq_at = np.interp(x, fq_x, fq_rate)
        _, pdf2 = tp.fig_sweep_error(x, rate, fq_at, stem=f"{args.stem}_residual")
        print(f"saved {pdf2}")
        print(f"RMS discrepancy vs Floquet: "
              f"{float(np.sqrt(np.nanmean((rate - fq_at) ** 2))):.5f} 1/s")


if __name__ == "__main__":
    main()
