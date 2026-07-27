"""Re-analyse cached runs with the upper-hull estimator — no resimulation.

Your existing cache/*.npz files already contain t and contact_force, so the
15 s run you paid an hour for can be re-fitted for free. Run this before
launching anything new.

Usage
-----
    python reanalyse.py                 # every run in cache/
    python reanalyse.py cache/run_ab12cd34ef56.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import main as mn


def reanalyse(path: Path) -> None:
    d = np.load(path, allow_pickle=False)
    t = d["t"]
    F = d["contact_force"]

    def scalar(key, default=None):
        return d[key].item() if key in d.files else default

    m1 = scalar("m1")
    V = scalar("V")
    spacing = scalar("spacing", 10.0)
    dt_run = scalar("dt")
    t_max = scalar("t_max")
    n_cells = scalar("n_cells")

    # growth_rate() reads spacing and V from main's module globals, so point
    # them at THIS run's values rather than whatever main.py is set to now.
    mn.spacing, mn.V = spacing, V

    print(f"\n=== {path.name} ===")
    print(f"  m1 = {m1:g} kg, V = {V:.4f} m/s, dt = {dt_run:g}, "
          f"t_max = {t_max:g} s, n_cells = {n_cells:g}")

    for hull_periods in (2.0, 4.0, 8.0):
        g = mn.growth_rate(t, F, hull_periods=hull_periods)
        if g["slope"] is None:
            print(f"  hull={hull_periods:>4.1f} T_pass: {g['verdict']}")
            continue
        print(
            f"  hull={hull_periods:>4.1f} T_pass: "
            f"Re(lambda) = {g['slope']:+.4f} 1/s   "
            f"halves {g['slope_first']:+.4f}/{g['slope_second']:+.4f} "
            f"[{'ok' if g['consistent'] else 'INCONSISTENT'}]   "
            f"floor {g['rate_floor']:.4f}   {g['verdict']}"
        )

    # Is the run even long enough to sign the rate the tongue implies?
    if mn.TONGUE is not None and m1 is not None:
        lo, hi = mn.TONGUE
        pos = (m1 - lo) / (hi - lo)
        if 0.0 < pos < 1.0:
            centre = 0.5 * (lo + hi)
            f_unstable = V / spacing / mn.TONGUE_ORDER
            rate_max = 0.5 * (0.5 * (hi - lo) / centre) * (2 * np.pi * f_unstable)
            rate = rate_max * 2 * np.sqrt(pos * (1 - pos))
            print(f"  estimated Re(lambda) at this m1 ~ {rate:+.4f} 1/s "
                  f"=> expected hull growth x{np.exp(rate * t[-1]):.1f} over the run")


def main() -> None:
    args = sys.argv[1:]
    paths = [Path(a) for a in args] if args else sorted(Path("cache").glob("run_*.npz"))
    if not paths:
        print("No cached runs found in cache/.")
        return
    for p in paths:
        try:
            reanalyse(p)
        except Exception as exc:                      # noqa: BLE001
            print(f"\n=== {p.name} ===\n  could not re-analyse: {exc}")


if __name__ == "__main__":
    main()
