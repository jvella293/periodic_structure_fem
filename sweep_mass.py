"""Run main.py's validation case at several contact masses m1.

main.py keeps every parameter as a module-level global and reads those
globals inside main(), so a sweep is just: rebind main.m1, call main.main().
Everything else (cache hashing, figure tags, validation_points.csv rows)
already keys off m1, so each mass gets its own cache entry, its own figures
and its own row in the results CSV.

Usage
-----
    python sweep_mass.py                       # the three masses below
    python sweep_mass.py 65 77.69 104.74       # or any list you pass in
"""

from __future__ import annotations

import sys
import traceback

import matplotlib
matplotlib.use("Agg")          # batch run: no windows, figures still written

import main


MASSES = [65.0, 77.6923883751325, 104.738757499957]


def run_one(m1: float) -> None:
    main.m1 = m1
    if main.model == "2dof":
        main.m2 = main.mu * m1          # keep the derived mass consistent
    main.SHOW_FIGURES = False           # never block the sweep on plt.show()
    main.main()


def main_sweep(masses: list[float]) -> None:
    for i, m1 in enumerate(masses, 1):
        print("\n" + "=" * 70)
        print(f"[{i}/{len(masses)}]  m1 = {m1:g} kg")
        print("=" * 70)
        try:
            run_one(m1)
        except Exception:
            print(f"\n!!! run failed for m1 = {m1:g}:")
            traceback.print_exc()


if __name__ == "__main__":
    args = [float(a) for a in sys.argv[1:]]
    main_sweep(args or MASSES)
