"""Diagnose WHY a growth-rate fit reports inconsistent halves.

Two candidate causes, distinguished by two tests:

  1. RESIDUAL TRANSIENT — the seed's broadband response has not died out by
     the start of the fit window. Symptom: Re(lambda) falls steadily as the
     window start is pushed later, heading toward some asymptote.
     Fix: raise SETTLE_PERIODS (free), or run longer.

  2. BEAT COMPARABLE TO t_max — the hull is modulated at a period of order
     the run length, so a straight-line log fit picks up whichever part of
     the beat it happens to sit on. Symptom: Re(lambda) oscillates in sign
     or magnitude as the window start moves, with no asymptote, AND the
     hull spectrum shows a peak at a period of order t_max.
     Fix: run long enough to cover several beat periods. Nothing else works.

Both are diagnosed WITHOUT resimulating.

Usage
-----
    python diagnose.py cache/run_3c5ab7862a1c.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import main as mn


def settle_sweep(t, F, V, spacing, hull_periods=4.0):
    """Re(lambda) as a function of how much of the start is discarded."""
    T_pass = spacing / V
    total = t[-1] - t[0]
    print(f"\n--- Test 1: window-start sweep (hull = {hull_periods:g} T_pass) ---")
    print(f"{'settle [T_pass]':>16} {'window start [s]':>17} {'window [s]':>11} "
          f"{'Re(lambda)':>12} {'floor':>9}  halves")
    rates = []
    for settle in (3.0, 6.0, 10.0, 15.0, 20.0, 25.0):
        if settle * T_pass > 0.75 * total:
            break
        g = mn.growth_rate(t, F, settle_periods=settle, hull_periods=hull_periods)
        if g["slope"] is None:
            print(f"{settle:>16.1f} {settle*T_pass:>17.2f} {'-':>11} "
                  f"{'too short':>12}")
            continue
        rates.append(g["slope"])
        print(f"{settle:>16.1f} {settle*T_pass:>17.2f} {g['t_win']:>11.2f} "
              f"{g['slope']:>+12.4f} {g['rate_floor']:>9.4f}  "
              f"{g['slope_first']:+.4f}/{g['slope_second']:+.4f}"
              f" [{'ok' if g['consistent'] else 'INCONSISTENT'}]")

    if len(rates) >= 3:
        r = np.array(rates)
        drift = abs(r[-1] - r[0])
        monotonic = np.all(np.diff(r) < 0) or np.all(np.diff(r) > 0)
        print(f"\n  spread across sweep = {r.min():+.4f} .. {r.max():+.4f} "
              f"(drift {drift:.4f})")
        if monotonic and drift > 0.3 * abs(r[0]):
            print("  => drifts monotonically: looks like RESIDUAL TRANSIENT. "
                  "Raise SETTLE_PERIODS and/or run longer.")
        elif drift > 0.5 * abs(np.mean(r)):
            print("  => wanders without settling: looks like BEAT contamination. "
                  "Only a longer run fixes this.")
        else:
            print("  => stable across window choices: the rate is trustworthy.")


def beat_spectrum(t, F, V, spacing, hull_periods=4.0, settle_periods=3.0):
    """Find the dominant modulation period of the log-hull."""
    g = mn.growth_rate(t, F, settle_periods=settle_periods,
                       hull_periods=hull_periods)
    if g["slope"] is None:
        print("\n--- Test 2: hull modulation --- (fit failed, skipping)")
        return

    th, eh = g["t_hull"], g["env_hull"]
    # remove the fitted exponential; what's left is the modulation
    resid = np.log(eh) - (g["intercept"] + g["slope"] * th)
    resid -= resid.mean()

    dt_h = float(np.median(np.diff(th)))
    n = resid.size
    spec = np.abs(np.fft.rfft(resid * np.hanning(n)))
    freq = np.fft.rfftfreq(n, dt_h)

    total = th[-1] - th[0]
    # ignore anything slower than the window itself (unresolvable)
    valid = freq > 1.0 / total
    if not valid.any():
        print("\n--- Test 2: hull modulation --- (window too short)")
        return
    order = np.argsort(spec[valid])[::-1][:3]
    f_valid, s_valid = freq[valid], spec[valid]

    print(f"\n--- Test 2: hull modulation spectrum "
          f"(window {total:.1f} s) ---")
    for i in order:
        period = 1.0 / f_valid[i]
        cycles = total / period
        note = ""
        if cycles < 3:
            note = "  <-- period comparable to run: FIT IS UNRELIABLE"
        print(f"  period {period:7.2f} s  ({cycles:5.1f} cycles in window)"
              f"  rel. power {s_valid[i]/s_valid.max():.2f}{note}")

    slowest = 1.0 / f_valid[order[0]]
    need = 5.0 * slowest + settle_periods * spacing / V
    print(f"\n  For ~5 beat cycles inside the fit window you need "
          f"t_max >~ {need:.0f} s.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--hull", type=float, default=4.0)
    args = ap.parse_args()

    d = np.load(args.path, allow_pickle=False)
    t, F = d["t"], d["contact_force"]
    V = d["V"].item()
    spacing = d["spacing"].item() if "spacing" in d.files else 10.0
    m1 = d["m1"].item() if "m1" in d.files else d["M_mass"].item()

    print(f"=== {args.path.name} ===")
    print(f"  m1 = {m1:g} kg, V = {V:.4f} m/s, "
          f"run length = {t[-1]-t[0]:.2f} s, {t.size} samples")

    saved = (mn.spacing, mn.V)
    mn.spacing, mn.V = spacing, V
    try:
        settle_sweep(t, F, V, spacing, hull_periods=args.hull)
        beat_spectrum(t, F, V, spacing, hull_periods=args.hull)
    finally:
        mn.spacing, mn.V = saved


if __name__ == "__main__":
    main()