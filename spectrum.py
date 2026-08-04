"""Spectral analysis of the perturbation response — what frequencies are here?

Why not just np.fft.rfft(F)
---------------------------
The signal is exponentially growing (or decaying). An exponential envelope
convolves the spectrum with a Lorentzian of half-width Re(lambda)/(2*pi),
smearing every line. So the fitted exponential is divided out first,
leaving a nearly stationary signal with sharp lines.

The Floquet comb
----------------
A Floquet solution is x(t) = exp(lambda*t) * p(t) with p periodic at the
support-passing period T = L/V. Expanding p as a Fourier series,

    x(t) = sum_n  c_n * exp(lambda*t) * exp(i*n*omega_pass*t)

so each solution contributes a COMB of lines at +/-f0 + n*f_pass, spaced
by f_pass, where f0 = |Im(lambda)|/(2*pi).

Several Floquet solutions can be excited at once, giving SEVERAL
interleaved combs with different offsets. This module extracts them
iteratively rather than assuming a single one.

IMPORTANT — why "two lines sum to f_pass" proves nothing
--------------------------------------------------------
Every comb with offset f0 automatically contains the lines f0 and
(f_pass - f0), and those always sum to exactly f_pass, whatever f0 is.
A sum test alone is therefore satisfied by ANY Floquet solution and is not
evidence of a combination resonance.

The non-degenerate test requires the two frequencies to coincide with two
DISTINCT NATURAL MODES of the oscillator:

    f0 ~ f_1   AND   (p*f_pass - f0) ~ f_2 ,   f_1 != f_2

That is what combination_test() checks.

Usage
-----
    python spectrum.py                          # newest cached run
    python spectrum.py cache/run_abc123def456.npz
    python spectrum.py --signal z1 --fmax 6
    python spectrum.py --all-signals
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import main as mn


# ----------------------------------------------------------------------
# Spectrum
# ----------------------------------------------------------------------

def detrended_spectrum(t, x, rate=None, settle_periods=3.0, T_pass=None,
                       pad_factor=4):
    """Amplitude spectrum of x(t) with any exponential trend removed."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    if settle_periods > 0 and T_pass:
        keep = t > settle_periods * T_pass
        t, x = t[keep], x[keep]
    if t.size < 16:
        raise ValueError("too few samples for a spectrum")

    if rate is not None and np.isfinite(rate):
        x = x * np.exp(-rate * (t - t[0]))

    x = x - x.mean()
    n = x.size
    n_fft = int(2 ** np.ceil(np.log2(n * pad_factor)))
    dt = float(np.median(np.diff(t)))

    spec = np.abs(np.fft.rfft(x * np.hanning(n), n=n_fft))
    freq = np.fft.rfftfreq(n_fft, dt)
    peak = spec.max()
    return freq, (spec / peak if peak > 0 else spec)


def find_lines(freq, amp, fmax, threshold=0.02, min_sep_hz=0.02):
    """Locate spectral lines above a relative amplitude threshold."""
    from scipy.signal import find_peaks

    band = freq <= fmax
    f, a = freq[band], amp[band]
    df = float(np.median(np.diff(f)))
    distance = max(1, int(round(min_sep_hz / df)))
    idx, props = find_peaks(a, height=threshold, distance=distance)
    order = np.argsort(props["peak_heights"])[::-1]
    return f[idx][order], a[idx][order]


# ----------------------------------------------------------------------
# Comb extraction
# ----------------------------------------------------------------------

def fold(f: float, f_pass: float) -> float:
    """Fold a frequency into [0, f_pass/2] — its comb offset."""
    r = f % f_pass
    return min(r, f_pass - r)


def find_combs(lines, amps, f_pass, tol_hz=None, min_members=2, max_combs=4):
    """Extract interleaved Floquet combs from a line list.

    Repeatedly takes the strongest unassigned line, folds it to get an
    offset, claims every line matching that comb, and continues. Several
    Floquet solutions excited at once produce several combs, and assuming a
    single one makes the other's lines look like errors.

    Returns
    -------
    list of dict
        Each with ``offset``, ``ratio`` (offset / f_pass), ``members``
        (list of (f, amp, n, sign)), ``strength`` (summed amplitude), and
        ``kind`` (interpretation of the offset).
    """
    if tol_hz is None:
        tol_hz = 0.01 * f_pass

    remaining = list(zip(np.asarray(lines, float), np.asarray(amps, float)))
    combs = []

    while remaining and len(combs) < max_combs:
        remaining.sort(key=lambda p: p[1], reverse=True)
        f_seed = remaining[0][0]
        offset = fold(f_seed, f_pass)

        members, leftover = [], []
        for f, a in remaining:
            best_err, best = np.inf, None
            for sign in (+1, -1):
                n = round((f - sign * offset) / f_pass)
                pred = sign * offset + n * f_pass
                if pred < 0:
                    continue
                err = abs(f - pred)
                if err < best_err:
                    best_err, best = err, (int(n), sign)
            if best_err < tol_hz:
                members.append((f, a, best[0], best[1]))
            else:
                leftover.append((f, a))

        if len(members) >= min_members:
            combs.append({
                "offset": offset,
                "ratio": offset / f_pass,
                "members": members,
                "strength": float(sum(m[1] for m in members)),
                "kind": classify_offset(offset, f_pass),
            })
            remaining = leftover
        else:
            # Seed did not form a comb; drop it and continue.
            remaining = [p for p in remaining if p[0] != f_seed]

    combs.sort(key=lambda c: c["strength"], reverse=True)
    return combs


def classify_offset(offset: float, f_pass: float, tol=0.03) -> str:
    """Interpret a comb offset."""
    r = offset / f_pass
    if r < tol:
        return "1T harmonic (integer multiples of f_pass)"
    if abs(r - 0.5) < tol:
        return "2T subharmonic (half-integer multiples of f_pass)"
    if abs(r - 0.5) < 0.10:
        split = (0.5 - r) * f_pass
        return (f"near-2T: split {split:+.4f} Hz either side of f_pass/2 "
                "— characteristic of being just OUTSIDE a 2T tongue "
                "(inside, the pair locks onto f_pass/2 exactly)")
    return f"incommensurate, offset = {r:.4f} * f_pass"


# ----------------------------------------------------------------------
# Combination resonance — the non-degenerate test
# ----------------------------------------------------------------------

def oscillator_modes(m1, m2, k01, k12) -> np.ndarray:
    """Natural frequencies [Hz] with the string held rigid (upper bounds)."""
    K = np.array([[k01 + k12, -k12], [-k12, k12]])
    M = np.diag([m1, m2])
    w = np.sqrt(np.clip(np.sort(np.linalg.eigvals(np.linalg.solve(M, K)).real),
                        0.0, None))
    return w / (2 * np.pi)


def combination_test(combs, f_pass, f_modes, p=1, tol_rel=0.08):
    """Does a comb pair up two DISTINCT oscillator modes?

    A comb with offset f0 always contains f0 and (p*f_pass - f0), summing
    to p*f_pass by construction — so the sum proves nothing. The real
    question is whether those two frequencies coincide with two different
    natural modes of the oscillator.

    Note that f_modes are rigid-string UPPER BOUNDS; string compliance
    lowers the true modes, so a measured line slightly below a bound is
    expected, whereas one above it cannot be that mode.
    """
    results = []
    for comb in combs:
        f0 = comb["offset"]
        partner = p * f_pass - f0
        if partner <= 0:
            continue

        def match(f):
            errs = np.abs(f_modes - f) / np.maximum(f_modes, 1e-12)
            i = int(np.argmin(errs))
            return i, float(errs[i]), bool(f <= f_modes[i] * (1 + tol_rel))

        i0, e0, below0 = match(f0)
        i1, e1, below1 = match(partner)
        ok = (e0 < tol_rel and e1 < tol_rel and i0 != i1 and below0 and below1)
        results.append({
            "offset": f0, "partner": partner,
            "mode_of_offset": i0, "err_offset": e0,
            "mode_of_partner": i1, "err_partner": e1,
            "distinct_modes": i0 != i1,
            "verdict": ok,
        })
    return results


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

SIGNALS = {
    "F": ("contact_force", r"$F_{\mathrm{tr}}$"),
    "z1": ("z1", r"$z_1$"),
    "z2": ("z2", r"$z_2$"),
    "wc": ("u_point", r"$w_c$"),
}


def newest_cache() -> Path | None:
    files = sorted(Path("cache").glob("run_*.npz"),
                   key=lambda q: q.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _get(d, key, default=np.nan):
    return d[key].item() if key in d.files else default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path)
    ap.add_argument("--signal", default="F", choices=list(SIGNALS))
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--no-detrend", action="store_true")
    ap.add_argument("--stem", default=None)
    ap.add_argument("--all-signals", action="store_true")
    args = ap.parse_args()

    path = args.path or newest_cache()
    if path is None or not path.exists():
        print("No cached run found. Run main.py first.")
        return

    d = np.load(path, allow_pickle=False)
    t = d["t"]
    V = _get(d, "V")
    spacing = _get(d, "spacing", 10.0)
    f_pass = V / spacing
    T_pass = 1.0 / f_pass
    fmax = args.fmax if args.fmax is not None else 6.0 * f_pass

    key, label = SIGNALS[args.signal]
    if key not in d.files:
        print(f"'{key}' not in this cache file.")
        return

    mn.spacing, mn.V = spacing, V
    g = mn.growth_rate(t, d["contact_force"])
    rate = None if args.no_detrend else g.get("slope")

    m1, m2 = _get(d, "m1"), _get(d, "m2")
    k01, k12 = _get(d, "k01"), _get(d, "k12")

    print(f"=== {path.name} ===")
    print(f"  m1 = {m1:g} kg, m2 = {m2:g} kg, V = {V:.4f} m/s, "
          f"run = {t[-1]-t[0]:.1f} s")
    print(f"  f_pass = {f_pass:.4f} Hz,  f_pass/2 = {f_pass/2:.4f} Hz")
    print(f"  Re(lambda) = {rate:+.4f} 1/s (divided out)" if rate is not None
          else "  no detrending applied")
    print(f"  frequency resolution = {1.0/(t[-1]-t[0]):.4f} Hz")

    freq, amp = detrended_spectrum(t, d[key], rate=rate, T_pass=T_pass)
    lines, line_amps = find_lines(freq, amp, fmax, threshold=args.threshold)

    combs = find_combs(lines, line_amps, f_pass)
    print(f"\n--- {len(combs)} Floquet comb(s) found in {label} "
          f"(lines above {args.threshold:.0%} of peak, up to {fmax:.2f} Hz) ---")

    for c_i, comb in enumerate(combs, 1):
        print(f"\n  Comb {c_i}:  f0 = {comb['offset']:.4f} Hz "
              f"= {comb['ratio']:.4f} * f_pass   (strength {comb['strength']:.2f})")
        print(f"    {comb['kind']}")
        print(f"    {'f [Hz]':>9} {'amp':>7}   position")
        for f, a, n, sign in sorted(comb["members"], key=lambda mm: -mm[1]):
            print(f"    {f:>9.4f} {a:>7.3f}   "
                  f"{'+' if sign > 0 else '-'}f0 {n:+d}*f_pass")

    if len(combs) > 1:
        d01 = abs(combs[0]["offset"] - combs[1]["offset"])
        print(f"\n  NOTE: {len(combs)} combs coexist — several Floquet solutions "
              f"are excited at once.")
        print(f"  Their offsets differ by {d01:.4f} Hz; each has its OWN "
              f"Re(lambda), and the")
        print(f"  envelope fit returns whichever dominates. To separate them, "
              f"band-pass")
        print(f"  around each comb and fit the rate of each band separately.")

    # --- combination resonance, tested properly ---
    f_modes = oscillator_modes(m1, m2, k01, k12)
    p = getattr(mn, "TONGUE_P", 1)
    print(f"\n--- Combination-resonance test (p = {p}) ---")
    print(f"  Rigid-string modes (UPPER bounds): "
          f"f_1 = {f_modes[0]:.4f} Hz, f_2 = {f_modes[1]:.4f} Hz")
    print("  A comb always contains f0 and (p*f_pass - f0), which sum to")
    print("  p*f_pass by construction — so a sum test alone proves nothing.")
    print("  The real question: do those two land on two DISTINCT modes?\n")

    for c_i, (comb, res) in enumerate(
            zip(combs, combination_test(combs, f_pass, f_modes, p=p)), 1):
        print(f"  Comb {c_i}: f0 = {res['offset']:.4f}, "
              f"partner = {res['partner']:.4f} Hz")
        print(f"    f0      closest to mode {res['mode_of_offset']+1} "
              f"({f_modes[res['mode_of_offset']]:.4f} Hz), "
              f"relative error {res['err_offset']*100:.1f}%")
        print(f"    partner closest to mode {res['mode_of_partner']+1} "
              f"({f_modes[res['mode_of_partner']]:.4f} Hz), "
              f"relative error {res['err_partner']*100:.1f}%")
        print(f"    => {'CONSISTENT with a combination resonance'
                       if res['verdict'] else
                       'NOT a clean combination resonance '
                       '(see errors / distinctness above)'}")

    # --- figure ---
    import thesis_plots as tp
    tp.use_thesis_style()
    stem = args.stem or f"fig_spectrum_{args.signal}"
    offsets = [c["offset"] for c in combs]

    if args.all_signals:
        series = []
        for name in ("F", "z1", "z2"):
            k, lab = SIGNALS[name]
            if k in d.files:
                fq, am = detrended_spectrum(t, d[k], rate=rate, T_pass=T_pass)
                series.append((fq, am, lab))
        _, pdf = tp.fig_spectrum_multi(series, f_pass=f_pass, fmax=fmax,
                                       offsets=offsets, stem=stem)
    else:
        _, pdf = tp.fig_spectrum(freq, amp, f_pass=f_pass, fmax=fmax,
                                 offsets=offsets, lines=lines,
                                 line_amps=line_amps, label=label, stem=stem)
    print(f"\nsaved {pdf}")


if __name__ == "__main__":
    main()
