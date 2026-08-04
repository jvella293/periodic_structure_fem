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


def find_combs(lines, amps, f_pass, tol_hz=None, min_members=2, max_combs=4,
               resolution=0.0):
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
                "kind": classify_offset(offset, f_pass, resolution),
            })
            remaining = leftover
        else:
            # Seed did not form a comb; drop it and continue.
            remaining = [p for p in remaining if p[0] != f_seed]

    combs.sort(key=lambda c: c["strength"], reverse=True)
    return combs


def classify_offset(offset: float, f_pass: float, resolution: float = 0.0) -> str:
    """Interpret a comb offset, using the frequency resolution as the yardstick.

    Whether an offset counts as "exactly f_pass/2" depends on how well the
    run resolves frequency: a 0.06 Hz deviation is noise in a 10 s run and
    a hard fact in a 150 s one. So the test compares the deviation against
    the resolution, not against an arbitrary fixed percentage.
    """
    r = offset / f_pass
    dev_hz = abs(offset - f_pass / 2)
    tol_hz = max(3.0 * resolution, 0.002 * f_pass)

    if offset < tol_hz:
        return "1T harmonic (lines at integer multiples of f_pass)"
    if dev_hz < tol_hz:
        return (f"2T subharmonic — locked to f_pass/2 within {tol_hz*1e3:.1f} mHz "
                "(consistent with being INSIDE a 2T tongue)")
    if dev_hz < 0.12 * f_pass:
        n_res = dev_hz / resolution if resolution > 0 else np.inf
        return (f"near-2T but NOT locked: the pair sits {dev_hz:.4f} Hz either "
                f"side of f_pass/2 ({n_res:.0f}x the frequency resolution). "
                "A resolved split means the exponents are still complex — "
                "i.e. just OUTSIDE the 2T tongue.")
    return f"incommensurate, offset = {r:.4f} * f_pass"


# ----------------------------------------------------------------------
# Combination resonance — the non-degenerate test
# ----------------------------------------------------------------------

def line_amplitude(freq, amp, f_target, window_hz):
    """Peak amplitude of a spectrum within +/-window_hz of f_target."""
    band = np.abs(freq - f_target) <= window_hz
    return float(amp[band].max()) if np.any(band) else 0.0


def mode_shape_test(comb, spectra, f_pass, p=1, resolution=0.0):
    """Do the two lines of a comb belong to DIFFERENT modes?

    This is the test that actually discriminates. A single mode at f0,
    parametrically modulated, produces lines at f0 and (p*f_pass - f0) —
    exactly the same positions as two modes in combination. Line positions
    therefore cannot decide between them.

    What does decide it is the mode shape. The ratio |z2|/|z1| is a
    property of the mode, so:

      * one mode + sidebands  -> the ratio is the SAME at every comb line
      * two distinct modes    -> the ratio DIFFERS between f0 and its partner

    Parameters
    ----------
    spectra : dict
        ``{"z1": (freq, amp), "z2": (freq, amp)}``.

    Returns
    -------
    dict or None
        None if z1/z2 spectra are unavailable.
    """
    if "z1" not in spectra or "z2" not in spectra:
        return None

    f1, a1 = spectra["z1"]
    f2, a2 = spectra["z2"]
    window = max(3.0 * resolution, 0.004 * f_pass)

    f0 = comb["offset"]
    partner = p * f_pass - f0
    if partner <= 0:
        return None

    def ratio(f):
        z1 = line_amplitude(f1, a1, f, window)
        z2 = line_amplitude(f2, a2, f, window)
        return (z2 / z1 if z1 > 0 else np.nan), z1, z2

    r0, z1_0, z2_0 = ratio(f0)
    rp, z1_p, z2_p = ratio(partner)

    if not (np.isfinite(r0) and np.isfinite(rp)) or min(r0, rp) <= 0:
        return {"ok": None, "reason": "line too weak in z1 or z2"}

    contrast = abs(np.log(r0 / rp))     # 0 => identical shapes
    return {
        "f0": f0, "partner": partner,
        "ratio_f0": r0, "ratio_partner": rp,
        "z1_f0": z1_0, "z2_f0": z2_0, "z1_partner": z1_p, "z2_partner": z2_p,
        "contrast": float(contrast),
        "distinct": bool(contrast > 0.22),   # ~25% difference in shape ratio
        "ok": True,
    }


def oscillator_modes(m1, m2, k01, k12) -> np.ndarray:
    """Natural frequencies [Hz] with the string held rigid.

    These are UPPER BOUNDS — the string is compliant, which lowers both
    modes by an amount that depends on the contact position and is not
    known a priori. They are useful for orientation but must NOT be used
    as a pass/fail criterion: a measured line 10-15% below a bound is
    entirely consistent with compliance.
    """
    K = np.array([[k01 + k12, -k12], [-k12, k12]])
    M = np.diag([m1, m2])
    w = np.sqrt(np.clip(np.sort(np.linalg.eigvals(np.linalg.solve(M, K)).real),
                        0.0, None))
    return w / (2 * np.pi)


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

    resolution = 1.0 / (t[-1] - t[0])
    combs = find_combs(lines, line_amps, f_pass, resolution=resolution)

    # z1/z2 spectra are needed for the mode-shape test below
    spectra = {}
    for name in ("z1", "z2"):
        if name in d.files:
            spectra[name] = detrended_spectrum(t, d[name], rate=rate,
                                               T_pass=T_pass)

    print(f"\n--- {len(combs)} Floquet comb(s) in {label} "
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
              f"are excited at once (offsets differ by {d01:.4f} Hz).")
        print("  Each has its OWN Re(lambda); the envelope fit returns whichever "
              "dominates.")

    # --- what kind of instability is each comb? ---
    p = getattr(mn, "TONGUE_P", 1)
    f_modes = oscillator_modes(m1, m2, k01, k12)
    print(f"\n--- Mode participation (p = {p}) ---")
    print("  A single mode with parametric sidebands and two modes in")
    print("  combination produce IDENTICAL line positions, so positions alone")
    print("  cannot tell them apart. What does: the mode shape |z2|/|z1|,")
    print("  which is constant across sidebands of one mode but differs")
    print("  between two genuinely distinct modes.\n")
    print(f"  Rigid-string modes (upper bounds, orientation only): "
          f"{f_modes[0]:.4f}, {f_modes[1]:.4f} Hz")

    if not spectra:
        print("\n  z1/z2 not stored in this cache file — cannot run the test.")
    for c_i, comb in enumerate(combs, 1):
        res = mode_shape_test(comb, spectra, f_pass, p=p, resolution=resolution)
        print(f"\n  Comb {c_i}: f0 = {comb['offset']:.4f} Hz, "
              f"partner = {p*f_pass - comb['offset']:.4f} Hz")
        if res is None:
            print("    (z1/z2 spectra unavailable)")
            continue
        if res.get("ok") is None:
            print(f"    inconclusive: {res['reason']}")
            continue
        print(f"    |z2|/|z1| at f0      = {res['ratio_f0']:.3f}  (arbitrary scale)")
        print(f"    |z2|/|z1| at partner = {res['ratio_partner']:.3f}  (same scale)")
        print(f"    shape contrast       = {res['contrast']:.3f} "
              f"(0 = identical mode shape; >0.22 = distinct)")
        if res["distinct"]:
            print("    => the two lines have DIFFERENT mode shapes: two distinct")
            print("       modes participate. Consistent with a COMBINATION "
                  "RESONANCE.")
        else:
            print("    => same mode shape at both lines: this is ONE mode with")
            print("       parametric sidebands, NOT a combination resonance.")
        for f_m, name in zip(f_modes, ("mode 1", "mode 2")):
            for f_obs, tag in ((comb["offset"], "f0"),
                               (p*f_pass - comb["offset"], "partner")):
                rel = (f_obs - f_m) / f_m * 100
                if abs(rel) < 25:
                    if rel < 2.0:
                        note = "plausible — string compliance lowers modes"
                    elif rel < 5.0:
                        note = "essentially at the bound"
                    else:
                        note = "ABOVE the rigid-string bound — cannot be this mode"
                    print(f"       {tag} vs {name}: {rel:+.1f}%  ({note})")

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
