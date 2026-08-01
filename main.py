"""Moving 2-DOF oscillator on a periodic taut string — time-domain validation.
 
Perturbation-stability run: gravity off, small seeded perturbation, the
growth/decay rate Re(lambda) of the contact-force envelope is the validation
quantity (compare against the Hill/Floquet dominant exponent at phi = 0).
 
Oscillator layout (matches the Floquet code conventions)
--------------------------------------------------------
    string ~~~ k01 ~~~ [m1, z1]   k01: contact spring (string -- m1)
                          |
                        k12       k12: secondary spring (m1 -- m2)
                          |
                       [m2, z2]   (ungrounded)
 
  model = "2dof"          : m2 = mu * m1        (fixed mass ratio mu = m2/m1)
  model = "2dof_fixedM2"  : m2 = m2_fixed       (constant secondary mass)
 
Growth-rate estimator
---------------------
Re(lambda) is fitted to the UPPER HULL of |F_tr| (rolling maximum), not to
find_peaks() maxima. find_peaks returns local maxima that sit inside the
beat nulls — three decades below the hull — and those drag a straight-line
log fit far more than a small Re(lambda) moves it.
 
For a 2T (subharmonic) instability the response lives at f_pass/2, so the
hull window must span SEVERAL subharmonic periods: HULL_PERIODS is measured
in support-passing periods, and 2T needs >= 4 (i.e. >= 2 subharmonic
periods). Set HULL_PERIODS = 2 only for a 1T (harmonic) tongue.
 
Input convention
----------------
Vc = V / c is the control parameter; c = sqrt(H / rhoA).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from periodic_string.assembly import mesh_parameters
from periodic_string.solver import solve_moving_load

try:
    from periodic_string.newmark import NEWMARK_GAMMA, NEWMARK_BETA
except ImportError:
    NEWMARK_GAMMA = 0.5
    NEWMARK_BETA = 0.25 * (NEWMARK_GAMMA + 0.5) ** 2


# ======================================================================
# Parameters
# ======================================================================

# --- time integration ---
# dt is limited by the moving contact (elements/step < 1), NOT by the
# oscillator modes: with a soft k01 the sqrt(k01/m1) mode is slow. See the
# "Numerical resolution" block printed at runtime. Verify any dt increase
# with the refinement protocol: halve dt, confirm Re(lambda) is unchanged.
dt = 5e-4

# --- string ---
tension = 2.0e4
damp_string = 0   # no string damping in the PDE
m = 1.1             # mass per unit length of the string [kg/m]

# --- derived wave speed and dimensional velocity ---
c = np.sqrt(tension / m)     # string wave speed [m/s]
#V = 22.247460415730487       # dimensional speed taken from the Floquet point
Vc = 0.22
#Vc = V / c
V = Vc * c

# --- contact oscillator: 2 DOF (names match the Floquet code) ---
model = "2dof"      # "2dof" (m2 = mu*m1) | "2dof_fixedM2" (m2 fixed in kg)
m1 = 147.757       # contact mass [kg] — CENTRE of the Floquet 2T tongue
mu = 0.5            # mass ratio m2/m1        (used only when model = "2dof")
m2_fixed = 50.0     # secondary mass [kg]     (used only when model = "2dof_fixedM2")


# --- Floquet tongue being validated (diagnostics only; nothing here feeds
#     the solver). Set TONGUE = None if you are not on a known tongue.
#
# TONGUE_TYPE selects which resonance condition is checked and reported:
#
#   "simple"            n * omega_i        = p * omega_pass
#                       Response sits at p*f_pass/n. For the classic 2T
#                       (subharmonic) tongue use p = 1, n = 2.
#   "combination_sum"   omega_1 + omega_2  = p * omega_pass
#   "combination_diff"  omega_2 - omega_1  = p * omega_pass
#
# For combination resonances the response has TWO components, at omega_1 and
# omega_2 — there is no single "response frequency", so none is reported.
TONGUE = (147.7429, 168.7682)      # (m1_lo, m1_hi) bracketing the tongue
TONGUE_TYPE = "combination_sum"    # "simple" | "combination_sum" | "combination_diff"
TONGUE_P = 1                       # harmonic p of the support-passing frequency
TONGUE_N = 2                       # subharmonic order n (TONGUE_TYPE="simple" only)
TONGUE_MODE = 0                    # which omega_i (0 or 1) for "simple"

# Backwards compatibility for reanalyse.py / diagnose.py, which read
# TONGUE_ORDER to size their expected-rate estimate.
TONGUE_ORDER = (TONGUE_N / TONGUE_P) if TONGUE_TYPE == "simple" else TONGUE_P
 
if model == "2dof":
    m2 = mu * m1
elif model == "2dof_fixedM2":
    m2 = m2_fixed
else:
    raise ValueError(f"unknown model {model!r}")

k01 = 1.0e5         # contact spring (string -- m1) [N/m]
k12 = 3.0e3         # secondary spring (m1 -- m2) [N/m]
c12 = 0.0           # secondary viscous damping [N s/m] (0 = undamped validation)

# --- periodic section ---
spacing = 15.0
n_cells = 1375
element_length_requested = 0.1

# --- vertical supports at periodic positions ---
Kv = 8.0e3
phi = 0                                        # support loss factor (0 = undamped validation)
omega_ref = 2.0 * np.pi * V / spacing

# --- run length ---
t_max = 100

# --- growth-rate estimator ---
SETTLE_PERIODS = 3.0    # support-passing periods discarded as transient
HULL_PERIODS = 4.0      # rolling-max window [support-passing periods]; >=4 for 2T

# --- cache / output control ---
FORCE_RERUN = False
CACHE_DIR = Path("cache")
FIG_DIR = Path("figures")

# --- derived mesh ---
element_length, n_elements_per_cell, n_nodes, catenary_length = mesh_parameters(
    spacing, element_length_requested, n_cells
)



# ======================================================================
# Helpers
# ======================================================================
 
def oscillator_frequencies() -> np.ndarray:
    """Natural frequencies [Hz] of the 2-DOF oscillator with the contact
    spring engaged and the string held rigid:
 
        K_osc = [[k01 + k12, -k12],     M_osc = diag(m1, m2)
                 [-k12,       k12]]
 
    These are UPPER BOUNDS on the true coupled frequencies (the string is
    compliant, which softens both modes).
    """
    k_osc = np.array([[k01 + k12, -k12], [-k12, k12]])
    m_osc = np.diag([m1, m2])
    eigvals = np.sort(np.linalg.eigvals(np.linalg.solve(m_osc, k_osc)).real)
    return np.sqrt(np.clip(eigvals, 0.0, None)) / (2.0 * np.pi)
 
 
def resonance_check(f_osc: np.ndarray, f_pass: float) -> dict:
    """Evaluate the resonance condition selected by TONGUE_TYPE.
 
    Returns the tuning ratio (which should be ~1 if the run is actually
    sitting on the tongue), the frequency that gets detuned as m1 varies,
    the response frequency where one exists, and the speed V that would
    tune the condition exactly.
 
    Note that f_osc are RIGID-STRING values and therefore upper bounds; the
    string's compliance lowers both modes, so a truly tuned V is slightly
    lower than the value reported here.
    """
    w1, w2 = 2 * np.pi * f_osc[0], 2 * np.pi * f_osc[1]
    w_pass = 2 * np.pi * f_pass
 
    if TONGUE_TYPE == "simple":
        w_i = (w1, w2)[TONGUE_MODE]
        f_detuned = TONGUE_N * w_i / (2 * np.pi)
        condition = (f"{TONGUE_N} * omega_{TONGUE_MODE+1} = {TONGUE_P} * omega_pass"
                     f"   ({TONGUE_N}T subharmonic)" if TONGUE_P == 1 else
                     f"{TONGUE_N} * omega_{TONGUE_MODE+1} = {TONGUE_P} * omega_pass")
        response = TONGUE_P * f_pass / TONGUE_N
    elif TONGUE_TYPE == "combination_sum":
        f_detuned = (w1 + w2) / (2 * np.pi)
        condition = f"omega_1 + omega_2 = {TONGUE_P} * omega_pass  (sum type)"
        response = None
    elif TONGUE_TYPE == "combination_diff":
        f_detuned = (w2 - w1) / (2 * np.pi)
        condition = f"omega_2 - omega_1 = {TONGUE_P} * omega_pass  (difference type)"
        response = None
    else:
        raise ValueError(f"unknown TONGUE_TYPE {TONGUE_TYPE!r}")
 
    ratio = (2 * np.pi * f_detuned) / (TONGUE_P * w_pass)
    # V that would tune the condition exactly (f_detuned is independent of V)
    V_tuned = f_detuned * spacing / TONGUE_P
 
    return {
        "condition": condition,
        "f_detuned": f_detuned,
        "response": response,
        "ratio": float(ratio),
        "tuned": bool(abs(ratio - 1.0) < 0.05),
        "V_tuned": float(V_tuned),
    }
 
 
def collect_params() -> dict:
    """Everything that affects the result. Any change => new hash => rerun.
 
    Estimator settings are deliberately EXCLUDED: they are post-processing,
    so changing them re-analyses the cached run instead of resimulating.
    """
    return {
        "Vc": Vc, "V": V, "tension": tension, "damp_string": damp_string, "m": m,
        "model": model, "m1": m1, "m2": m2,
        "k01": k01, "k12": k12, "c12": c12,
        "spacing": spacing, "n_cells": n_cells,
        "element_length_requested": element_length_requested,
        "Kv": Kv, "phi": phi, "dt": dt, "t_max": t_max,
        "gamma": NEWMARK_GAMMA, "beta": NEWMARK_BETA,
    }
 
 
def param_hash(params: dict) -> str:
    """Stable 12-char hash of the run parameters."""
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
 
 
def readable_tag() -> str:
    """Human-browsable filename stem encoding the key physics/numerics."""
    return (
        f"{model}_Vc{Vc:.3f}_m1_{m1:g}_m2_{m2:g}"
        f"_k01_{k01:.0e}_k12_{k12:.0e}_phi{phi:g}"
        f"_g{NEWMARK_GAMMA:g}_dt{dt:g}_el{element_length:g}_nc{n_cells}_t{t_max:g}"
    )
 
 
def run_or_load(params: dict):
    """Return (t, u_point, z1, z2, contact_force), from cache if available."""
    CACHE_DIR.mkdir(exist_ok=True)
    h = param_hash(params)
    cache_file = CACHE_DIR / f"run_{h}.npz"
 
    if cache_file.exists() and not FORCE_RERUN:
        print(f"Cache HIT ({h}) -> loading (set FORCE_RERUN=True to recompute).")
        d = np.load(cache_file)
        return d["t"], d["u_point"], d["z1"], d["z2"], d["contact_force"]
 
    reason = "FORCE_RERUN" if cache_file.exists() else "no cache"
    print(f"Cache MISS ({h}, {reason}) -> running simulation.")
    output = solve_moving_load(
        tension=tension, damp_string=damp_string, mass_per_length=m,
        kv=Kv, phi=phi, omega_ref=omega_ref,
        head_mass=m1, frame_mass=m2,
        susp_stiffness=k12, susp_damping=c12,
        base_stiffness=0.0, base_damping=0.0,   # m2 ungrounded, as in Floquet
        contact_stiffness=k01,
        element_length=element_length,
        n_elements_per_cell=n_elements_per_cell,
        n_nodes=n_nodes, catenary_length=catenary_length,
        dt=dt, velocity=V, t_max=t_max,
    )
    np.savez(
        cache_file,
        t=output.t, u_point=output.u_point,
        z1=output.z_head, z2=output.z_frame,
        contact_force=output.contact_force, **params,
    )
    print(f"  cached -> {cache_file}")
    run_or_load._last_model = output.model
    return output.t, output.u_point, output.z_head, output.z_frame, output.contact_force
 
 
def growth_rate(t, contact_force, settle_periods=SETTLE_PERIODS,
                hull_periods=HULL_PERIODS):
    """Beat-robust Re(lambda) fitted to the UPPER HULL of |F_tr|.
 
    A rolling maximum over ``hull_periods`` support-passing periods removes
    the zero crossings and the beat nulls, leaving the envelope. Reports a
    split-half consistency check (is this really a single exponential?) and
    a resolvability floor (a rate below ln(1.1)/T changes the envelope by
    less than 10% end-to-end and cannot be signed).
    """
    env = np.abs(np.asarray(contact_force, dtype=float))
    t = np.asarray(t, dtype=float)
 
    T_pass = spacing / V
    dt_out = float(np.median(np.diff(t)))
    win = max(3, int(round(hull_periods * T_pass / dt_out)))
    pad = win // 2
 
    # Rolling maximum via a strided view — O(n * win) but vectorised.
    padded = np.pad(env, (pad, pad), mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(padded, 2 * pad + 1)
    hull = strided.max(axis=1)[: env.size]
 
    keep = t > settle_periods * T_pass
    if pad > 0:
        keep[-pad:] = False          # rolling max is edge-biased here
    keep &= hull > 0.0
 
    out = {"env": env, "hull": hull, "slope": None}
    t_h, e_h = t[keep], hull[keep]
    if t_h.size < 10:
        out["verdict"] = "too few hull points - run longer"
        return out
 
    log_e = np.log(e_h)
    (slope, intercept), cov = np.polyfit(t_h, log_e, 1, cov=True)
    slope_err = float(np.sqrt(cov[0, 0]))
 
    half = t_h.size // 2
    s1 = float(np.polyfit(t_h[:half], log_e[:half], 1)[0])
    s2 = float(np.polyfit(t_h[half:], log_e[half:], 1)[0])
    # polyfit's error is optimistic (hull points are strongly correlated),
    # so the split-half test uses a relative tolerance with an absolute floor.
    consistent = abs(s1 - s2) < max(0.25 * max(abs(s1), abs(s2)), 0.01)
 
    T_win = t_h[-1] - t_h[0]
    rate_floor = float(np.log(1.1) / T_win)
 
    if abs(slope) < max(2.0 * slope_err, rate_floor):
        verdict = f"NEUTRAL (|rate| below resolvable {rate_floor:.4f} 1/s)"
    elif slope > 0:
        verdict = "UNSTABLE"
    else:
        verdict = "stable"
    if not consistent:
        verdict += "  [!] halves disagree - not a clean exponential"
 
    out.update(
        slope=float(slope), intercept=float(intercept), slope_err=slope_err,
        slope_first=s1, slope_second=s2, consistent=bool(consistent),
        rate_floor=rate_floor, verdict=verdict,
        t_hull=t_h, env_hull=e_h, n_hull=int(t_h.size),
        ratio_hull=float(e_h[-1] / e_h[0]), t_win=float(T_win),
    )
    return out
 
 
# ======================================================================
# Main
# ======================================================================
 
def main() -> None:
    params = collect_params()
 
    print(
        f"Mesh: requested element length={element_length_requested:.6g} m, "
        f"effective={element_length:.6g} m, "
        f"{n_elements_per_cell} elements/cell, {n_nodes} nodes"
    )
 
    f_osc = oscillator_frequencies()
    f_pass = V / spacing
    T_pass = 1.0 / f_pass
 
    points_per_fast_period = (1.0 / f_osc[-1]) / dt if f_osc[-1] > 0 else np.inf
    elements_per_step = V * dt / element_length
 
    wake_safety_factor = 0.8
    t_wake = catenary_length / (c + V)
    t_wake_safe = wake_safety_factor * t_wake
 
    print("\n--- Physics ---")
    print(f"  Model                         = {model}"
          + (f"  (mu = {mu:g})" if model == "2dof" else f"  (m2 = {m2_fixed:g} kg)"))
    print(f"  Masses                        = m1 = {m1:g} kg, m2 = {m2:g} kg")
    print(f"  Wave speed c                  = {c:.2f} m/s")
    print(f"  V/c                           = {Vc:.4f}  (V = {V:.4f} m/s)")
    print(f"  Oscillator modes (rigid str.) = {f_osc[0]:.4f} Hz, {f_osc[1]:.4f} Hz  (upper bounds)")
    print(f"  Support-passing frequency     = {f_pass:.4f} Hz  (T_pass = {T_pass:.4f} s)")
 
    res = resonance_check(f_osc, f_pass)
    print(f"  Resonance condition           = {res['condition']}")
    if res["response"] is not None:
        print(f"  Response frequency            = {res['response']:.4f} Hz  "
              f"(period {1/res['response']:.4f} s)")
    else:
        print(f"  Response frequencies          = {f_osc[0]:.4f} Hz AND "
              f"{f_osc[1]:.4f} Hz (two components — no single subharmonic)")
    print(f"  Tuning ratio (want ~1.000)    = {res['ratio']:.4f}"
          + ("" if res["tuned"] else
             f"   <-- OFF TUNE: try V ~ {res['V_tuned']:.4f} m/s"))
    if not res["tuned"]:
        print("      (rigid-string modes are upper bounds, so the true tuned V "
              "is slightly LOWER than this)")
 
    if TONGUE is not None:
        lo, hi = TONGUE
        centre, width = 0.5 * (lo + hi), hi - lo
        pos = (m1 - lo) / width
        inside = 0.0 < pos < 1.0
        print(f"\n--- Floquet tongue ({TONGUE_TYPE}, p = {TONGUE_P}) ---")
        print(f"  m1 bracket                    = [{lo:.3f}, {hi:.3f}] kg, "
              f"centre {centre:.3f}, width {width:.3f} kg ({width/centre*100:.2f}%)")
        print(f"  m1 = {m1:g} sits at {pos*100:.1f}% across "
              f"({'INSIDE' if inside else 'OUTSIDE — expect no growth'})")
        if inside:
            print(f"  Rate relative to tongue peak  ~ {2*np.sqrt(pos*(1-pos)):.2f}"
                  f"  (semicircular profile)")
 
        # Peak rate ~ tongue half-width measured in the DETUNED frequency.
        # With mu fixed, every oscillator frequency scales as m1^(-1/2), so
        # the relative half-width in frequency is half that in mass.
        # Combination resonances share the detuning between two modes, hence
        # the extra factor 1/2.
        omega_detuned = 2 * np.pi * res["f_detuned"]
        share = 0.5 if TONGUE_TYPE.startswith("combination") else 1.0
        rate_max = share * 0.5 * (0.5 * width / centre) * omega_detuned
        rate_here = rate_max * (2 * np.sqrt(pos * (1 - pos)) if inside else 0.0)
        print(f"  ESTIMATED Re(lambda)_max      ~ {rate_max:.4f} 1/s  "
              f"(order-of-magnitude only"
              + ("; combination type, halved" if share != 1.0 else "") + ")")
        if inside:
            print(f"  ESTIMATED Re(lambda) here     ~ {rate_here:.4f} 1/s"
                  f"  => x{np.exp(rate_here*t_max):.1f} over t_max = {t_max:g} s")
            print(f"  t_max for one decade          ~ {np.log(10)/max(rate_here,1e-12):.0f} s")
        print("  >>> Replace this estimate with the ACTUAL Floquet Re(lambda) "
              "when you have it.")
 
    print("\n--- Numerical resolution ---")
    print(f"  Points per FASTEST osc period = {points_per_fast_period:.1f}  (want >= 20)")
    print(f"  Elements per step             = {elements_per_step:.3f}  (want < 1)")
    print(f"  Newmark gamma / beta          = {NEWMARK_GAMMA:g} / {NEWMARK_BETA:.4f}")
    if NEWMARK_GAMMA > 0.5:
        print("  WARNING: gamma > 0.5 adds algorithmic damping, which biases "
              "Re(lambda) DOWNWARD. Use gamma = 0.5 for rate measurement.")
 
    print("\n--- Wake horizon ---")
    print(f"  Wake re-encounter at t        = {t_wake:.2f} s")
    print(f"  Safe horizon ({wake_safety_factor:.0%} margin)       = {t_wake_safe:.2f} s")
    print(f"  Requested t_max               = {t_max:.2f} s")
    if t_max > t_wake_safe:
        need = int(np.ceil((c + V) * t_max / (wake_safety_factor * spacing)))
        print(f"  WARNING: t_max exceeds the safe horizon. "
              f"Reduce t_max to <= {t_wake_safe:.2f} s or set n_cells >= {need}.")
 
    # --- run (or load from cache) ---
    t, u_point, z1, z2, contact_force = run_or_load(params)
 
    fem = getattr(run_or_load, "_last_model", None)
    if fem is not None:
        print(f"\nSprings: {fem.spring_nodes.size}")
        print(f"n_dof = {fem.n_dof}, expected {n_nodes + 2}")
        print(f"m1 on z1 DOF: {fem.mass[fem.contact_dof, fem.contact_dof]:.4g}, "
              f"m2 on z2 DOF: {fem.mass[fem.frame_dof, fem.frame_dof]:.4g}")
        print(f"k12 coupling K[z1,z2]: {fem.stiffness[fem.contact_dof, fem.frame_dof]:.4g}"
              f" (expected {-k12:g})")
 
    # --- growth / decay ---
    g = growth_rate(t, contact_force)
    print("\n--- Perturbation response (upper-hull fit) ---")
    if g["slope"] is not None:
        print(f"  Fit window                    = {g['t_win']:.2f} s, {g['n_hull']} hull points")
        print(f"  Hull ratio over window        = {g['ratio_hull']:.3e}")
        print(f"  Growth rate Re(lambda)        = {g['slope']:+.4f} +/- {g['slope_err']:.4f} 1/s")
        print(f"  Split-half (1st / 2nd)        = {g['slope_first']:+.4f} / "
              f"{g['slope_second']:+.4f} 1/s  "
              f"[{'consistent' if g['consistent'] else 'INCONSISTENT'}]")
        print(f"  Resolvable floor              = {g['rate_floor']:.4f} 1/s")
        print(f"  Verdict                       = {g['verdict']}")
    else:
        print(f"  {g['verdict']}")
 
    # --- figure ---
    model_str = (f"{model} (mu={mu:g})" if model == "2dof"
                 else f"{model} (m2={m2_fixed:g} kg)")
    param_text = (
        f"{model_str}: m1={m1:g} kg, m2={m2:g} kg   V/c={Vc:.4f} (V={V:.3f} m/s)\n"
        f"k01={k01:.0e}  k12={k12:.0e}  c12={c12:g}   phi={phi:g}   "
        f"gamma={NEWMARK_GAMMA:g}  beta={NEWMARK_BETA:.4f}   dt={dt:g} s  "
        f"el={element_length:g} m\n"
        f"H={tension:g}  rhoA={m:g}  ks={Kv:g}  L={spacing:g}  n_cells={n_cells}  "
        f"t_max={t_max:g}   Re(lambda)="
        + (f"{g['slope']:+.4f} 1/s  [{g['verdict']}]" if g['slope'] is not None else "n/a")
    )
 
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 9))
 
    axes[0].plot(t, u_point, label="w_c(t) string at contact")
    axes[0].plot(t, z1, label="z1(t) contact mass m1")
    axes[0].plot(t, z2, label="z2(t) secondary mass m2")
    axes[0].set_ylabel("Displacement [m]")
    axes[0].legend()
    axes[0].grid(True)
 
    axes[1].plot(t, contact_force)
    axes[1].set_ylabel("Contact force [N]")
    axes[1].grid(True)
 
    axes[2].semilogy(t, g["env"], lw=0.6, alpha=0.5, rasterized=True, label="|F_tr|")
    axes[2].semilogy(t, g["hull"], lw=1.2, color="r", label="upper hull (rolling max)")
    if g["slope"] is not None:
        axes[2].semilogy(g["t_hull"],
                         np.exp(g["intercept"] + g["slope"] * g["t_hull"]),
                         "k--", label=f"fit: {g['slope']:+.4f} 1/s")
    axes[2].legend(loc="lower right", fontsize=8)
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("|F_tr| [N]  (log)")
    axes[2].grid(True, which="both")
 
    fig.suptitle(param_text, fontsize=8, family="monospace")
    fig.subplots_adjust(top=0.88)
 
    FIG_DIR.mkdir(exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d_%H%M%S_") + readable_tag()
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved: {path}")
 
    plt.show()
 
 
if __name__ == "__main__":
    main()