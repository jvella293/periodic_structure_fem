# Created by RvL
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

Model selection (same names/semantics as the Floquet code; m1 is the
stability-plane coordinate in both cases):

  model = "2dof"          : m2 = mu * m1        (fixed mass ratio mu = m2/m1)
  model = "2dof_fixedM2"  : m2 = m2_fixed       (constant secondary mass)

(The Floquet "sdof" case corresponds to the old 1-DOF time-domain code.)

Input convention
----------------
The non-dimensional speed ``Vc = V / c`` is the control parameter; the
dimensional speed ``V`` is derived from the string wave speed
``c = sqrt(H / rhoA)``. Figure names and titles use V/c.

Features
--------
  * Parameter-hash caching: identical parameters are NOT re-simulated; the
    cached result is loaded and re-plotted instead. Set FORCE_RERUN = True
    after changing solver/newmark CODE (the hash tracks parameters only,
    not source code).
  * Figures saved as PNG (quick view, 300 dpi) and PDF (vector, for the
    thesis), with all parameters printed across the top and encoded in the
    filename. The dense |F_tr| trace is rasterised inside the PDF to keep it
    small while axes/text/peaks stay vector.
  * Newmark gamma/beta are imported from newmark.py so the filename always
    reflects what actually ran.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from periodic_string.assembly import mesh_parameters
from periodic_string.solver import solve_moving_load

# Surface the Newmark coefficients so the filename reflects what actually ran.
try:
    from periodic_string.newmark import NEWMARK_GAMMA, NEWMARK_BETA
except ImportError:
    NEWMARK_GAMMA = 0.5
    NEWMARK_BETA = 0.25 * (NEWMARK_GAMMA + 0.5) ** 2


# ======================================================================
# Parameters
# ======================================================================

# --- time integration ---
dt = 1e-4

# --- string ---
tension = 1.0e4
damp_string = 0.0   # no string damping in the PDE
m = 1.1             # mass per unit length of the string [kg/m]

# --- derived wave speed and dimensional velocity ---
c = np.sqrt(tension / m)     # string wave speed [m/s]
Vc = 0.2
V = Vc * c
V = 22.247460415730487
Vc = V/c

# --- contact oscillator: 2 DOF (names match the Floquet code) ---
model = "2dof"      # "2dof" (m2 = mu*m1) | "2dof_fixedM2" (m2 fixed in kg)
m1 = 86.69           # contact mass [kg] — stability-plane coordinate
mu = 0.5            # mass ratio m2/m1        (used only when model = "2dof")
m2_fixed = 50.0     # secondary mass [kg]     (used only when model = "2dof_fixedM2")

if model == "2dof":
    m2 = mu * m1
elif model == "2dof_fixedM2":
    m2 = m2_fixed
else:
    raise ValueError(f"unknown model {model!r}")

k01 = 1.0e4         # contact spring (string -- m1) [N/m]
k12 = 1.0e3         # secondary spring (m1 -- m2) [N/m]
c12 = 0.0           # secondary viscous damping [N s/m] (0 = undamped validation)

# --- periodic section ---
spacing = 10.0
n_cells = 800
element_length_requested = 0.05

# --- vertical supports at periodic positions ---
Kv = 4.0e3                                    # discrete support stiffness [N/m]
phi = 0                                        # support loss factor (0 = undamped validation)
omega_ref = 2.0 * np.pi * V / spacing          # support-passing frequency [rad/s]

# --- run length ---
t_max = 15

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
    spring engaged (string held rigid): eig of M^-1 K on

        K_osc = [[k01 + k12, -k12],     M_osc = diag(m1, m2)
                 [-k12,       k12]]

    These are the frequencies that can tune into parametric resonance
    with the support-passing frequency V / L.
    """
    k_osc = np.array([[k01 + k12, -k12], [-k12, k12]])
    m_osc = np.diag([m1, m2])
    eigvals = np.linalg.eigvals(np.linalg.solve(m_osc, k_osc))
    eigvals = np.sort(np.real(eigvals))
    eigvals = np.clip(eigvals, 0.0, None)   # guard tiny negative round-off
    return np.sqrt(eigvals) / (2.0 * np.pi)


def collect_params() -> dict:
    """Everything that affects the result. Any change => new hash => rerun.

    m1 is the control parameter; m2 is stored too so the hash is unique
    regardless of which model derived it.
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
        f"_g{NEWMARK_GAMMA:g}_dt{dt:g}_nc{n_cells}"
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
        tension=tension,
        damp_string=damp_string,
        mass_per_length=m,
        kv=Kv,
        phi=phi,
        omega_ref=omega_ref,
        head_mass=m1,
        frame_mass=m2,
        susp_stiffness=k12,
        susp_damping=c12,
        base_stiffness=0.0,   # m2 is ungrounded (matches the Floquet model)
        base_damping=0.0,
        contact_stiffness=k01,
        element_length=element_length,
        n_elements_per_cell=n_elements_per_cell,
        n_nodes=n_nodes,
        catenary_length=catenary_length,
        dt=dt,
        velocity=V,
        t_max=t_max,
    )
    np.savez(
        cache_file,
        t=output.t, u_point=output.u_point,
        z1=output.z_head, z2=output.z_frame,
        contact_force=output.contact_force,
        **params,
    )
    print(f"  cached -> {cache_file}")
    run_or_load._last_model = output.model
    return output.t, output.u_point, output.z_head, output.z_frame, output.contact_force


def growth_rate(t, contact_force, settle_periods=3.0):
    """Robust Re(lambda) from the UPPER envelope (local maxima).

    Fitting the peaks (not the raw |F|) ignores beat dips that would corrupt a
    single-line slope. Returns a dict including the uncertainty and a
    noise-gated verdict so a near-neutral point is not falsely signed.
    """
    env = np.abs(contact_force)
    T_period = spacing / V
    settle = t > settle_periods * T_period
    t_fit, env_fit = t[settle], env[settle]

    peak_idx, _ = find_peaks(env_fit)
    out = {
        "env": env, "init": env[0], "final": env[-1],
        "ratio_endpoints": env[-1] / max(env[0], 1e-300),
        "slope": None,
    }
    n10 = max(1, len(env) // 10)
    out["ratio_robust"] = env[-n10:].mean() / max(env[:n10].mean(), 1e-300)

    if peak_idx.size >= 3:
        t_pk = t_fit[peak_idx]
        env_pk = env_fit[peak_idx]
        good = env_pk > 0.0
        (slope, intercept), cov = np.polyfit(
            t_pk[good], np.log(env_pk[good]), 1, cov=True
        )
        slope_err = float(np.sqrt(cov[0, 0]))
        if abs(slope) < 2.0 * slope_err:
            verdict = "NEUTRAL (rate within noise)"
        elif slope > 0:
            verdict = "UNSTABLE"
        else:
            verdict = "stable"
        out.update(slope=float(slope), intercept=float(intercept),
                   slope_err=slope_err, verdict=verdict,
                   t_pk=t_pk, env_pk=env_pk, n_peaks=int(t_pk.size))
    else:
        out["verdict"] = f"too few peaks ({peak_idx.size}) - run longer"
    return out


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    params = collect_params()

    print(
        "Mesh: "
        f"requested element length={element_length_requested:.6g} m, "
        f"effective={element_length:.6g} m, "
        f"{n_elements_per_cell} elements/cell, {n_nodes} nodes"
    )

    # --- physics-derived diagnostics ---
    f_osc = oscillator_frequencies()          # both oscillator modes [Hz]
    f_max = f_osc[-1]
    T_min = 1.0 / f_max if f_max > 0 else np.inf
    f_pass = V / spacing

    # --- numerical-resolution diagnostics ---
    points_per_fast_period = T_min / dt
    load_advance_per_step = V * dt
    elements_per_step = load_advance_per_step / element_length

    # --- wake re-encounter horizon ---
    wake_safety_factor = 0.8
    t_wake = catenary_length / (c + V)
    t_wake_safe = wake_safety_factor * t_wake
    loops_in_run = V * t_max / catenary_length

    print("\n--- Physics ---")
    print(f"  Model                         = {model}"
          + (f"  (mu = m2/m1 = {mu:g})" if model == "2dof" else f"  (m2 fixed = {m2_fixed:g} kg)"))
    print(f"  Masses                        = m1 = {m1:g} kg, m2 = {m2:g} kg")
    print(f"  Wave speed c                  = {c:.2f} m/s")
    print(f"  V/c                           = {Vc:.3f}  (V = {V:.2f} m/s, "
          f"{'sub-critical' if Vc < 1 else 'super-critical'})")
    print(f"  Oscillator modes (k01 engaged)= {f_osc[0]:.2f} Hz, {f_osc[1]:.2f} Hz")
    print(f"  Support-passing frequency     = {f_pass:.2f} Hz")
    print(f"  f_i / f_pass                  = {f_osc[0]/f_pass:.3f}, {f_osc[1]/f_pass:.3f}"
          f"  (=n/2 => parametric tongues)")

    print("\n--- Numerical resolution ---")
    print(f"  Points per FASTEST osc period = {points_per_fast_period:.1f}  (want >= 20)")
    print(f"  Load advance per step         = {load_advance_per_step*1e3:.3f} mm")
    print(f"  Elements per step             = {elements_per_step:.3f}  (want < 1)")
    print(f"  Newmark gamma / beta          = {NEWMARK_GAMMA:g} / {NEWMARK_BETA:.4f}")

    print("\n--- Wake horizon ---")
    print(f"  Wake re-encounter at t        = {t_wake:.2f} s")
    print(f"  Safe horizon ({wake_safety_factor:.0%} margin)        = {t_wake_safe:.2f} s")
    print(f"  Requested t_max               = {t_max:.2f} s")
    print(f"  Load loops around domain      = {loops_in_run:.1f}")
    if t_max > t_wake_safe:
        print(f"  WARNING: t_max exceeds the {wake_safety_factor:.0%}-safe horizon by "
              f"{t_max - t_wake_safe:.2f} s.")
        print(f"  Reduce t_max to <= {t_wake_safe:.2f} s, or extend domain to "
              f">= {(c + V) * t_max / wake_safety_factor:.0f} m "
              f"(n_cells >= {int(np.ceil((c + V) * t_max / (wake_safety_factor * spacing)))}).")

    # --- run (or load from cache) ---
    t, u_point, z1, z2, contact_force = run_or_load(params)

    # structural sanity only available on a fresh run (model isn't cached)
    fem = getattr(run_or_load, "_last_model", None)
    if fem is not None:
        print(f"\nSprings: {fem.spring_nodes.size}")
        print(f"n_dof = {fem.n_dof}, expected {n_nodes + 2}")
        print(f"contact_dof (z1) = {fem.contact_dof}, frame_dof (z2) = {fem.frame_dof}")
        print(f"m1 on z1 DOF: {fem.mass[fem.contact_dof, fem.contact_dof]:.4g}")
        print(f"m2 on z2 DOF: {fem.mass[fem.frame_dof, fem.frame_dof]:.4g}")
        print(f"k12 coupling K[z1,z2]: {fem.stiffness[fem.contact_dof, fem.frame_dof]:.4g}"
              f" (expected {-k12:g})")

    # --- growth / decay ---
    g = growth_rate(t, contact_force)
    print("\n--- Perturbation response ---")
    print(f"  Initial |F_tr|                = {g['init']:.3e} N")
    print(f"  Final   |F_tr|                = {g['final']:.3e} N")
    print(f"  Envelope ratio (endpoints)    = {g['ratio_endpoints']:.3e}  (beat-phase sensitive)")
    print(f"  Envelope ratio (robust 10%)   = {g['ratio_robust']:.3e}")
    if g["slope"] is not None:
        print(f"  Peaks used                    = {g['n_peaks']}")
        print(f"  Growth rate Re(lambda)        = {g['slope']:+.4f} +/- {g['slope_err']:.4f} 1/s")
        print(f"  Verdict                       = {g['verdict']}")
    else:
        print(f"  {g['verdict']}")

    # --- figure ---
    model_str = (f"{model} (mu={mu:g})" if model == "2dof"
                 else f"{model} (m2={m2_fixed:g} kg)")
    param_text = (
        f"{model_str}: m1={m1:g} kg, m2={m2:g} kg   V/c={Vc:.3f} (V={V:.2f} m/s)\n"
        f"k01={k01:.0e}  k12={k12:.0e}  c12={c12:g}   phi={phi:g}   "
        f"gamma={NEWMARK_GAMMA:g}  beta={NEWMARK_BETA:.4f}   dt={dt:g} s\n"
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

    axes[2].semilogy(t, g["env"], lw=0.8, rasterized=True)
    if g["slope"] is not None:
        axes[2].semilogy(g["t_pk"], g["env_pk"], "r.", ms=5, label="envelope peaks")
        axes[2].semilogy(g["t_pk"], np.exp(g["intercept"] + g["slope"] * g["t_pk"]),
                         "k--", label=f"fit: {g['slope']:+.4f} 1/s")
        axes[2].legend()
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("|F_tr| [N]  (log)")
    axes[2].grid(True, which="both")

    fig.suptitle(param_text, fontsize=8, family="monospace")
    fig.subplots_adjust(top=0.88)

    FIG_DIR.mkdir(exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d_%H%M%S_") + readable_tag()
    png_path = FIG_DIR / f"{stem}.png"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"\nsaved: {png_path}")
    print(f"saved: {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()