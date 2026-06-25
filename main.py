# Created by RvL
"""Moving load on a periodic taut string."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from periodic_string.assembly import mesh_parameters
from periodic_string.solver import solve_moving_load


# --- time integration ---
dt = 1e-4 
V = 50

# --- string ---
tension = 2.0e4
damp_string = 0.0 # no string damping in the PDE
m = 1.1 # mass per unit length of the string [kg/m] (CHECK! kg or kg/m)

# --- contact oscillator ---
M_mass = 2.5          # mass [kg],
K_contact = 1.0e4     # contact spring stiffness [N/m]

# --- periodic section ---
spacing = 10.0
n_cells = 275 # increase to reduce wake re-encounter artifacts or reduce t_max to avoid them (try to make them equal to SAFE HORIZON)
element_length_requested = 0.05

# --- vertical supports at periodic positions ---
Kv =  4.0e3                                   # discrete support stiffness [N/m], = ek
phi = 1.0e-4                                  # support loss factor
omega_ref = 2.0 * np.pi * V / spacing         # support-passing frequency [rad/s]

# --- derived mesh ---
element_length, n_elements_per_cell, n_nodes, catenary_length = mesh_parameters(
    spacing, element_length_requested, n_cells
)
#t_max = 0.7 * n_cells * spacing / V
t_max = 6 

def main() -> None:
    """Run the default moving-load simulation and plot displacement."""
    print(
        "Mesh: "
        f"requested element length={element_length_requested:.6g} m, "
        f"effective={element_length:.6g} m, "
        f"{n_elements_per_cell} elements/cell, "
        f"{n_nodes} nodes"
    )

    # --- physics-derived diagnostics ---
    c = np.sqrt(tension / m)                      # string wave speed [m/s]
    mach = V / c                                  # sub/super-critical ratio
    omega_c = np.sqrt(K_contact / M_mass)         # contact mode [rad/s]
    f_c = omega_c / (2 * np.pi)                   # contact mode [Hz]
    T_c = 1.0 / f_c                               # contact period [s]
    f_pass = V / spacing                          # support-passing frequency [Hz]

    # --- numerical-resolution diagnostics ---
    points_per_contact_period = T_c / dt
    load_advance_per_step = V * dt
    elements_per_step = load_advance_per_step / element_length

    # --- wake re-encounter horizon (periodic domain) ---
    wake_safety_factor = 0.8   # use at most 80% of the wake horizon
    t_wake = catenary_length / (c + V)
    t_wake_safe = wake_safety_factor * t_wake
    loops_in_run = V * t_max / catenary_length


    print("\n--- Physics ---")
    print(f"  Wave speed c                  = {c:.2f} m/s")
    print(f"  Mach V/c                      = {mach:.3f}  ({'sub-critical' if mach < 1 else 'super-critical'})")
    print(f"  Contact mode                  = {f_c:.1f} Hz  (period {T_c*1e3:.2f} ms)")
    print(f"  Support-passing frequency     = {f_pass:.2f} Hz")

    print("\n--- Numerical resolution ---")
    print(f"  Points per contact period     = {points_per_contact_period:.1f}  (want >= 20)")
    print(f"  Load advance per step         = {load_advance_per_step*1e3:.3f} mm")
    print(f"  Elements per step             = {elements_per_step:.3f}  (want < 1)")

    print("\n--- Wake horizon ---")
    print(f"  Wake re-encounter at t        = {t_wake:.2f} s")
    print(f"  Safe horizon ({wake_safety_factor:.0%} margin)        = {t_wake_safe:.2f} s")
    print(f"  Requested t_max               = {t_max:.2f} s")
    print(f"  Load loops around domain      = {loops_in_run:.1f}")
    if t_max > t_wake_safe:
        print(f"  WARNING: t_max exceeds the {wake_safety_factor:.0%}-safe horizon by "
              f"{t_max - t_wake_safe:.2f} s.")
        print(f"  Either reduce t_max to <= {t_wake_safe:.2f} s, or")
        print(f"  extend domain to >= {(c + V) * t_max / wake_safety_factor:.0f} m "
              f"(n_cells >= {int(np.ceil((c + V) * t_max / (wake_safety_factor * spacing)))}).")
        print()

    output = solve_moving_load(
        tension=tension,
        damp_string=damp_string,
        mass_per_length=m,
        kv=Kv,
        phi=phi,
        omega_ref=omega_ref,
        contact_mass=M_mass,
        contact_stiffness=K_contact,
        element_length=element_length,
        n_elements_per_cell=n_elements_per_cell,
        n_nodes=n_nodes,
        catenary_length=catenary_length,
        dt=dt,
        velocity=V,
        t_max=t_max,
    )
    print(f"Springs: {output.model.spring_nodes.size}")

    print(f"n_dof = {output.model.n_dof}, expected {n_nodes + 1}")
    print(f"mass_dof = {output.model.mass_dof}, expected {n_nodes}")
    print(f"M on mass DOF: {output.model.mass[output.model.mass_dof, output.model.mass_dof]}")

    print(f"Mean gap (z - w_c): {np.mean(output.z_mass - output.u_point):.3e} m")
    print(f"Expected: {M_mass * 9.81 / K_contact:.3e} m")

    mask = output.t > 1.0   # skip the first 1 s of transient
    print(f"Mean contact force: {np.mean(output.contact_force[mask]):.2f} N")
    print(f"Expected (Mg): {M_mass * 9.81:.2f} N")


    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(output.t, output.u_point, label="w_c(t) string at contact")
    axes[0].plot(output.t, output.z_mass, label="z(t) mass")
    axes[0].set_ylabel("Displacement [m]")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(output.t, output.contact_force)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Contact force [N]")
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
