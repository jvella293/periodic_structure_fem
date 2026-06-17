# Created by RvL
"""Moving load on a periodic taut string."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from periodic_string.assembly import mesh_parameters
from periodic_string.solver import solve_moving_load

# --- time integration ---
dt = 1.0e-4
V = 50.0

# --- string ---
tension = 2.0e4
damp_string = 0.0 # no string damping in the PDE
m = 1.1 # mass per unit length of the string [kg/m] (CHECK! kg or kg/m)

# --- contact oscillator ---
M_mass = 5.0          # mass [kg], placeholder pantograph value
K_contact = 1.0e4     # contact spring stiffness [N/m]

# --- periodic section ---
spacing = 10.0
n_cells = 50
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
t_max = 2.0 

def main() -> None:
    """Run the default moving-load simulation and plot displacement."""
    print(
        "Mesh: "
        f"requested element length={element_length_requested:.6g} m, "
        f"effective={element_length:.6g} m, "
        f"{n_elements_per_cell} elements/cell, "
        f"{n_nodes} nodes"
    )

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
