# Created by RvL
"""Moving load on a periodic taut string."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from periodic_string.assembly import mesh_parameters
from periodic_string.solver import solve_moving_load

# --- time integration ---
dt = 0.0005
V = 50.0
F = 1.0  # moving point load magnitude [N]
omega_p = 0.0 # moving oscillating point load frequency - if set to 0 constant moving load!

# --- string ---
tension = 2.0e4
damp_string = 0.0 # no string damping in the PDE
m = 2.0 # mass per unit length of the string [kg/m] (CHECK! kg or kg/m)

# --- contact oscillator ---
M_mass = 5.0          # mass [kg], placeholder pantograph value
K_contact = 1.0e8     # contact spring stiffness [N/m]

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
t_max = 0.7 * n_cells * spacing / V


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
        omega_p=omega_p,
        force=F,
    )
    print(f"Springs: {output.model.spring_nodes.size}")

    print(f"n_dof = {output.model.n_dof}, expected {n_nodes + 1}")
    print(f"mass_dof = {output.model.mass_dof}, expected {n_nodes}")
    print(f"M on mass DOF: {output.model.mass[output.model.mass_dof, output.model.mass_dof]}")

    plt.figure()
    plt.plot(output.t, output.u_point)
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement under load [m]")
    plt.title("Moving load response")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
