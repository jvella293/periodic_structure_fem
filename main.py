# Created by RvL
"""Moving load on a periodic Euler-Bernoulli beam."""

from __future__ import annotations

import matplotlib.pyplot as plt

from periodic_string.assembly import mesh_parameters
from periodic_string.solver import solve_moving_load

# --- time integration ---
dt = 0.005
V = 20.0
F = 1.0  # moving point load magnitude [N]
omega_p = 0.0 # moving oscillating point load frequency - if set to 0 constant moving load!

# --- beam (former rail) ---
EI = 25e9
damp_rail = 7.1429e-4
m = 1400.0

# --- periodic section ---
spacing = 16.0
n_cells = 50
element_length_requested = 0.05

# --- vertical and rotational springs at section boundaries ---
Kv = spacing * 28e6
Kt = 1e9
c_d = 20e3
damp_rp = c_d * spacing / Kv

# --- derived mesh ---
element_length, n_elements_per_cell, n_nodes, track_length = mesh_parameters(
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
        ei=EI,
        damp_rail=damp_rail,
        mass_per_length=m,
        kv=Kv,
        kt=Kt,
        damp_rp=damp_rp,
        element_length=element_length,
        n_elements_per_cell=n_elements_per_cell,
        n_nodes=n_nodes,
        track_length=track_length,
        dt=dt,
        velocity=V,
        t_max=t_max,
        omega_p=omega_p,
        force=F,
    )
    print(f"Springs: {output.model.spring_nodes.size}")

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
