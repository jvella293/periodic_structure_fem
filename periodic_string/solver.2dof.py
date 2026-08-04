from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from periodic_string.assembly import AssembledModel, assemble_model
from periodic_string.moving_load import wrap_load_position, contact_triplet
from periodic_string.newmark import NewmarkIntegrator, NewmarkState, ContactSolveCache


@dataclass(frozen=True)
class SolverOutput:
    """Time histories from a moving-load simulation.

    Attributes
    ----------
    t : numpy.ndarray
        Output times (subsampled by ``dt_out``).
    t_all : numpy.ndarray
        Full time vector used during integration.
    u_point : numpy.ndarray
        String displacement at the contact point, ``w_c(t) = N(t).T @ w``.
    z_head : numpy.ndarray
        Vertical position of the contact mass, ``z1(t)``.
    z_frame : numpy.ndarray
        Vertical position of the secondary mass, ``z2(t)``.
    contact_force : numpy.ndarray
        Contact spring force ``F(t) = K * (z1(t) - w_c(t))``.
    model : AssembledModel
        Assembled finite-element model used in the simulation.
    n_contact_solves : int
        Number of extra back-substitutions spent on ``A^-1 d`` (diagnostic:
        should be roughly the number of elements traversed, NOT the number
        of time steps).
    """

    t: np.ndarray
    t_all: np.ndarray
    u_point: np.ndarray
    z_head: np.ndarray
    z_frame: np.ndarray
    contact_force: np.ndarray
    model: AssembledModel
    n_contact_solves: int = 0


def solve_moving_load(
    *,
    tension: float,
    damp_string: float,
    mass_per_length: float,
    kv: float,
    phi: float,
    omega_ref: float,
    head_mass: float,
    frame_mass: float,
    susp_stiffness: float,
    susp_damping: float,
    base_stiffness: float,
    base_damping: float,
    contact_stiffness: float,
    element_length: float,
    n_elements_per_cell: int,
    n_nodes: int,
    catenary_length: float,
    dt: float,
    velocity: float,
    t_max: float,
    dt_out: float | None = None,
    show_progress: bool = True,
) -> SolverOutput:
    """Simulate a moving 2-DOF oscillator on a periodic string.

    Perturbation-stability formulation: no gravity (it cancels on
    subtraction), a small seeded perturbation on both oscillator DOFs, and
    the contact spring applied as a rank-1 stiffness update at each step.

    Parameters are as in the previous version; see AssembledModel for the
    oscillator layout (m1 on the contact spring, m2 below through k12).
    """
    if dt_out is None:
        dt_out = dt

    damp_rp = phi / omega_ref if omega_ref else 0.0

    model = assemble_model(
        tension=tension,
        damp_string=damp_string,
        mass_per_length=mass_per_length,
        kv=kv,
        damp_rp=damp_rp,
        head_mass=head_mass,
        frame_mass=frame_mass,
        susp_stiffness=susp_stiffness,
        susp_damping=susp_damping,
        base_stiffness=base_stiffness,
        base_damping=base_damping,
        contact_stiffness=contact_stiffness,
        element_length=element_length,
        n_elements_per_cell=n_elements_per_cell,
        n_nodes=n_nodes,
        catenary_length=catenary_length,
        show_progress=show_progress,
    )
    integrator = NewmarkIntegrator.from_model(
        mass=model.mass,
        stiffness=model.stiffness,
        damping=model.damping,
        free_dofs=model.free_dofs,
        dt=dt,
    )
    cache = ContactSolveCache(integrator, model.n_dof, model.contact_dof)

    external_force = np.zeros(model.n_dof)
    x_min, x_max = 0.0, catenary_length

    state = integrator.initial_state(model.n_dof)
    # Seed BOTH oscillator DOFs: a z1-only seed can under-excite a mode with
    # a near-node at the contact mass.
    state.velocity[model.contact_dof] = 1.0e-6
    state.velocity[model.frame_dof] = 1.0e-6

    t_all = np.arange(0.0, t_max + 0.5 * dt, dt)
    output_stride = max(1, int(round(dt_out / dt)))
    n_out = (t_all.size + output_stride - 1) // output_stride

    t_out = np.empty(n_out)
    u_point = np.empty(n_out)
    z_head = np.empty(n_out)
    z_frame = np.empty(n_out)
    contact_force = np.empty(n_out)
    n_written = 0

    for step_index, time in enumerate(
        tqdm(t_all, desc="Newmark time integration", disable=not show_progress)
    ):
        load_x = velocity * time if velocity != 0.0 else 0.5 * catenary_length
        load_x = wrap_load_position(load_x, x_min, x_max)

        dofs, values, left, right, w_left, w_right = contact_triplet(
            load_x, element_length, n_nodes, model.contact_dof
        )
        contact_solved = cache.get(left, right, w_left, w_right)

        state = integrator.step(
            state=state,
            contact_dofs=dofs,
            contact_values=values,
            contact_solved=contact_solved,
            contact_stiffness=model.contact_stiffness,
            external_force=external_force,
            mass=model.mass,
            damping=model.damping,
        )

        if step_index % output_stride == 0:
            u = state.displacement
            if not np.isfinite(u[model.contact_dof]):
                print(f"Non-finite state at t = {time:.4f} s — stopping early.")
                break
            # w_c(t) = N(t).T @ w — only two entries are nonzero.
            w_c = w_left * u[left] + w_right * u[right]
            z1 = u[model.contact_dof]

            t_out[n_written] = time
            u_point[n_written] = w_c
            z_head[n_written] = z1
            z_frame[n_written] = u[model.frame_dof]
            contact_force[n_written] = model.contact_stiffness * (z1 - w_c)
            n_written += 1

    return SolverOutput(
        t=t_out[:n_written],
        t_all=t_all,
        u_point=u_point[:n_written],
        z_head=z_head[:n_written],
        z_frame=z_frame[:n_written],
        contact_force=contact_force[:n_written],
        model=model,
        n_contact_solves=cache.n_solves,
    )
