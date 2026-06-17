from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import splu

from periodic_string.assembly import AssembledModel, assemble_model
from periodic_string.moving_load import displacement_at_load, load_shape_vector, wrap_load_position, contact_direction
from periodic_string.newmark import NewmarkIntegrator, NewmarkState


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
        Vertical displacement of the string at the contact point ``w_c(t) = N(t).T @ w``
    z_mass : numpy.ndarray
        Vertical position of the moving mass, ``z(t)``.
    contact_force : numpy.ndarray
        Contact spring force ``F(t) = K * (z(t) - w_c(t))``.
    model : AssembledModel
        Assembled finite-element model used in the simulation.
    """

    t: np.ndarray
    t_all: np.ndarray
    u_point: np.ndarray
    z_mass: np.ndarray
    contact_force: np.ndarray
    model: AssembledModel

def _static_initial_state(
    model: AssembledModel,
    external_force: np.ndarray,
    load_x_init: float,
) -> NewmarkState:
    """Compute the static-equilibrium initial state at the initial load position.

    Solves ``(K_static + K * d_init d_init.T) u_0 = f`` with gravity on the
    moving-mass DOF and the contact spring placed at ``load_x_init``. Returns
    a state with ``u_0`` as the displacement and zero velocity/acceleration.

    This removes the gravity-switch-on transient that would otherwise excite
    the high-frequency contact mode at ``sqrt(K / M)``.

    Parameters
    ----------
    model : AssembledModel
        Assembled finite-element model.
    external_force : numpy.ndarray
        Constant nodal force vector (gravity on the mass DOF).
    load_x_init : float
        Load position at ``t = 0``.

    Returns
    -------
    NewmarkState
        Initial state at static equilibrium with zero velocity and
        acceleration.
    """
    free = model.free_dofs
    d_init = contact_direction(
        model.node_x,
        load_x_init,
        model.catenary_length,
        model.n_dof,
        model.mass_dof,
    )
    d_init_free = d_init[free]
    d_init_sp = sparse.csr_matrix(d_init_free.reshape(-1, 1))
    contact_block = model.contact_stiffness * (d_init_sp @ d_init_sp.T)
    k_total_free = (model.stiffness[free, :][:, free] + contact_block).tocsc()

    u_0 = np.zeros(model.n_dof)
    u_0[free] = splu(k_total_free).solve(external_force[free])

    return NewmarkState(
        displacement=u_0,
        velocity=np.zeros(model.n_dof),
        acceleration=np.zeros(model.n_dof),
    )


def _static_initial_state(
    model: AssembledModel,
    external_force: np.ndarray,
    load_x_init: float,
) -> NewmarkState:
    """Compute the static-equilibrium initial state at the initial load position.

    Solves ``(K_static + K * d_init d_init.T) u_0 = f`` with gravity on the
    moving-mass DOF and the contact spring placed at ``load_x_init``. Returns
    a state with ``u_0`` as the displacement and zero velocity/acceleration.

    This removes the gravity-switch-on transient that would otherwise excite
    the high-frequency contact mode at ``sqrt(K / M)``.

    Parameters
    ----------
    model : AssembledModel
        Assembled finite-element model.
    external_force : numpy.ndarray
        Constant nodal force vector (gravity on the mass DOF).
    load_x_init : float
        Load position at ``t = 0``.

    Returns
    -------
    NewmarkState
        Initial state at static equilibrium with zero velocity and
        acceleration.
    """
    free = model.free_dofs
    d_init = contact_direction(
        model.node_x,
        load_x_init,
        model.catenary_length,
        model.n_dof,
        model.mass_dof,
    )
    d_init_free = d_init[free]
    d_init_sp = sparse.csr_matrix(d_init_free.reshape(-1, 1))
    contact_block = model.contact_stiffness * (d_init_sp @ d_init_sp.T)
    k_total_free = (model.stiffness[free, :][:, free] + contact_block).tocsc()

    u_0 = np.zeros(model.n_dof)
    u_0[free] = splu(k_total_free).solve(external_force[free])

    return NewmarkState(
        displacement=u_0,
        velocity=np.zeros(model.n_dof),
        acceleration=np.zeros(model.n_dof),
    )


def solve_moving_load(
    *,
    tension: float,
    damp_string: float,
    mass_per_length: float,
    kv: float,
    phi: float,
    omega_ref: float,
    contact_mass: float,
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
    """Simulate a moving point load on a periodic string with an
    auxiliary moving-mass DOF (uncoupled in this step).

    Assembles the string + moving-mass model, applies gravity on the
    mass DOF, integrates with Newmark's method including the moving
    contact spring as a rank-1 stiffness update at each step, and
    records the string displacement at contact, the mass position, and
    the contact force.

    Parameters
    ----------
    tension : float
        Tensile force in the string.
    damp_string : float
        Rayleigh-type damping factor on the string stiffness. 
        Set to 0 for an undamped string as in the model equation.
    mass_per_length : float
        Mass per unit length of the string.
    kv : float
        Vertical spring stiffness at cell boundaries.
    phi : float
        Loss factor of the complex support stiffness ``kv * (1 + i * phi)``.
    omega_ref : float
        Reference circular frequency [rad/s] at which the hysteretic
        support damping is converted to equivalent viscous damping,
        ``damp_rp = phi / omega_ref``.
    contact_mass : float
        Mass ``M`` of the moving oscillator [kg].
    contact_stiffness : float
        Contact spring stiffness ``K`` [N/m].
    element_length : float
        Length of each string element.
    n_elements_per_cell : int
        Number of string elements per periodic cell.
    n_nodes : int
        Total number of nodes in the mesh.
    catenary_length : float
        Total length of one periodic catenary span.
    dt : float
        Integration time step.
    velocity : float
        Load travel speed along the catenary. 
        If zero, the load is placed at mid-span.
    t_max : float
        End time of the simulation.
    dt_out : float or None, optional
        Output sampling interval. Defaults to ``dt``.
    show_progress : bool, optional
        If ``True``, display tqdm progress bars (default True).

    Returns
    -------
    SolverOutput
        Output time histories and the assembled model.
    """
    if dt_out is None:
        dt_out = dt

    damp_rp = phi / omega_ref

    model = assemble_model(
        tension=tension,
        damp_string=damp_string,
        mass_per_length=mass_per_length,
        kv=kv,
        damp_rp=damp_rp,
        contact_mass=contact_mass,
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

    g = 9.81  # gravitational acceleration [m/s^2]

    # Constant external force: gravity on the moving-mass DOF.
    external_force = np.zeros(model.n_dof)
    external_force[model.mass_dof] = model.mass[model.mass_dof, model.mass_dof] * g

    x_min = model.node_x.min()
    x_max = catenary_length

    # Static initial condition at t = 0: avoids the gravity-switch-on transient.
    load_x_init = 0.0 if velocity != 0.0 else 0.5 * catenary_length
    load_x_init = wrap_load_position(load_x_init, x_min, x_max)
    state = _static_initial_state(model, external_force, load_x_init)

    # Constant external force: gravity on the moving-mass DOF.
    external_force = np.zeros(model.n_dof)
    external_force[model.mass_dof] = model.mass[model.mass_dof, model.mass_dof] * g

    t_all = np.arange(0.0, t_max + 0.5 * dt, dt)
    output_stride = max(1, int(round(dt_out / dt)))
    t_out: list[float] = []
    u_point: list[float] = []
    z_mass: list[float] = []
    contact_force: list[float] = []

    x_min = model.node_x.min()
    x_max = catenary_length

    for step_index, time in enumerate(
        tqdm(t_all, desc="Newmark time integration", disable=not show_progress)
    ):
        if velocity != 0.0:
            load_x = velocity * time
        else:
            load_x = 0.5 * catenary_length

        load_x = wrap_load_position(load_x, x_min, x_max)
        d = contact_direction(model.node_x, load_x, x_max, model.n_dof, model.mass_dof)

        state = integrator.step(
            state=state,
            contact_direction=d,
            contact_stiffness=model.contact_stiffness,
            external_force=external_force,
            mass=model.mass,
            damping=model.damping,
        )

        if step_index % output_stride == 0:
            # w_c(t) = N(t).T @ w  — the string DOF entries of d are N(t)
            shape = d.copy()
            shape[model.mass_dof] = 0.0
            w_c = float(shape @ state.displacement)
            z = float(state.displacement[model.mass_dof])

            t_out.append(time)
            u_point.append(w_c)
            z_mass.append(z)
            contact_force.append(model.contact_stiffness * (z - w_c))

    return SolverOutput(
        t=np.asarray(t_out),
        t_all=t_all,
        u_point=np.asarray(u_point),
        z_mass=np.asarray(z_mass),
        contact_force=np.asarray(contact_force),
        model=model,
    )
