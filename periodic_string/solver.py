from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from periodic_string.assembly import AssembledModel, assemble_model
from periodic_string.moving_load import displacement_at_load, load_shape_vector, wrap_load_position
from periodic_string.newmark import NewmarkIntegrator


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
        Vertical displacement under the load at each output time.
    model : AssembledModel
        Assembled finite-element model used in the simulation.
    """

    t: np.ndarray
    t_all: np.ndarray
    u_point: np.ndarray
    model: AssembledModel


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
    omega_p: float = 0.0,
    force: float = 1.0,
    dt_out: float | None = None,
    show_progress: bool = True,
) -> SolverOutput:
    """Simulate a moving point load on a periodic string with an
    auxiliary moving-mass DOF (uncoupled in this step).

    Assembles the string model with an extra DOF for the moving mass
    ``z(t)``, integrates with Newmark's method, and records vertical
    displacement at the load position. In this step the mass DOF is
    not coupled to the string; the contact spring is added later.

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
    omega_p : float, optional
        Parametric excitation circular frequency (default 0).
    force : float, optional
        Moving point load magnitude [N] (default 1).
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
    state = integrator.initial_state(model.n_dof)

    t_all = np.arange(0.0, t_max + 0.5 * dt, dt)
    output_stride = max(1, int(round(dt_out / dt)))
    t_out: list[float] = []
    u_point: list[float] = []

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
        shape = load_shape_vector(model.node_x, load_x, x_max, model.n_dof)
        state = integrator.step(
            state=state,
            load_shape=shape,
            force=force,
            omega_p=omega_p,
            time=time,
            mass=model.mass,
            damping=model.damping,
        )

        if step_index % output_stride == 0:
            t_out.append(time)
            u_point.append(displacement_at_load(shape, state.displacement))

    return SolverOutput(
        t=np.asarray(t_out),
        t_all=t_all,
        u_point=np.asarray(u_point),
        model=model,
    )
